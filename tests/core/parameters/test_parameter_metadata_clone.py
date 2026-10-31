"""A clone of a parameter tree owns its metadata dicts and shares their values.

``clone`` on a node, parameter, dated value, scale or bracket gives every
object in the copy its own metadata dict, holding the same values as the
source's. So a key set, updated, popped or replaced on one side never shows
on the other, while a nested value (a list of references, a dict) is one
object for both until a key is replaced. ``interpolate_parameters`` marks the
parameter it has interpolated, not the ``interpolation`` dict, for that reason.

``test_parameter_metadata_clone_property.py`` checks the same over random
trees and edits, against a reference clone that deep-copies metadata.
"""

import copy

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template import entities as template_entities
from policyengine_core.parameters import (
    Parameter,
    ParameterAtInstant,
    ParameterNode,
    interpolate_parameters,
    propagate_parameter_metadata,
)
from policyengine_core.reforms import Reform
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from tests.fixtures.parameter_metadata_clones import build, snapshot

# Each kind of object that carries metadata, and how to reach it in the
# tree from ``build``.
KINDS = {
    "node": lambda root: root.benefit,
    "parameter": lambda root: root.benefit.amount,
    "value": lambda root: root.benefit.amount.values_list[-1],
    "scale": lambda root: root.tax,
    "bracket": lambda root: root.tax.brackets[0],
    "bracket component": lambda root: root.tax.brackets[0].rate,
}

# Each way to change a metadata dict through its own methods, or by
# replacing it.
EDITS = {
    "set a key": lambda metadata: metadata.__setitem__("label", "Edited"),
    "replace a nested value": lambda metadata: metadata.__setitem__(
        "reference", [{"title": "Edited", "href": "https://example.gov/edited"}]
    ),
    "update": lambda metadata: metadata.update(label="Edited", added=[1]),
    "setdefault": lambda metadata: metadata.setdefault("added", {"x": 1}),
    "pop": lambda metadata: metadata.pop("reference", None),
    "delete": lambda metadata: metadata.__delitem__(next(iter(metadata))),
    "clear": lambda metadata: metadata.clear(),
}


def _metadata_of(item):
    if not item.metadata:
        # Give the object something to edit and to share.
        item.metadata.update(label="Label", reference=[{"title": "Act"}])
    return item.metadata


@pytest.mark.parametrize("edit", EDITS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("edited_side", ["clone", "source"])
def test_a_metadata_edit_stays_on_its_side(kind, edit, edited_side):
    source = build()
    _metadata_of(KINDS[kind](source))
    clone = source.clone()
    edited, other = (clone, source) if edited_side == "clone" else (source, clone)
    other_before = snapshot(other)
    edited_metadata = KINDS[kind](edited).metadata
    expected = copy.deepcopy(edited_metadata)
    EDITS[edit](expected)

    EDITS[edit](edited_metadata)

    assert KINDS[kind](edited).metadata == expected
    assert snapshot(other) == other_before


@pytest.mark.parametrize("kind", KINDS)
def test_a_clone_shares_nested_metadata_values_with_its_source(kind):
    """The documented rule: own dict, shared values (no deep copy)."""
    source = build()
    item = KINDS[kind](source)
    _metadata_of(item)
    cloned = KINDS[kind](source.clone())

    assert cloned.metadata is not item.metadata
    assert cloned.metadata == item.metadata
    for key, value in item.metadata.items():
        assert cloned.metadata[key] is value


@pytest.mark.parametrize(
    "make",
    [
        lambda: Parameter("p", {"2015-01-01": 1}),
        lambda: ParameterAtInstant("p", "2015-01-01", data=1),
        lambda: ParameterNode("n", data={"p": {"2015-01-01": 1}}),
    ],
    ids=["parameter", "value", "node"],
)
def test_a_clone_copies_metadata_as_copy_copy_does(make):
    """``None`` stays ``None`` and a dict subclass keeps its type, as before."""

    class Metadata(dict):
        pass

    item = make()
    item.metadata = None
    assert item.clone().metadata is None

    item.metadata = Metadata(label="Label")
    cloned = item.clone()
    assert type(cloned.metadata) is Metadata
    assert cloned.metadata == {"label": "Label"}
    assert cloned.metadata is not item.metadata


def test_interpolating_a_tree_interpolates_the_clone_it_holds():
    """The shape ``TaxBenefitSystem.__init__`` builds.

    It adds a clone of the tree as ``parameters.baseline``, then interpolates
    the whole tree. The clone shares the source's ``interpolation`` dict, so
    marking that dict complete in place would skip whichever copy came second.
    """
    root = build()
    root.add_child("baseline", root.clone())

    interpolate_parameters(root)

    for amount in (root.benefit.amount, root.baseline.benefit.amount):
        assert amount("2016-06-01") == 110
        assert amount("2018-06-01") == 130
        assert amount.interpolated


def test_a_tax_benefit_system_interpolates_its_baseline_copy(tmp_path):
    (tmp_path / "amount.yaml").write_text(
        "description: An interpolated amount.\n"
        "metadata:\n"
        "  interpolation:\n"
        "    interval: year\n"
        "values:\n"
        "  2015-01-01: 100\n"
        "  2019-01-01: 140\n"
    )

    class System(TaxBenefitSystem):
        entities = template_entities.entities
        parameters_dir = str(tmp_path)

    system = System()

    assert system.parameters.amount("2016-06-01") == 110
    assert system.parameters.baseline.amount("2016-06-01") == 110


def test_a_tax_benefit_system_interpolates_each_parameter_once(tmp_path):
    """``TaxBenefitSystem.__init__`` propagates metadata again after interpolating.

    That gives each parameter the node's ``interpolation`` dict again, so a
    mark on the parameter's own copy of the dict would be lost, and the next
    interpolation would add every entry a second time.
    """
    node = tmp_path / "node"
    node.mkdir()
    (node / "index.yaml").write_text(
        "metadata:\n"
        "  propagate_metadata_to_children: true\n"
        "  interpolation:\n"
        "    interval: year\n"
    )
    for name, start, end in (("a", 100, 140), ("b", 200, 280)):
        (node / f"{name}.yaml").write_text(
            "description: An amount.\n"
            "values:\n"
            f"  2015-01-01: {start}\n"
            f"  2019-01-01: {end}\n"
        )

    class System(TaxBenefitSystem):
        entities = template_entities.entities
        parameters_dir = str(tmp_path)

    system = System()
    before = snapshot(system.parameters)

    interpolate_parameters(system.parameters)

    assert snapshot(system.parameters) == before
    for tree in (system.parameters, system.parameters.baseline):
        assert len(tree.node.a.values_list) == 5
        assert tree.node.a("2016-06-01") == 110
        assert tree.node.b("2016-06-01") == 220


def test_interpolating_writes_no_metadata():
    root = propagate_parameter_metadata(build())
    root.add_child("baseline", root.clone())
    metadata_before = [
        (item, copy.deepcopy(item.metadata)) for item in root.get_descendants()
    ]

    interpolate_parameters(root)

    assert root.allowance.couple("2016-06-01") == 220
    for item, metadata in metadata_before:
        assert item.metadata == metadata


def test_every_parameter_a_node_propagates_interpolation_to_is_interpolated():
    """Propagation gives each child the node's ``interpolation`` dict.

    Marking that one dict complete skipped every child after the first.
    """
    root = ParameterNode(
        data={
            "node": {
                "metadata": {
                    "propagate_metadata_to_children": True,
                    "interpolation": {"interval": "year"},
                },
                "a": {"values": {"2015-01-01": 100, "2019-01-01": 140}},
                "b": {"values": {"2015-01-01": 200, "2019-01-01": 280}},
            }
        }
    )

    interpolate_parameters(propagate_parameter_metadata(root))

    assert root.node.a("2016-06-01") == 110
    assert root.node.b("2016-06-01") == 220


def test_interpolating_a_clone_leaves_its_source_uninterpolated():
    source = build()
    clone = source.clone()
    source_before = snapshot(source)

    interpolate_parameters(clone)

    assert clone.benefit.amount("2016-06-01") == 110
    assert snapshot(source) == source_before
    assert not source.benefit.amount.interpolated
    # And the source still interpolates when asked.
    interpolate_parameters(source)
    assert source.benefit.amount("2016-06-01") == 110


def test_a_parameter_interpolates_again_once_its_flag_is_cleared():
    root = ParameterNode(
        data={
            "amount": {
                "metadata": {"interpolation": {"interval": "year"}},
                "values": {"2015-01-01": 100, "2015-07-01": 200},
            }
        }
    )
    interpolate_parameters(root)
    assert len(root.amount.values_list) == 2

    root.amount.metadata["interpolation"] = {"interval": "month"}
    root.amount.interpolated = False
    interpolate_parameters(root)

    assert len(root.amount.values_list) == 7
    assert root.amount("2015-06-01") == pytest.approx(100 + 100 * 5 / 6)


def test_a_completed_key_in_the_metadata_still_stops_interpolation():
    """A tree pickled by an older Core carries the key its interpolation wrote."""
    root = ParameterNode(
        data={
            "amount": {
                "metadata": {"interpolation": {"interval": "year", "completed": True}},
                "values": {"2015-01-01": 100, "2019-01-01": 140},
            }
        }
    )

    interpolate_parameters(root)

    assert len(root.amount.values_list) == 2


def test_a_parameter_that_skipped_init_interpolates():
    """A subclass can skip ``Parameter.__init__``.

    So can a parameter pickled before ``interpolated`` existed. Neither has
    the attribute on the instance.
    """
    source = build().benefit.amount
    parameter = Parameter.__new__(Parameter)
    parameter.__dict__.update(
        {key: value for key, value in vars(source).items() if key != "interpolated"}
    )
    root = ParameterNode(data={})
    root.add_child("amount", parameter)

    interpolate_parameters(root)

    assert parameter("2016-06-01") == 110


@pytest.mark.parametrize(
    "copy_tree",
    [lambda tree: tree.clone(), copy.deepcopy],
    ids=["clone", "deepcopy"],
)
def test_a_copy_of_an_interpolated_tree_is_not_interpolated_again(copy_tree):
    source = interpolate_parameters(build())
    clone = copy_tree(source)
    values_before = snapshot(clone)

    interpolate_parameters(clone)

    assert snapshot(clone) == values_before


@pytest.mark.parametrize("edited_side", ["clone", "source"])
def test_propagating_metadata_on_one_side_leaves_the_other_alone(edited_side):
    source = build()
    clone = source.clone()
    edited, other = (clone, source) if edited_side == "clone" else (source, clone)
    other_before = snapshot(other)

    propagate_parameter_metadata(edited)

    assert edited.benefit.amount.metadata["notes"] == {"sources": ["benefit.pdf"]}
    assert snapshot(other) == other_before


@pytest.mark.parametrize("edited_side", ["clone", "source"])
def test_tax_benefit_system_clones_keep_their_own_metadata(edited_side):
    source = CountryTaxBenefitSystem()
    clone = source.clone()
    edited, other = (clone, source) if edited_side == "clone" else (source, clone)
    other_before = snapshot(other.parameters)
    tree = edited.parameters

    tree.taxes.income_tax_rate.metadata["label"] = "Edited"
    tree.taxes.income_tax_rate.values_list[0].metadata["note"] = "Edited"
    tree.taxes.social_security_contribution.metadata.pop("reference", None)
    tree.benefits.metadata.update(label="Edited", reference=[])

    assert snapshot(other.parameters) == other_before
    assert tree.taxes.income_tax_rate.metadata["label"] == "Edited"


def test_a_reform_that_edits_metadata_leaves_its_baseline_alone():
    baseline = CountryTaxBenefitSystem()
    baseline_before = snapshot(baseline.parameters)

    class EditMetadata(Reform):
        def apply(self):
            rate = self.parameters.taxes.income_tax_rate
            rate.metadata["label"] = "Reformed"
            rate.values_list[0].metadata["note"] = "Reformed"
            rate.update(period="year:2017:1", value=0.3)

    reformed = EditMetadata(baseline)

    assert reformed.parameters.taxes.income_tax_rate.metadata["label"] == "Reformed"
    assert snapshot(baseline.parameters) == baseline_before


def test_cloning_a_parameter_tree_deep_copies_no_metadata(monkeypatch):
    """The copy that dominated the cost of cloning a large tree is gone."""
    tree = CountryTaxBenefitSystem().parameters
    calls = []
    deepcopy = copy.deepcopy

    def counting_deepcopy(*args, **kwargs):
        calls.append(args[0])
        return deepcopy(*args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", counting_deepcopy)
    clone = tree.clone()

    assert calls == []
    assert snapshot(clone) == snapshot(tree)
