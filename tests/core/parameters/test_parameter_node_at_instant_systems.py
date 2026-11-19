"""Nodes at instant that resolve children on first read, as reforms and
country packages use them.

The sharing tests follow how policyengine-us's ``spm.py`` handles parameter
trees and their caches: a shallow copy of the root with its own cache for a
traced request, systems that share one ``_parameters_at_instant_cache``, path
copies that share parameter values, and resetting every cache after editing
values in place.
"""

import copy

import pytest

from policyengine_core.parameters import (
    Parameter,
    ParameterNode,
    ParameterNodeAtInstant,
    ParameterScale,
)
from policyengine_core.reforms import Reform
from policyengine_core.tracers import TracingParameterNodeAtInstant
from tests.fixtures.parameter_nodes_at_instant import (
    build_tree,
    materialise,
    ordered,
    snapshot,
)

INSTANT = "2017-03-01"
SYSTEM_INSTANT = "2017-01-01"


# Reforms, on the country template.


def test_a_reform_reads_its_update_and_the_baseline_keeps_its_values(
    isolated_tax_benefit_system,
):
    system = isolated_tax_benefit_system
    held = system.parameters(SYSTEM_INSTANT)
    assert held.taxes.income_tax_rate == 0.15

    class RaiseRate(Reform):
        def apply(self):
            self.parameters.taxes.income_tax_rate.update(period="2017", value=0.3)

    reformed = RaiseRate(system)

    assert reformed.parameters(SYSTEM_INSTANT).taxes.income_tax_rate == 0.3
    assert reformed.get_parameters_at_instant(SYSTEM_INSTANT).taxes.income_tax_rate == (
        0.3
    )
    assert system.parameters(SYSTEM_INSTANT).taxes.income_tax_rate == 0.15
    # A child of the held baseline node read only now reads the baseline.
    assert held.benefits.basic_income == 600


def test_a_reform_that_reads_before_it_updates_reads_the_update(
    isolated_tax_benefit_system,
):
    class ReadThenRaise(Reform):
        def apply(self):
            assert self.parameters(SYSTEM_INSTANT).taxes.income_tax_rate == 0.15
            self.parameters.taxes.income_tax_rate.update(period="2017", value=0.3)
            assert self.parameters(SYSTEM_INSTANT).taxes.income_tax_rate == 0.3

    reformed = ReadThenRaise(isolated_tax_benefit_system)

    assert reformed.parameters(SYSTEM_INSTANT).taxes.income_tax_rate == 0.3


def test_a_reform_that_adds_a_parameter_after_reading_reads_it(
    isolated_tax_benefit_system,
):
    class AddAllowance(Reform):
        def apply(self):
            self.parameters(SYSTEM_INSTANT).taxes.income_tax_rate
            self.parameters.taxes.add_child(
                "allowance",
                Parameter("taxes.allowance", data={"values": {"2010-01-01": 100}}),
            )

    reformed = AddAllowance(isolated_tax_benefit_system)

    assert reformed.parameters(SYSTEM_INSTANT).taxes.allowance == 100
    assert reformed.get_parameters_at_instant(SYSTEM_INSTANT).taxes.allowance == 100


def test_modify_parameters_copies_partly_read_nodes_onto_the_reform_tree(
    isolated_tax_benefit_system,
):
    system = isolated_tax_benefit_system
    # The baseline caches a node for ``benefits`` whose children are not read.
    system.parameters(SYSTEM_INSTANT).benefits

    def raise_rate(parameters):
        parameters.taxes.income_tax_rate.update(period="2017", value=0.3)
        return parameters

    class RaiseRate(Reform):
        def apply(self):
            self.modify_parameters(raise_rate)

    reformed = RaiseRate(system)
    # A change to the baseline after the reform was built stays in the
    # baseline, including for the copied nodes' children not read yet.
    system.parameters.benefits.basic_income.update(period="2017", value=700)

    reform_node = reformed.get_parameters_at_instant(SYSTEM_INSTANT)
    assert reform_node.taxes.income_tax_rate == 0.3
    assert reform_node.benefits.basic_income == 600
    assert system.parameters(SYSTEM_INSTANT).benefits.basic_income == 700
    assert system.parameters(SYSTEM_INSTANT).taxes.income_tax_rate == 0.15


# Sharing trees and caches, as policyengine-us's spm.py does.


class RecordingTracer:
    def __init__(self):
        self.accesses = []

    def record_parameter_access(self, name, period, branch_name, value):
        self.accesses.append((name, period, branch_name, value))


def test_a_shallow_copy_of_the_root_with_its_own_cache_reads_like_the_original():
    tree = build_tree()
    original = tree(INSTANT)
    original.group.x
    # As ``isolate_parameter_tracing``: the copy shares the children and gets
    # its own cache and tracer.
    private = copy.copy(tree)
    private._at_instant_cache = {}
    private.trace = True
    private.tracer = RecordingTracer()
    private.branch_name = "request"

    traced = private(INSTANT)
    rate = traced.group.x

    assert isinstance(traced, TracingParameterNodeAtInstant)
    node = traced.parameter_node_at_instant
    assert ordered(materialise(node)) == ordered(snapshot(tree, INSTANT))
    # The children's caches are shared, as when every child was built up front.
    assert node.group is original.group
    assert rate == 0.1
    assert private.tracer.accesses == [("group.x", INSTANT, "request", 0.1)]
    assert tree._at_instant_cache[INSTANT] is original
    assert not isinstance(tree(INSTANT), TracingParameterNodeAtInstant)


def test_systems_sharing_one_cache_read_the_same_nodes(
    isolated_tax_benefit_system,
):
    system = isolated_tax_benefit_system
    policy = system.clone()
    policy.parameters = system.parameters
    policy._parameters_at_instant_cache = system._parameters_at_instant_cache

    from_system = system.get_parameters_at_instant(SYSTEM_INSTANT)
    from_policy = policy.get_parameters_at_instant(SYSTEM_INSTANT)

    assert from_policy is from_system
    assert from_policy.taxes.income_tax_rate == 0.15
    assert from_system.taxes is from_policy.taxes
    assert ordered(materialise(from_policy)) == ordered(
        snapshot(system.parameters, SYSTEM_INSTANT)
    )


def path_copy(node, parent=None):
    """Copy the parameter nodes and share the dated values, as policyengine-us's
    ``copy_parameter_nodes`` does."""
    copied = object.__new__(type(node))
    copied.__dict__.update(node.__dict__)
    copied.metadata = dict(node.metadata)
    if isinstance(node, Parameter):
        copied.values_list = list(node.values_list)
    elif isinstance(node, ParameterScale):
        copied.brackets = [path_copy(bracket, copied) for bracket in node.brackets]
    else:
        copied.children = {
            name: path_copy(child, copied) for name, child in node.children.items()
        }
        for name, child in copied.children.items():
            setattr(copied, name, child)
        copied._at_instant_cache = {}
    copied.parent = parent
    return copied


def test_a_path_copy_reads_like_the_original_and_keeps_its_updates():
    tree = build_tree()
    held = tree(INSTANT)
    held.group.x

    copied = path_copy(tree)

    assert ordered(materialise(copied(INSTANT))) == ordered(snapshot(tree, INSTANT))
    copied.flat_rate.update(period="2017", value=0.5)
    copied.group.x.update(period="2017", value=0.6)
    assert copied(INSTANT).flat_rate == 0.5
    assert copied(INSTANT).group.x == 0.6
    assert held.flat_rate == 0.1
    assert tree(INSTANT).flat_rate == 0.1
    assert tree(INSTANT).group.x == 0.1


def clear_every_cache(system_or_tree):
    root = getattr(system_or_tree, "parameters", system_or_tree)
    if root is not system_or_tree:
        system_or_tree._parameters_at_instant_cache.clear()
    for node in (root, *root.get_descendants()):
        cache = getattr(node, "_at_instant_cache", None)
        if cache is not None:
            cache.clear()


def test_values_edited_in_place_are_read_once_every_cache_is_cleared(
    isolated_tax_benefit_system,
):
    system = isolated_tax_benefit_system
    system.get_parameters_at_instant(SYSTEM_INSTANT).taxes.income_tax_rate
    system.parameters(SYSTEM_INSTANT).benefits

    for value in system.parameters.taxes.income_tax_rate.values_list:
        value.value = 0.5
    system.parameters.benefits.basic_income.values_list[0].value = 900
    clear_every_cache(system)

    for node in (
        system.get_parameters_at_instant(SYSTEM_INSTANT),
        system.parameters(SYSTEM_INSTANT),
    ):
        assert node.taxes.income_tax_rate == 0.5
        assert node.benefits.basic_income == 900


def test_values_edited_in_place_are_read_once_the_parents_cache_is_cleared():
    # As policyengine-us's per-capita uprating: edit, then clear the caches
    # above the edited parameter.
    tree = build_tree()
    tree(INSTANT).group.x
    tree(INSTANT).flat_rate

    tree.group.x.values_list[0].value = 0.7
    tree.group.x.parent.clear_parent_cache()

    assert tree(INSTANT).group.x == 0.7
    assert tree.group(INSTANT).x == 0.7


# Instants that are not dates.


def test_a_tree_with_a_scale_cannot_be_read_at_none_as_before():
    tree = build_tree()

    with pytest.raises(ValueError):
        tree(None)

    assert set(tree._at_instant_cache) == set()


def test_a_tree_without_a_scale_reads_at_none_as_before():
    tree = ParameterNode(
        "",
        data={
            "a": {"values": {"2010-01-01": 1, "2020-01-01": 2}},
            "b": {"c": {"values": {"2015-01-01": 3}}},
        },
    )

    node = tree(None)

    assert isinstance(node, ParameterNodeAtInstant)
    assert ordered(materialise(node)) == ordered(snapshot(tree, "None"))


# Updates below a scale bracket.

SCALE_REFORM = {
    "taxes.social_security_contribution[1].rate": {"2017-01-01.2017-12-31": 0.5}
}
SALARY_SITUATION = {
    "persons": {"bill": {"salary": {"2017-01": 20000}}},
    "households": {"h": {"parents": ["bill"]}},
}


def contribution_after_reform(read_the_scale_first):
    from policyengine_core.country_template import CountryTaxBenefitSystem
    from policyengine_core.simulations import SimulationBuilder

    simulation = SimulationBuilder().build_from_entities(
        CountryTaxBenefitSystem(), SALARY_SITUATION
    )
    simulation.calculate("income_tax", "2017-01")
    if read_the_scale_first:
        simulation.calculate("social_security_contribution", "2017-01")
    simulation.apply_reform(SCALE_REFORM)
    return simulation.calculate("social_security_contribution", "2017-01")[0]


def test_a_reform_of_a_bracket_reads_the_same_whatever_was_calculated_first():
    # ``Parameter.update`` on a bracket's rate clears the caches above the
    # scale, because the bracket's ``parent`` is its scale.
    assert contribution_after_reform(False) == contribution_after_reform(True)


@pytest.mark.parametrize("read_the_scale_first", [False, True])
def test_nodes_requested_after_a_bracket_update_read_it(read_the_scale_first):
    tree = build_tree()
    tree(INSTANT).flat_rate
    if read_the_scale_first:
        tree(INSTANT).scale

    tree.scale.brackets[1].rate.update(period="2017", value=0.9)

    assert list(tree(INSTANT).scale.rates) == [0.1, 0.9]
    assert list(tree.scale(INSTANT).rates) == [0.1, 0.9]


# Updates below a node that homogenization put in place of a missing
# breakdown value.


def homogenized_tree():
    from policyengine_core.parameters.operations.homogenize_parameters import (
        homogenize_parameter_structures,
    )

    tree = ParameterNode(
        "",
        data={
            "rates": {
                "metadata": {"breakdown": [["A", "B"], ["X", "Y"]]},
                "A": {
                    "X": {"values": {"2010-01-01": 1}},
                    "Y": {"values": {"2010-01-01": 2}},
                },
            },
        },
    )
    homogenize_parameter_structures(tree, {})
    return tree


def test_homogenization_links_the_nodes_it_creates_to_their_parents():
    tree = homogenized_tree()

    created = tree.rates.children["B"]

    assert created.parent is tree.rates
    assert created.children["X"].parent is created


@pytest.mark.parametrize("read_the_leaf_first", [False, True])
def test_nodes_requested_after_an_update_below_a_homogenized_node_read_it(
    read_the_leaf_first,
):
    tree = homogenized_tree()
    node = tree(INSTANT)
    node.rates.A.X
    if read_the_leaf_first:
        node.rates["B"]["X"]

    tree.rates.children["B"].children["X"].update(period="2017", value=50)

    assert tree(INSTANT).rates["B"]["X"] == 50


def test_a_bracket_update_marks_the_scale_and_the_nodes_above_it_modified():
    # As in a cloned tree, where brackets were already linked to their scale.
    tree = build_tree()

    tree.scale.brackets[1].rate.update(period="2017", value=0.9)

    assert tree.scale.brackets[1].modified
    assert tree.scale.modified
    assert tree.modified
