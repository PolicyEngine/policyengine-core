"""``copy_paths`` copies only the nodes on the paths it is given."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import Parameter
from policyengine_core.parameters.path_copy import copy_paths
from policyengine_core.reforms import set_parameter

SYSTEM = CountryTaxBenefitSystem()
INSTANTS = ["2012-06-01", "2015-01-01", "2017-01-01", "2020-01-01", "2035-01-01"]


def _leaf_values(tree, instant):
    """Every parameter's value at ``instant``, by name."""
    return {
        descendant.name: descendant(instant)
        for descendant in tree.get_descendants()
        if isinstance(descendant, Parameter)
    }


def _set(tree, path, value):
    return set_parameter(path, value, period="year:2000:40", return_modifier=True)(tree)


def _leaf_paths(tree):
    """A path to every parameter of ``tree`` in ``set_parameter``'s syntax."""
    paths = []

    def visit(node, prefix):
        if isinstance(node, Parameter):
            paths.append(prefix)
        elif hasattr(node, "brackets"):
            for index, bracket in enumerate(node.brackets):
                visit(bracket, f"{prefix}[{index}]")
        else:
            for name, child in node.children.items():
                visit(child, f"{prefix}.{name}" if prefix else name)

    visit(tree, "")
    return [path for path in paths if not path.startswith("baseline.")]


LEAF_PATHS = _leaf_paths(SYSTEM.parameters)


def test_only_the_nodes_on_the_path_are_new():
    original = SYSTEM.parameters.clone()
    copied = copy_paths(original, ["taxes.income_tax_rate"])

    assert copied is not original
    assert copied.taxes is not original.taxes
    assert copied.taxes.income_tax_rate is not original.taxes.income_tax_rate
    assert copied.benefits is original.benefits
    assert copied.taxes.housing_tax is original.taxes.housing_tax
    assert copied.children["taxes"] is copied.taxes
    assert copied.taxes.parent is copied
    assert copied.taxes.income_tax_rate.parent is copied.taxes


def test_a_bracket_path_copies_the_scale_and_the_bracket():
    original = SYSTEM.parameters.clone()
    path = "taxes.social_security_contribution[1].rate"
    copied = copy_paths(original, [path])

    scale = copied.taxes.social_security_contribution
    original_scale = original.taxes.social_security_contribution
    assert scale is not original_scale
    assert scale.brackets[1] is not original_scale.brackets[1]
    assert scale.brackets[1].parent is scale
    assert scale.brackets[0] is original_scale.brackets[0]
    assert scale.brackets[1].threshold is original_scale.brackets[1].threshold

    _set(copied, path, 0.5)
    assert copied.get_child(path)("2020-01-01") == 0.5
    assert original.get_child(path)("2020-01-01") == 0.06


def test_updating_the_copy_leaves_the_original_and_its_caches_alone():
    original = SYSTEM.parameters.clone()
    warm_node = original.get_at_instant("2017-01-01")
    warm_benefits = original.benefits.get_at_instant("2017-01-01")

    copied = copy_paths(original, ["taxes.income_tax_rate"])
    _set(copied, "taxes.income_tax_rate", 0.3)

    assert copied.taxes.income_tax_rate("2017-01-01") == 0.3
    assert copied.get_at_instant("2017-01-01").taxes.income_tax_rate == 0.3
    assert original.taxes.income_tax_rate("2017-01-01") == 0.15
    assert original.get_at_instant("2017-01-01") is warm_node
    assert warm_node.taxes.income_tax_rate == 0.15
    # A branch the change did not touch keeps the values it cached.
    assert copied.benefits.get_at_instant("2017-01-01") is warm_benefits


def test_paths_with_a_common_prefix_share_their_copies():
    original = SYSTEM.parameters.clone()
    copied = copy_paths(
        original, ["taxes.housing_tax.rate", "taxes.housing_tax.minimal_amount"]
    )
    housing_tax = copied.taxes.housing_tax
    assert housing_tax is not original.taxes.housing_tax
    assert housing_tax.rate.parent is housing_tax
    assert housing_tax.minimal_amount.parent is housing_tax


def test_an_unknown_path_raises_before_anything_is_copied():
    original = SYSTEM.parameters.clone()
    with pytest.raises(ValueError, match="Could not find the parameter"):
        copy_paths(original, ["taxes.income_tax_rate", "taxes.no_such_tax"])


@settings(max_examples=60, deadline=None)
@given(
    changes=st.lists(
        st.tuples(
            st.sampled_from(LEAF_PATHS),
            st.floats(min_value=0, max_value=1e6, allow_nan=False),
        ),
        min_size=1,
        max_size=4,
        unique_by=lambda change: change[0],
    ),
    warm=st.booleans(),
)
def test_copying_paths_matches_cloning_the_whole_tree(changes, warm):
    """For any set of parameter changes, a tree that copies only their paths
    reads exactly what a full clone with the same changes reads, and the
    original reads what it did before, whether or not its at-instant caches
    were warm when it was copied."""
    original = SYSTEM.parameters.clone()
    before = {instant: _leaf_values(original, instant) for instant in INSTANTS}
    if warm:
        for instant in INSTANTS:
            original.get_at_instant(instant)

    paths = [path for path, _ in changes]
    copied = copy_paths(original, paths)
    reference = original.clone()
    for path, value in changes:
        _set(copied, path, value)
        _set(reference, path, value)

    for instant in INSTANTS:
        assert _leaf_values(copied, instant) == _leaf_values(reference, instant)
        assert _leaf_values(original, instant) == before[instant]
        # Read through the at-instant trees too, which is how formulas read
        # parameters and where a stale shared cache would show.
        assert _values_at_instant(copied, instant) == _values_at_instant(
            reference, instant
        )
        assert _values_at_instant(original, instant) == {
            path: before[instant][original.get_child(path).name]
            for path in PLAIN_LEAF_PATHS
        }


# Paths outside scales, which an at-instant tree reaches by attribute.
PLAIN_LEAF_PATHS = [path for path in LEAF_PATHS if "[" not in path]


def _values_at_instant(tree, instant):
    """Each plain parameter's value read through ``tree``'s at-instant tree,
    or ``None`` where that tree leaves it out (it has no value then)."""
    node_at_instant = tree.get_at_instant(instant)
    values = {}
    for path in PLAIN_LEAF_PATHS:
        value = node_at_instant
        try:
            for name in path.split("."):
                value = getattr(value, name)
        except ParameterNotFoundError:
            value = None
        values[path] = value
    return values
