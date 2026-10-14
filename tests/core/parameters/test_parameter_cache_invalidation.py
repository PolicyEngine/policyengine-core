"""A change to a parameter tree reaches every node at instant requested after it.

``ParameterNode`` caches its node at each instant, and a node at instant
resolves its children on first read. A change therefore has to clear the
cache of every node above the one it changes, or a node requested afterwards
would read old values for children an earlier read had resolved and new ones
for the rest. These tests pin each way of changing a tree, with and without
an earlier read.
"""

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import Parameter, ParameterNode
from policyengine_core.reforms import Reform
from tests.fixtures.parameter_nodes_at_instant import build_tree, materialise, snapshot

INSTANT = "2017-03-01"


@pytest.mark.parametrize("read_before", [False, True])
def test_a_bracket_update_reaches_a_node_requested_afterwards(read_before):
    """A freshly loaded scale's brackets know their scale, so updating a
    bracket's rate clears the caches above the scale too."""
    tree = build_tree()
    if read_before:
        assert tree.get_at_instant(INSTANT).scale.rates == [0.1, 0.2]

    rate = tree.children["scale"].brackets[1].children["rate"]
    rate.update(period="year:2017:1", value=0.5)

    assert tree.get_at_instant(INSTANT).scale.rates == [0.1, 0.5]


@pytest.mark.parametrize("read_before", [False, True])
def test_add_child_reaches_a_node_requested_afterwards(read_before):
    tree = build_tree()
    if read_before:
        assert tree.get_at_instant(INSTANT).group.x == 0.1

    tree.children["group"].add_child(
        "added", Parameter("group.added", data={"values": {"2010-01-01": 9}})
    )

    group = tree.get_at_instant(INSTANT).group
    assert group.added == 9
    assert "added" in list(group)


@pytest.mark.parametrize("read_before", [False, True])
def test_clear_instant_caches_after_a_direct_edit(read_before):
    """``values_list`` edits bypass ``Parameter.update``; clearing the caches
    afterwards makes them visible."""
    tree = build_tree()
    if read_before:
        assert tree.get_at_instant(INSTANT).group.x == 0.1

    for value_at_instant in tree.children["group"].children["x"].values_list:
        value_at_instant.value = 0.7
    tree.children["group"].clear_instant_caches()

    assert tree.get_at_instant(INSTANT).group.x == 0.7
    assert materialise(tree.get_at_instant(INSTANT)) == snapshot(tree, INSTANT)


def test_clear_instant_caches_clears_the_subtree_and_the_nodes_above():
    tree = build_tree()
    materialise(tree.get_at_instant(INSTANT))
    group = tree.children["group"]
    assert tree._at_instant_cache and group._at_instant_cache

    group.clear_instant_caches()

    assert not tree._at_instant_cache
    assert not group._at_instant_cache
    assert not group.children["inner"]._at_instant_cache
    # A sibling's cache holds nothing that depends on the group.
    assert tree.children["by_zone"]._at_instant_cache


def test_a_system_reads_values_changed_after_a_reform_read_them_during_construction():
    """Reforms applied while a system is built run before the parameter
    operations, which change values in place; the system clears its caches
    once they have run."""

    class read_then_change_in_place(Reform):
        def apply(self):
            assert self.parameters(INSTANT).taxes.income_tax_rate == 0.15
            for value_at_instant in self.parameters.taxes.income_tax_rate.values_list:
                value_at_instant.value = 0.4

    system = CountryTaxBenefitSystem(reform=read_then_change_in_place)

    assert system.parameters(INSTANT).taxes.income_tax_rate == 0.4
    assert system.get_parameters_at_instant(INSTANT).taxes.income_tax_rate == 0.4


def test_a_node_held_across_a_change_keeps_what_it_read():
    """The documented exception: a node requested before the change."""
    tree = build_tree()
    held = tree.get_at_instant(INSTANT).group
    assert held.x == 0.1

    tree.children["group"].children["x"].update(period="year:2017:1", value=0.9)

    assert held.x == 0.1
    assert tree.get_at_instant(INSTANT).group.x == 0.9


@pytest.mark.parametrize(
    "parent_child",
    [{"values": {"2010-01-01": 1}}, {"x": {"values": {"2010-01-01": 1}}}],
    ids=["parameter", "node"],
)
def test_a_child_named_parent_does_not_trap_cache_clearing(parent_child):
    """``add_child`` sets each child as an attribute, so a child named
    ``parent`` replaces the node's ``parent`` and points back at the node
    until the node is attached; clearing caches meanwhile must not loop."""
    tree = ParameterNode(
        "",
        data={
            "group": {"parent": parent_child, "later": {"values": {"2010-01-01": 2}}}
        },
    )
    group = tree.children["group"]
    assert group.parent is tree

    group.children["later"].update(period="year:2017:1", value=3)

    assert tree.get_at_instant(INSTANT).group.later == 3


def test_add_child_still_rejects_a_name_in_use():
    tree = build_tree()
    with pytest.raises(ValueError):
        tree.children["group"].add_child(
            "x", Parameter("group.x", data={"values": {"2010-01-01": 1}})
        )
    with pytest.raises(ParameterNotFoundError):
        tree.get_at_instant(INSTANT).group.missing
