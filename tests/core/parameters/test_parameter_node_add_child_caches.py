"""``ParameterNode.add_child`` clears the nodes at instant cached by the node it
changes and by every node above it.

A node at instant lists the children its parameter node had when it was
built. Without clearing, a node at instant cached before ``add_child`` was
served again afterwards, without the new child.
"""

import pytest

from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import Parameter, ParameterNode

INSTANT = "2020-01-01"


def leaf(name: str, value) -> Parameter:
    return Parameter(name, data={"values": {"2010-01-01": value}})


def three_level_tree() -> ParameterNode:
    return ParameterNode(
        "", data={"gov": {"tax": {"rate": {"values": {"2010-01-01": 0.1}}}}}
    )


def read_the_root(tree: ParameterNode):
    tree(INSTANT)


def read_down_to_the_leaf(tree: ParameterNode):
    tree(INSTANT).gov.tax.rate


def read_every_level(tree: ParameterNode):
    tree(INSTANT).gov.tax.rate
    tree.gov(INSTANT).tax.rate
    tree.gov.tax(INSTANT).rate


@pytest.mark.parametrize(
    "read_first", [read_the_root, read_down_to_the_leaf, read_every_level]
)
def test_nodes_requested_after_add_child_have_the_new_child(read_first):
    tree = three_level_tree()
    read_first(tree)

    tree.gov.tax.add_child("allowance", leaf("gov.tax.allowance", 500))

    assert tree(INSTANT).gov.tax.allowance == 500
    assert tree.gov(INSTANT).tax.allowance == 500
    assert tree.gov.tax(INSTANT).allowance == 500
    assert list(tree(INSTANT).gov.tax) == ["rate", "allowance"]


def test_merged_children_are_in_nodes_requested_afterwards():
    tree = three_level_tree()
    read_down_to_the_leaf(tree)

    tree.gov.tax.merge(
        ParameterNode("other", data={"allowance": {"values": {"2010-01-01": 500}}})
    )

    assert tree(INSTANT).gov.tax.allowance == 500


def test_add_child_leaves_other_branches_cached():
    tree = ParameterNode(
        "",
        data={
            "gov": {"tax": {"rate": {"values": {"2010-01-01": 0.1}}}},
            "other": {"amount": {"values": {"2010-01-01": 3}}},
        },
    )
    other_at_instant = tree(INSTANT).other

    tree.gov.tax.add_child("allowance", leaf("gov.tax.allowance", 500))

    assert tree(INSTANT).other is other_at_instant


def test_a_node_held_from_before_add_child_keeps_the_children_it_had():
    tree = three_level_tree()
    held = tree(INSTANT)
    tax_at_instant = held.gov.tax

    tree.gov.tax.add_child("allowance", leaf("gov.tax.allowance", 500))

    with pytest.raises(ParameterNotFoundError):
        tax_at_instant.allowance
    assert list(held.gov.tax) == ["rate"]


@pytest.mark.parametrize(
    "parent_child",
    [
        {"values": {"2010-01-01": True}},
        {"nested": {"values": {"2010-01-01": True}}},
    ],
    ids=["parameter", "node"],
)
def test_a_node_with_a_child_named_parent_loads_and_reads(parent_child):
    # policyengine-us has ``gov.hhs.medicaid.eligibility.categories.parent``.
    # While a node loads, ``add_child`` sets each child as an attribute, so
    # until the node is attached, ``categories.parent`` is that child, whose
    # own parent is ``categories``. Children added after it must not walk
    # that loop for ever.
    tree = ParameterNode(
        "",
        data={
            "categories": {
                "adult": {"values": {"2010-01-01": True}},
                "parent": parent_child,
                "senior": {"values": {"2010-01-01": False}},
            }
        },
    )

    categories = tree(INSTANT).categories
    assert categories.adult is True
    assert categories.senior is False
    assert categories["parent"] is not None
    assert tree.categories.parent is tree


def test_a_child_named_like_the_cache_can_be_added():
    # A reform can add a child of any name. policyengine-us adds one named
    # after the cache and reads the tree through a clone.
    tree = three_level_tree()
    read_down_to_the_leaf(tree)

    tree.gov.add_child("_at_instant_cache", leaf("gov._at_instant_cache", 1))

    assert tree.clone()(INSTANT).gov.tax.rate == 0.1
    assert tree.clone()(INSTANT).gov["_at_instant_cache"] == 1
