"""Every node in a loaded parameter tree links to the node that holds it.

``Parameter.update`` walks upward through ``parent`` to clear cached
at-instant nodes and to mark ancestors as modified, so a missing or wrong
``parent`` link leaves stale values and unmarked ancestors. A cloned tree links
every node, so a loaded tree and its clone must behave the same.
"""

import numpy
import pytest

from policyengine_core.parameters import (
    Parameter,
    ParameterNode,
    ParameterScale,
    ParameterScaleBracket,
)


def build_tree_with_scale() -> ParameterNode:
    return ParameterNode(
        data={
            "tax": {
                "credit": {
                    "base": {
                        "brackets": [
                            {
                                "threshold": {"values": {"2020-01-01": 0}},
                                "amount": {"values": {"2020-01-01": 100}},
                            },
                            {
                                "threshold": {"values": {"2020-01-01": 10}},
                                "amount": {"values": {"2020-01-01": 50}},
                            },
                        ]
                    },
                },
                "rate": {"values": {"2020-01-01": 0.2}},
            }
        }
    )


def iter_tree_with_holders(root):
    """Yield (holding node, child) for every link in the tree, brackets included."""
    if isinstance(root, ParameterScale):
        for bracket in root.brackets:
            yield root, bracket
            yield from iter_tree_with_holders(bracket)
    elif isinstance(root, ParameterNode):
        for child in root.children.values():
            yield root, child
            yield from iter_tree_with_holders(child)


def modified_names(root) -> set:
    return {
        child.name
        for _, child in iter_tree_with_holders(root)
        if getattr(child, "modified", False)
    } | ({root.name} if getattr(root, "modified", False) else set())


def test_scale_brackets_link_to_their_scale():
    root = build_tree_with_scale()
    scale = root.tax.credit.base

    assert len(scale.brackets) == 2
    for bracket in scale.brackets:
        assert bracket.parent is scale


def test_every_loaded_node_links_to_its_holder():
    root = build_tree_with_scale()

    for holder, child in iter_tree_with_holders(root):
        assert child.parent is holder, child.name


def test_bracket_update_marks_same_nodes_on_loaded_tree_and_clone():
    loaded_root = build_tree_with_scale()
    cloned_root = build_tree_with_scale().clone()

    for root in (loaded_root, cloned_root):
        root.tax.credit.base.brackets[0].amount.update(period="year:2021:1", value=200)

    loaded_modified = modified_names(loaded_root)
    assert loaded_modified == modified_names(cloned_root)
    assert {
        "tax.credit.base[0].amount",
        "tax.credit.base[0]",
        "tax.credit.base",
        "tax.credit",
        "tax",
    } <= loaded_modified
    assert "tax.rate" not in loaded_modified


def test_bracket_update_clears_cached_ancestor_values():
    root = build_tree_with_scale()
    income = numpy.array([5])
    before = root("2021-01-01").tax.credit.base.calc(income)
    assert before[0] == 100

    root.tax.credit.base.brackets[0].amount.update(period="year:2021:1", value=200)

    assert root("2021-01-01").tax.credit.base.calc(income)[0] == 200


def test_child_named_parent_does_not_replace_the_parent_link():
    root = ParameterNode(
        data={
            "categories": {
                "parent": {"values": {"2020-01-01": 1}},
                "child": {"values": {"2020-01-01": 2}},
            }
        }
    )
    categories = root.children["categories"]
    parent_category = categories.children["parent"]

    assert categories.parent is root
    assert parent_category.parent is categories

    standalone = ParameterNode(data={"parent": {"values": {"2020-01-01": 1}}})
    assert standalone.parent is None
    assert standalone.children["parent"].parent is standalone

    parent_category.update(period="year:2021:1", value=3)
    assert root.modified
    assert root("2021-01-01").categories.parent == 3


@pytest.mark.parametrize("reserved_name", ["children", "name", "modified", "parent"])
def test_child_name_cannot_overwrite_node_attributes(reserved_name):
    root = ParameterNode(data={"node": {reserved_name: {"values": {"2020-01-01": 1}}}})
    node = root.children["node"]

    assert isinstance(node.children, dict)
    assert node.name == "node"
    assert node.parent is root
    assert node.modified is False
    assert node.children[reserved_name].parent is node
    assert root("2021-01-01").node[reserved_name] == 1


def test_child_name_cannot_shadow_node_methods():
    root = ParameterNode(data={"clone": {"values": {"2020-01-01": 1}}})

    assert callable(root.clone)
    cloned = root.clone()
    assert cloned.children["clone"].parent is cloned
    assert cloned("2021-01-01").clone == 1


def test_replace_child_links_the_replacement():
    root = ParameterNode(data={"node": {"value": {"values": {"2020-01-01": 1}}}})
    node = root.children["node"]
    replacement = ParameterNode("node.value", data={"a": {"values": {"2020-01-01": 5}}})

    node.replace_child("value", replacement)

    assert node.children["value"] is replacement
    assert node.value is replacement
    assert replacement.parent is node
    assert root("2021-01-01").node.value.a == 5


def test_replace_child_requires_an_existing_child():
    node = ParameterNode(data={"value": {"values": {"2020-01-01": 1}}})

    with pytest.raises(KeyError):
        node.replace_child("missing", Parameter("missing", {"2020-01-01": 1}))


def test_bracket_is_a_parameter_node():
    # The linking code treats brackets as ParameterNode instances.
    assert issubclass(ParameterScaleBracket, ParameterNode)


def test_bracket_appended_by_template_reform_links_to_its_scale():
    from policyengine_core.country_template import CountryTaxBenefitSystem
    from policyengine_core.country_template.reforms.modify_social_security_taxation import (
        modify_social_security_taxation,
    )

    reformed = modify_social_security_taxation(CountryTaxBenefitSystem())
    scale = reformed.parameters.taxes.social_security_contribution

    for bracket in scale.brackets:
        assert bracket.parent is scale, bracket.name


def test_replace_child_clears_cached_values_and_detaches_old_child():
    root = ParameterNode(data={"node": {"value": {"values": {"2020-01-01": 1}}}})
    node = root.children["node"]
    assert root("2021-01-01").node.value == 1
    old_child = node.children["value"]

    node.replace_child("value", Parameter("node.value", {"2020-01-01": 9}))

    assert root("2021-01-01").node.value == 9
    assert old_child.parent is None
    assert not node.modified
    assert not root.modified


def test_clone_keeps_modified_flag_when_a_child_is_named_modified():
    root = ParameterNode(data={"node": {"modified": {"values": {"2020-01-01": 1}}}})

    cloned_node = root.clone().children["node"]

    assert cloned_node.modified is False
    assert cloned_node.children["modified"].parent is cloned_node
