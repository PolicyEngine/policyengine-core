"""Nodes that homogenization builds for further breakdown levels are linked.

``homogenize_parameter_node`` replaces a placeholder ``Parameter`` with a
``ParameterNode`` when a breakdown has more than one level. The replacement
must link to the node holding it, and that node's attribute must point at the
replacement rather than the placeholder. Placeholders that homogenization adds
for missing breakdown values must stay linked too.
"""

from policyengine_core.parameters import Parameter, ParameterNode
from policyengine_core.parameters.operations.homogenize_parameters import (
    homogenize_parameter_structures,
)


def build_two_level_tree(first_level_values) -> ParameterNode:
    return ParameterNode(
        data={
            "program": {
                "amount": {
                    str(first_level_values[0]): {
                        "1": {"2020-01-01": 10},
                    },
                    "metadata": {
                        "breakdown": [first_level_values, [1, 2]],
                    },
                }
            }
        }
    )


def test_homogenized_replacements_link_to_their_holder():
    root = build_two_level_tree(["A", "B"])
    homogenize_parameter_structures(root, variables={})
    amount = root.program.amount

    for key in ("A", "B"):
        replacement = amount.children[key]
        assert isinstance(replacement, ParameterNode)
        assert replacement.parent is amount
        assert getattr(amount, key) is replacement
        for leaf in replacement.children.values():
            assert leaf.parent is replacement


def test_update_under_homogenized_node_marks_every_ancestor_and_refreshes_cached_value():
    root = build_two_level_tree(["A", "B"])
    homogenize_parameter_structures(root, variables={})

    before = root("2021-01-01").program.amount.B["2"]
    assert before == 0

    leaf = root.program.amount.children["B"].children["2"]
    leaf.update(period="year:2021:1", value=7)

    assert root.program.amount.children["B"].modified
    assert root.program.amount.modified
    assert root.program.modified
    assert root.modified
    assert root("2021-01-01").program.amount.B["2"] == 7


def test_missing_breakdown_value_named_parent_keeps_the_parent_link():
    # "parent" is a missing breakdown value, so homogenization adds it as a
    # placeholder Parameter and then replaces it with a ParameterNode.
    root = build_two_level_tree(["child", "parent"])
    homogenize_parameter_structures(root, variables={})
    program = root.children["program"]
    amount = program.children["amount"]

    assert amount.parent is program
    replacement = amount.children["parent"]
    assert isinstance(replacement, ParameterNode)
    assert replacement.parent is amount

    replacement.children["2"].update(period="year:2021:1", value=4)
    assert amount.modified
    assert program.modified
    assert root.modified
    assert root("2021-01-01").program.amount["parent"]["2"] == 4


def test_single_level_homogenization_links_added_placeholders():
    root = ParameterNode(
        data={
            "program": {
                "A": {"2020-01-01": 1},
                "metadata": {"breakdown": [["A", "B"]]},
            }
        }
    )
    homogenize_parameter_structures(root, variables={})
    program = root.program

    placeholder = program.children["B"]
    assert isinstance(placeholder, Parameter)
    assert placeholder.parent is program
    assert program.B is placeholder
