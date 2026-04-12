import ast
import logging
from typing import Any, Dict, List, Type

from policyengine_core.enums import Enum
from policyengine_core.parameters.parameter import Parameter
from policyengine_core.parameters.parameter_node import ParameterNode
from policyengine_core.variables import Variable


def homogenize_parameter_structures(
    root: ParameterNode, variables: Dict[str, Variable], default_value: Any = 0
) -> ParameterNode:
    """
    Homogenize the structure of a parameter tree to match the structure of the variables.

    Args:
        root (ParameterNode): The root of the parameter tree.
        variables (Dict[str, Variable]): The variables to match the structure to.
        default_value (Any, optional): The default value to use for missing parameters. Defaults to 0.

    Returns:
        ParameterNode: The root of the homogenized parameter tree.
    """
    for node in root.get_descendants():
        if isinstance(node, ParameterNode):
            breakdown = get_breakdown_variables(node)
            node = homogenize_parameter_node(node, breakdown, variables, default_value)
    return root


def get_breakdown_variables(node: ParameterNode) -> List[str]:
    """
    Returns the list of variables that are used to break down the parameter.
    """
    breakdown = node.metadata.get("breakdown")
    if breakdown is not None:
        if isinstance(breakdown, str):
            # Single element, cast to list.
            breakdown = [breakdown]
        if not isinstance(breakdown, list):
            # Not a list, skip process and warn.
            logging.warning(
                f"Invalid breakdown metadata for parameter {node.name}: {type(breakdown)}"
            )
            return None
        return breakdown
    else:
        return None


def homogenize_parameter_node(
    node: ParameterNode,
    breakdown: List[str],
    variables: Dict[str, Variable],
    default_value: Any,
) -> ParameterNode:
    if breakdown is None:
        return node
    first_breakdown = breakdown[0]
    if isinstance(first_breakdown, list):
        possible_values = first_breakdown
    elif first_breakdown in variables:
        dtype = variables[first_breakdown].value_type
        if dtype == Enum:
            possible_values = list(
                map(
                    lambda enum: enum.name,
                    variables[first_breakdown].possible_values,
                )
            )
        elif dtype == bool:
            possible_values = [True, False]
    else:
        possible_values = evaluate_dynamic_breakdown(first_breakdown)
    if not hasattr(node, "children"):
        node = ParameterNode(
            node.name,
            data={
                child: {
                    "0000-01-01": default_value,
                    "2040-01-01": default_value,
                }
                for child in possible_values
            },
        )
    missing_values = set(possible_values) - set(node.children)
    further_breakdown = len(breakdown) > 1
    for value in missing_values:
        if str(value) not in node.children:
            # Integers behave strangely, this fixes it.
            node.add_child(
                str(value),
                Parameter(
                    node.name + "." + str(value),
                    {"0000-01-01": default_value, "2040-01-01": default_value},
                ),
            )
    possible_values_str = {str(v) for v in possible_values}
    extra_children = []
    for child in node.children:
        child_key = child.split(".")[-1]
        if (
            child_key not in possible_values_str
            and str(child_key) not in possible_values_str
        ):
            extra_children.append(child_key)
    if extra_children:
        raise ValueError(
            f"Parameter {node.name} has children {extra_children} "
            f"that are not in the possible values of the breakdown "
            f"variable '{first_breakdown}'. Check that the breakdown "
            f"metadata references the correct variable and that all "
            f"parameter keys are valid enum values."
        )
    for child in node.children:
        if further_breakdown:
            node.children[child] = homogenize_parameter_node(
                node.children[child], breakdown[1:], variables, default_value
            )
    return node


def evaluate_dynamic_breakdown(expression: str) -> List[Any]:
    """Safely evaluate a dynamic breakdown expression.

    The parameter metadata only needs literal collections and the documented
    ``range(...)`` / ``list(range(...))`` forms. Anything else is rejected.
    """

    parsed = ast.parse(expression, mode="eval")
    evaluated = evaluate_dynamic_breakdown_node(parsed.body)
    if isinstance(evaluated, range):
        return list(evaluated)
    if isinstance(evaluated, (list, tuple)):
        return list(evaluated)
    if isinstance(evaluated, set):
        return list(evaluated)
    raise ValueError(
        f"Invalid dynamic breakdown expression '{expression}'. "
        "Only literal collections and range() calls are allowed."
    )


def evaluate_dynamic_breakdown_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [evaluate_dynamic_breakdown_node(element) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(evaluate_dynamic_breakdown_node(element) for element in node.elts)
    if isinstance(node, ast.Set):
        return {evaluate_dynamic_breakdown_node(element) for element in node.elts}
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub)
    ):
        operand = evaluate_dynamic_breakdown_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range":
            args = [evaluate_dynamic_breakdown_node(arg) for arg in node.args]
            if node.keywords:
                raise ValueError("range() keyword arguments are not allowed")
            return range(*args)
        if node.func.id == "list":
            if len(node.args) != 1 or node.keywords:
                raise ValueError("list() must contain a single positional argument")
            return list(evaluate_dynamic_breakdown_node(node.args[0]))
    raise ValueError(
        f"Unsupported dynamic breakdown expression: {ast.unparse(node) if hasattr(ast, 'unparse') else type(node).__name__}"
    )
