from policyengine_core.parameters.parameter import Parameter
from policyengine_core.parameters.parameter_node import ParameterNode


def get_parameter(root: ParameterNode, parameter: str) -> Parameter:
    """Gets a parameter from the tree by name.

    Args:
        root (ParameterNode): The root of the parameter tree.
        parameter (str): The name of the parameter to get.

    Returns:
        Parameter: The parameter.
    """
    node = root
    for name in parameter.split("."):
        if "[" not in name:
            try:
                node = node.children[name]
            except KeyError:
                suggestions = _find_similar_parameters(node, name)
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(
                    f"Parameter '{name}' not found in path '{parameter}'.{suggestion_text}"
                )
        else:
            node = _process_bracket_notation(node, name, parameter)
    
    return node


def _find_similar_parameters(node, name):
    """Find parameter names similar to the requested one."""
    if not hasattr(node, "children"):
        return []
    
    return [
        key for key in node.children.keys()
        if name.lower() in key.lower()
    ]


def _process_bracket_notation(node, name, full_parameter_path):
    """Process a parameter name with bracket notation (e.g., 'brackets[0]')."""
    try:
        name_part, bracket_part = name.split("[")
        if not bracket_part.endswith("]"):
            raise ValueError(
                f"Invalid bracket syntax at '{name}' in parameter path '{full_parameter_path}'. "
                "Use format: parameter_name[index], e.g., brackets[0].rate"
            )
            
        index = int(bracket_part[:-1])
        
        try:
            child_node = node.children[name_part]
        except KeyError:
            suggestions = _find_similar_parameters(node, name_part)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(
                f"Parameter '{name_part}' not found in path '{full_parameter_path}'.{suggestion_text}"
            )
            
        if not hasattr(child_node, "brackets"):
            raise AttributeError(
                f"'{name_part}' in parameter path '{full_parameter_path}' does not support bracket indexing. "
                "Only scale parameters (like brackets) support indexing."
            )
            
        try:
            return child_node.brackets[index]
        except IndexError:
            max_index = len(child_node.brackets) - 1
            raise ValueError(
                f"Bracket index out of range in '{name}' of parameter path '{full_parameter_path}'. "
                f"Valid indices are 0 to {max_index}."
            )
            
    except ValueError as e:
        # If it's our own ValueError with detailed message, propagate it
        if "parameter path" in str(e):
            raise
        # Otherwise, it's likely a syntax error in the bracket notation
        raise ValueError(
            f"Invalid bracket syntax at '{name}' in parameter path '{full_parameter_path}'. "
            "Use format: parameter_name[index], e.g., brackets[0].rate"
        )
