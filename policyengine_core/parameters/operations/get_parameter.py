from policyengine_core.parameters.parameter import Parameter
from policyengine_core.parameters.parameter_node import ParameterNode
from policyengine_core.errors.parameter_path_error import ParameterPathError


def get_parameter(root: ParameterNode, parameter: str) -> Parameter:
    """Gets a parameter from the tree by name.

    Args:
        root (ParameterNode): The root of the parameter tree.
        parameter (str): The name of the parameter to get.

    Returns:
        Parameter: The parameter.
    """
    node = root
    
    for path_component in parameter.split("."):
        node = _navigate_to_node(node, path_component, parameter)
    
    return node


def _navigate_to_node(node, path_component, full_path):
    """Navigate to the next node in the parameter tree."""
    if hasattr(node, "brackets") and "[" in path_component:
        # Handle case where we're already at a scale parameter and need to access a bracket
        return _handle_bracket_access(node, path_component, full_path)
        
    # Parse the path component into name part and optional bracket part
    name_part, index = _parse_path_component(path_component, full_path)
    
    # For regular node navigation (no brackets)
    if not hasattr(node, "children"):
        raise ParameterPathError(
            f"Cannot access '{path_component}' in '{full_path}'. "
            "The parent is not a parameter node with children.",
            parameter_path=full_path,
            failed_at=path_component
        )
    
    # Look up the name in the node's children
    try:
        child_node = node.children[name_part]
    except KeyError:
        suggestions = _find_similar_parameters(node, name_part)
        suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ParameterPathError(
            f"Parameter '{name_part}' not found in path '{full_path}'.{suggestion_text}",
            parameter_path=full_path, 
            failed_at=name_part
        )
        
    # If we have a bracket index, access the brackets property
    if index is not None:
        return _access_bracket(child_node, name_part, index, path_component, full_path)
    
    return child_node


def _handle_bracket_access(node, path_component, full_path):
    """Handle bracket access on an existing ParameterScale object."""
    # Extract the bracket index from the path component
    if not path_component.startswith("[") or not path_component.endswith("]"):
        raise ParameterPathError(
            f"Invalid bracket syntax at '{path_component}' in parameter path '{full_path}'. "
            "Use format: [index], e.g., [0]",
            parameter_path=full_path,
            failed_at=path_component
        )
    
    try:
        index = int(path_component[1:-1])
    except ValueError:
        raise ParameterPathError(
            f"Invalid bracket index in '{path_component}' of parameter path '{full_path}'. "
            "Index must be an integer.",
            parameter_path=full_path,
            failed_at=path_component
        )
    
    try:
        return node.brackets[index]
    except IndexError:
        max_index = len(node.brackets) - 1
        raise ParameterPathError(
            f"Bracket index out of range in '{path_component}' of parameter path '{full_path}'. "
            f"Valid indices are 0 to {max_index}.",
            parameter_path=full_path,
            failed_at=path_component
        )


def _parse_path_component(path_component, full_path):
    """Parse a path component into name and optional bracket index."""
    if "[" not in path_component:
        return path_component, None
        
    try:
        parts = path_component.split("[", 1)  # Split only on the first occurrence of "["
        name_part = parts[0]
        bracket_part = "[" + parts[1]  # Include the "[" for consistent reporting
        
        if not bracket_part.endswith("]"):
            raise ParameterPathError(
                f"Invalid bracket syntax at '{path_component}' in parameter path '{full_path}'. "
                "Use format: parameter_name[index], e.g., brackets[0].rate",
                parameter_path=full_path,
                failed_at=name_part + bracket_part  # Use the original bracket part for error reporting
            )
            
        try:
            index = int(bracket_part[1:-1])  # Remove both "[" and "]"
            return name_part, index
        except ValueError:
            raise ParameterPathError(
                f"Invalid bracket index in '{path_component}' of parameter path '{full_path}'. "
                "Index must be an integer.",
                parameter_path=full_path,
                failed_at=name_part + bracket_part
            )
            
    except ValueError:  # More than one "[" found
        raise ParameterPathError(
            f"Invalid bracket syntax at '{path_component}' in parameter path '{full_path}'. "
            "Use format: parameter_name[index], e.g., brackets[0].rate",
            parameter_path=full_path,
            failed_at=path_component
        )


def _access_bracket(node, name_part, index, path_component, full_path):
    """Access a bracket by index on a node."""
    if not hasattr(node, "brackets"):
        raise ParameterPathError(
            f"'{name_part}' in parameter path '{full_path}' does not support bracket indexing. "
            "Only scale parameters (like brackets) support indexing.",
            parameter_path=full_path,
            failed_at=path_component
        )
        
    try:
        return node.brackets[index]
    except IndexError:
        max_index = len(node.brackets) - 1
        raise ParameterPathError(
            f"Bracket index out of range in '{path_component}' of parameter path '{full_path}'. "
            f"Valid indices are 0 to {max_index}.",
            parameter_path=full_path,
            failed_at=path_component
        )


def _find_similar_parameters(node, name):
    """Find parameter names similar to the requested one."""
    if not hasattr(node, "children"):
        return []
    
    return [
        key for key in node.children.keys()
        if name.lower() in key.lower()
    ]
