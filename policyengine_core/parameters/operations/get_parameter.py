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
        try:
            if "[" not in name:
                node = node.children[name]
            else:
                try:
                    name, index = name.split("[")
                    index = int(index[:-1])
                    node = node.children[name].brackets[index]
                except Exception:
                    # Enhanced error message for bracket syntax errors
                    raise ValueError(
                        f"Invalid bracket syntax at '{name}' in parameter path '{parameter}'. "
                        "Use format: parameter_name[index], e.g., brackets[0].rate"
                    )
        except KeyError:
            # Enhanced error message for parameter not found
            similar_params = []
            if hasattr(node, "children"):
                similar_params = [
                    key
                    for key in node.children.keys()
                    if (
                        name.lower() in key.lower()
                        or key.lower().startswith(name.lower())
                    )
                    and "[" not in name
                ]

            suggestions = (
                f" Did you mean: {', '.join(similar_params)}?"
                if similar_params
                else ""
            )
            raise ValueError(
                f"Parameter '{name}' not found in path '{parameter}'.{suggestions}"
            )
        except (IndexError, AttributeError) as e:
            # Enhanced error message for bracket index errors
            if isinstance(e, IndexError) and "[" in name:
                try:
                    name_part = name.split("[")[0]
                    max_index = len(node.children[name_part].brackets) - 1
                    raise ValueError(
                        f"Bracket index out of range in '{name}' of parameter path '{parameter}'. "
                        f"Valid indices are 0 to {max_index}."
                    )
                except Exception:
                    pass
            elif isinstance(e, AttributeError) and "[" in name:
                name_part = name.split("[")[0]
                raise ValueError(
                    f"'{name_part}' in parameter path '{parameter}' does not support bracket indexing. "
                    "Only scale parameters (like brackets) support indexing."
                )

            # Generic error message
            raise ValueError(
                f"Could not find the parameter '{parameter}' (failed at '{name}'). {str(e)}"
            )
        except ValueError:
            # Re-raise ValueError directly
            raise
        except Exception as e:
            # Catch-all for other errors
            raise ValueError(
                f"Error accessing parameter '{parameter}' at '{name}': {str(e)}"
            )

    return node
