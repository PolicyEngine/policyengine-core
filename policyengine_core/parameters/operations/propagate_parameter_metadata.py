from policyengine_core.parameters.parameter import Parameter
from policyengine_core.parameters.parameter_node import ParameterNode


def propagate_parameter_metadata(root: ParameterNode) -> ParameterNode:
    """Passes parameter metadata to descendents where this is specified.

    Breakdown metadata is ignored.

    Args:
        root (ParameterNode): The root node.

    Returns:
        ParameterNode: The edited parameter root.
    """

    UNPROPAGAGED_METADATA = ["breakdown", "label", "name", "description"]

    # Pre-compute all descendants once to avoid O(n²) complexity
    all_descendants = list(root.get_descendants())

    # Find parameters that need to propagate metadata
    propagators = [
        p for p in all_descendants
        if p.metadata.get("propagate_metadata_to_children")
    ]

    # For each parameter that propagates metadata, update its descendants
    for parameter in propagators:
        # Get metadata to propagate
        metadata_to_propagate = {
            key: value
            for key, value in parameter.metadata.items()
            if key not in UNPROPAGAGED_METADATA
        }

        # Only call get_descendants() if there's metadata to propagate
        if metadata_to_propagate:
            for descendant in parameter.get_descendants():
                descendant.metadata.update(metadata_to_propagate)

    return root
