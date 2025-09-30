from typing import TYPE_CHECKING, Any

import numpy
from numpy.typing import ArrayLike

from policyengine_core import parameters
from policyengine_core.enums import Enum, EnumArray
from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import helpers

if TYPE_CHECKING:
    from policyengine_core.parameters.parameter_node import ParameterNode


class VectorialParameterNodeAtInstant:
    """
    Parameter node of the legislation at a given instant which has been vectorized.
    Vectorized parameters allow requests such as parameters.housing_benefit[zipcode], where zipcode is a vector
    """

    @staticmethod
    def build_from_node(
        node: "ParameterNode",
    ) -> "VectorialParameterNodeAtInstant":
        VectorialParameterNodeAtInstant.check_node_vectorisable(node)
        subnodes_name = sorted(node._children.keys())
        # Recursively vectorize the children of the node
        vectorial_subnodes = tuple(
            [
                (
                    VectorialParameterNodeAtInstant.build_from_node(
                        node[subnode_name]
                    ).vector
                    if isinstance(
                        node[subnode_name], parameters.ParameterNodeAtInstant
                    )
                    else node[subnode_name]
                )
                for subnode_name in subnodes_name
            ]
        )
        # A vectorial node is a wrapper around a numpy recarray
        # We first build the recarray
        recarray = numpy.array(
            [vectorial_subnodes],
            dtype=[
                (
                    subnode_name,
                    (
                        subnode.dtype
                        if isinstance(subnode, numpy.recarray)
                        else "float"
                    ),
                )
                for (subnode_name, subnode) in zip(
                    subnodes_name, vectorial_subnodes
                )
            ],
        )

        return VectorialParameterNodeAtInstant(
            node._name, recarray.view(numpy.recarray), node._instant_str
        )

    @staticmethod
    def check_node_vectorisable(node: "ParameterNode") -> None:
        """
        Check that a node can be casted to a vectorial node, in order to be able to use fancy indexing.
        """
        MESSAGE_PART_1 = "Cannot use fancy indexing on parameter node '{}', as"
        MESSAGE_PART_3 = "To use fancy indexing on parameter node, its children must be homogenous."
        MESSAGE_PART_4 = "See more at <https://openfisca.org/doc/coding-the-legislation/legislation_parameters#computing-a-parameter-that-depends-on-a-variable-fancy-indexing>."

        def raise_key_inhomogeneity_error(
            node_with_key, node_without_key, missing_key
        ):
            message = " ".join(
                [
                    MESSAGE_PART_1,
                    "'{}' exists, but '{}' doesn't.",
                    MESSAGE_PART_3,
                    MESSAGE_PART_4,
                ]
            ).format(
                node._name,
                ".".join([node_with_key, missing_key]),
                ".".join([node_without_key, missing_key]),
            )

            raise ValueError(message)

        def raise_type_inhomogeneity_error(node_name, non_node_name):
            message = " ".join(
                [
                    MESSAGE_PART_1,
                    "'{}' is a node, but '{}' is not.",
                    MESSAGE_PART_3,
                    MESSAGE_PART_4,
                ]
            ).format(
                node._name,
                node_name,
                non_node_name,
            )

            raise ValueError(message)

        def raise_not_implemented(node_name, node_type):
            message = " ".join(
                [
                    MESSAGE_PART_1,
                    "'{}' is a '{}', and fancy indexing has not been implemented yet on this kind of parameters.",
                    MESSAGE_PART_4,
                ]
            ).format(
                node._name,
                node_name,
                node_type,
            )
            raise NotImplementedError(message)

        def extract_named_children(node):
            return {
                ".".join([node._name, key]): value
                for key, value in node._children.items()
            }

        def check_nodes_homogeneous(named_nodes):
            # PolicyEngine models skip this check. The upsides:
            # * It cuts runtimes down by about 40%.
            # * Developers no longer get a helpful error message when attempting to use fancy
            #   indexing on a non-homogeneous parameter.
            return True
            """
            Check than several nodes (or parameters, or baremes) have the same structure.
            """
            names = list(named_nodes.keys())
            nodes = list(named_nodes.values())
            first_node = nodes[0]
            first_name = names[0]
            if isinstance(first_node, parameters.ParameterNodeAtInstant):
                children = extract_named_children(first_node)
                for node, name in list(zip(nodes, names))[1:]:
                    if not isinstance(node, parameters.ParameterNodeAtInstant):
                        raise_type_inhomogeneity_error(first_name, name)
                    first_node_keys = first_node._children.keys()
                    node_keys = node._children.keys()
                    if not first_node_keys == node_keys:
                        missing_keys = set(first_node_keys).difference(
                            node_keys
                        )
                        if (
                            missing_keys
                        ):  # If the first_node has a key that node hasn't
                            raise_key_inhomogeneity_error(
                                first_name, name, missing_keys.pop()
                            )
                        else:  # If If the node has a key that first_node doesn't have
                            missing_key = (
                                set(node_keys)
                                .difference(first_node_keys)
                                .pop()
                            )
                            raise_key_inhomogeneity_error(
                                name, first_name, missing_key
                            )
                    children.update(extract_named_children(node))
                check_nodes_homogeneous(children)
            elif isinstance(first_node, float) or isinstance(first_node, int):
                for node, name in list(zip(nodes, names))[1:]:
                    if isinstance(node, int) or isinstance(node, float):
                        pass
                    elif isinstance(node, parameters.ParameterNodeAtInstant):
                        raise_type_inhomogeneity_error(name, first_name)
                    else:
                        raise_not_implemented(name, type(node).__name__)

            else:
                raise_not_implemented(first_name, type(first_node).__name__)

        check_nodes_homogeneous(extract_named_children(node))

    def __init__(self, name: str, vector: ArrayLike, instant_str: str):
        self.vector = vector
        self._name = name
        self._instant_str = instant_str

    def __getattr__(self, attribute: str) -> Any:
        result = getattr(self.vector, attribute)
        if isinstance(result, numpy.recarray):
            return VectorialParameterNodeAtInstant(result)
        return result

    def __getitem__(self, key: str) -> Any:
        # If the key is a string, just get the subnode
        if isinstance(key, str):
            return self.__getattr__(key)
        # If the key is a vector, e.g. ['zone_1', 'zone_2', 'zone_1']
        elif isinstance(key, numpy.ndarray):
            if not numpy.issubdtype(key.dtype, numpy.str_):
                # In case the key is not a string vector, stringify it
                if key.dtype == object and issubclass(type(key[0]), Enum):
                    enum = type(key[0])
                    key = numpy.select(
                        [key == item for item in enum],
                        [str(item.name) for item in enum],
                        default="unknown",
                    )
                elif isinstance(key, EnumArray):
                    enum = key.possible_values
                    key = numpy.select(
                        [key == item.index for item in enum],
                        [item.name for item in enum],
                        default="unknown",
                    )
                else:
                    key = key.astype("str")
            names = list(
                self.dtype.names
            )  # Get all the names of the subnodes, e.g. ['zone_1', 'zone_2']
            conditions = [key == name for name in names]
            values = [self.vector[name] for name in names]

            # NumPy 2.x requires all arrays in numpy.select to have identical dtypes
            # For structured arrays with different field sets, we need to normalize them
            if (
                len(values) > 0
                and hasattr(values[0].dtype, "names")
                and values[0].dtype.names
            ):
                # Check if all values have the same dtype
                dtypes_match = all(
                    val.dtype == values[0].dtype for val in values
                )

                if not dtypes_match:
                    # Find the union of all field names across all values, preserving first seen order
                    all_fields = []
                    seen = set()
                    for val in values:
                        for field in val.dtype.names:
                            if field not in seen:
                                all_fields.append(field)
                                seen.add(field)

                    # Create unified dtype with all fields
                    unified_dtype = numpy.dtype(
                        [(f, "<f8") for f in all_fields]
                    )

                    # Cast all values to unified dtype
                    values_cast = []
                    for val in values:
                        casted = numpy.zeros(len(val), dtype=unified_dtype)
                        for field in val.dtype.names:
                            casted[field] = val[field]
                        values_cast.append(casted)

                    default = numpy.zeros(
                        len(values_cast[0]), dtype=unified_dtype
                    )
                    # Fill with NaN
                    for field in unified_dtype.names:
                        default[field] = numpy.nan

                    result = numpy.select(conditions, values_cast, default)
                else:
                    # All dtypes match, use original logic
                    default = numpy.full_like(values[0], numpy.nan)
                    result = numpy.select(conditions, values, default)
            else:
                # Non-structured array case
                default = numpy.full_like(
                    values[0] if values else self.vector[key[0]], numpy.nan
                )
                result = numpy.select(conditions, values, default)

            # Check for unexpected keys (NaN results from missing keys)
            if helpers.contains_nan(result):
                unexpected_keys = set(key).difference(self.vector.dtype.names)
                if unexpected_keys:
                    unexpected_key = unexpected_keys.pop()
                    raise ParameterNotFoundError(
                        ".".join([self._name, unexpected_key]),
                        self._instant_str,
                    )

            # If the result is not a leaf, wrap the result in a vectorial node.
            if numpy.issubdtype(
                result.dtype, numpy.record
            ) or numpy.issubdtype(result.dtype, numpy.void):
                return VectorialParameterNodeAtInstant(
                    self._name, result.view(numpy.recarray), self._instant_str
                )

            return result
