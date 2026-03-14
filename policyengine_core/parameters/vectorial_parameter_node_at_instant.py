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
                    if isinstance(node[subnode_name], parameters.ParameterNodeAtInstant)
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
                    (subnode.dtype if isinstance(subnode, numpy.recarray) else "float"),
                )
                for (subnode_name, subnode) in zip(subnodes_name, vectorial_subnodes)
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
        MESSAGE_PART_3 = (
            "To use fancy indexing on parameter node, its children must be homogenous."
        )
        MESSAGE_PART_4 = "See more at <https://openfisca.org/doc/coding-the-legislation/legislation_parameters#computing-a-parameter-that-depends-on-a-variable-fancy-indexing>."

        def raise_key_inhomogeneity_error(node_with_key, node_without_key, missing_key):
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
                        missing_keys = set(first_node_keys).difference(node_keys)
                        if missing_keys:  # If the first_node has a key that node hasn't
                            raise_key_inhomogeneity_error(
                                first_name, name, missing_keys.pop()
                            )
                        else:  # If If the node has a key that first_node doesn't have
                            missing_key = (
                                set(node_keys).difference(first_node_keys).pop()
                            )
                            raise_key_inhomogeneity_error(name, first_name, missing_key)
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
        # Convert pandas arrays (e.g., StringArray from pandas 3) to numpy
        # before checking, since StringArray has __array__ but is not hashable
        if hasattr(key, "__array__") and not isinstance(key, numpy.ndarray):
            key = numpy.asarray(key)
        if isinstance(key, numpy.ndarray):
            names = self.dtype.names
            # Build name->child-index mapping (cached on instance)
            if not hasattr(self, "_name_to_child_idx"):
                self._name_to_child_idx = {name: i for i, name in enumerate(names)}

            name_to_child_idx = self._name_to_child_idx
            n = len(key)
            SENTINEL = len(names)

            # Convert key to integer indices directly, avoiding
            # expensive intermediate string arrays where possible.
            if isinstance(key, EnumArray):
                # EnumArray: map enum int codes -> child indices via
                # a pre-built lookup table (O(N), no string comparison).
                enum = key.possible_values
                cache_key = id(enum)
                if not hasattr(self, "_enum_lut_cache"):
                    self._enum_lut_cache = {}
                lut = self._enum_lut_cache.get(cache_key)
                if lut is None:
                    enum_items = list(enum)
                    max_code = max(item.index for item in enum_items) + 1
                    lut = numpy.full(max_code, SENTINEL, dtype=numpy.intp)
                    for item in enum_items:
                        child_idx = name_to_child_idx.get(item.name)
                        if child_idx is not None:
                            lut[item.index] = child_idx
                    self._enum_lut_cache[cache_key] = lut
                idx = lut[numpy.asarray(key)]
            elif (
                key.dtype == object and len(key) > 0 and issubclass(type(key[0]), Enum)
            ):
                # Object array of Enum instances
                enum = type(key[0])
                cache_key = id(enum)
                if not hasattr(self, "_enum_lut_cache"):
                    self._enum_lut_cache = {}
                lut = self._enum_lut_cache.get(cache_key)
                if lut is None:
                    enum_items = list(enum)
                    max_code = max(item.index for item in enum_items) + 1
                    lut = numpy.full(max_code, SENTINEL, dtype=numpy.intp)
                    for item in enum_items:
                        child_idx = name_to_child_idx.get(str(item.name))
                        if child_idx is not None:
                            lut[item.index] = child_idx
                    self._enum_lut_cache[cache_key] = lut
                codes = numpy.array([v.index for v in key], dtype=numpy.intp)
                idx = lut[codes]
            else:
                # String keys: map via dict lookup
                if not numpy.issubdtype(key.dtype, numpy.str_):
                    key = key.astype("str")
                # Vectorised dict lookup using numpy unique + scatter
                uniq, inverse = numpy.unique(key, return_inverse=True)
                uniq_idx = numpy.array(
                    [name_to_child_idx.get(u, SENTINEL) for u in uniq],
                    dtype=numpy.intp,
                )
                idx = uniq_idx[inverse]

            # Gather values by child index using take on a stacked array.
            values = [self.vector[name] for name in names]

            is_structured = (
                len(values) > 0
                and hasattr(values[0].dtype, "names")
                and values[0].dtype.names
            )

            if is_structured:
                dtypes_match = all(val.dtype == values[0].dtype for val in values)
                v0_len = len(values[0])

                if v0_len <= 1:
                    # 1-element structured arrays: simple concat + index
                    if not dtypes_match:
                        all_fields = []
                        seen = set()
                        for val in values:
                            for field in val.dtype.names:
                                if field not in seen:
                                    all_fields.append(field)
                                    seen.add(field)

                        unified_dtype = numpy.dtype([(f, "<f8") for f in all_fields])

                        values_cast = []
                        for val in values:
                            casted = numpy.zeros(len(val), dtype=unified_dtype)
                            for field in val.dtype.names:
                                casted[field] = val[field]
                            values_cast.append(casted)

                        default = numpy.zeros(1, dtype=unified_dtype)
                        for field in unified_dtype.names:
                            default[field] = numpy.nan
                        stacked = numpy.concatenate(values_cast + [default])
                        result = stacked[idx]
                    else:
                        default = numpy.full(1, numpy.nan, dtype=values[0].dtype)
                        stacked = numpy.concatenate(values + [default])
                        result = stacked[idx]
                else:
                    # N-element structured arrays: check if fields are
                    # simple scalars (fast path) or nested records
                    # (fall back to numpy.select).
                    first_field = values[0].dtype.names[0]
                    field_dtype = values[0][first_field].dtype
                    is_nested = (
                        hasattr(field_dtype, "names") and field_dtype.names is not None
                    )

                    if is_nested:
                        # Nested structured: fall back to numpy.select
                        conditions = [idx == i for i in range(len(values))]
                        if not dtypes_match:
                            all_fields = []
                            seen = set()
                            for val in values:
                                for field in val.dtype.names:
                                    if field not in seen:
                                        all_fields.append(field)
                                        seen.add(field)
                            unified_dtype = numpy.dtype(
                                [(f, "<f8") for f in all_fields]
                            )
                            values_cast = []
                            for val in values:
                                casted = numpy.zeros(len(val), dtype=unified_dtype)
                                for field in val.dtype.names:
                                    casted[field] = val[field]
                                values_cast.append(casted)
                            default = numpy.zeros(v0_len, dtype=unified_dtype)
                            for field in unified_dtype.names:
                                default[field] = numpy.nan
                            result = numpy.select(conditions, values_cast, default)
                        else:
                            default = numpy.full_like(values[0], numpy.nan)
                            result = numpy.select(conditions, values, default)
                    else:
                        # Flat structured: fast per-field indexing
                        if not dtypes_match:
                            all_fields = []
                            seen = set()
                            for val in values:
                                for field in val.dtype.names:
                                    if field not in seen:
                                        all_fields.append(field)
                                        seen.add(field)
                            unified_dtype = numpy.dtype(
                                [(f, "<f8") for f in all_fields]
                            )
                            values_unified = []
                            for val in values:
                                casted = numpy.zeros(len(val), dtype=unified_dtype)
                                for field in val.dtype.names:
                                    casted[field] = val[field]
                                values_unified.append(casted)
                            field_names = all_fields
                            result_dtype = unified_dtype
                        else:
                            values_unified = values
                            field_names = values[0].dtype.names
                            result_dtype = values[0].dtype

                        result = numpy.empty(n, dtype=result_dtype)
                        arange_n = numpy.arange(v0_len)
                        for field in field_names:
                            field_stack = numpy.empty(
                                (len(values_unified) + 1, v0_len),
                                dtype=numpy.float64,
                            )
                            for i, v in enumerate(values_unified):
                                field_stack[i] = v[field]
                            field_stack[-1] = numpy.nan
                            result[field] = field_stack[idx, arange_n]
            else:
                # Non-structured: values are either scalars (1-elem arrays)
                # or N-element vectors (after prior vectorial indexing).
                if values:
                    v0 = numpy.asarray(values[0])
                    if v0.ndim == 0 or v0.shape[0] <= 1:
                        # Scalar per child: 1D lookup
                        scalar_vals = numpy.empty(len(values) + 1, dtype=numpy.float64)
                        for i, v in enumerate(values):
                            scalar_vals[i] = float(v)
                        scalar_vals[-1] = numpy.nan
                        result = scalar_vals[idx]
                    else:
                        # N-element vectors: stack into (K+1, N) matrix
                        m = v0.shape[0]
                        stacked = numpy.empty((len(values) + 1, m), dtype=numpy.float64)
                        for i, v in enumerate(values):
                            stacked[i] = v
                        stacked[-1] = numpy.nan
                        result = stacked[idx, numpy.arange(m)]
                else:
                    result = numpy.full(n, numpy.nan)

            # Check for unexpected keys
            if helpers.contains_nan(result):
                unexpected_keys = set(
                    numpy.asarray(key, dtype=str)
                    if not numpy.issubdtype(numpy.asarray(key).dtype, numpy.str_)
                    else key
                ).difference(self.vector.dtype.names)
                if unexpected_keys:
                    unexpected_key = unexpected_keys.pop()
                    raise ParameterNotFoundError(
                        ".".join([self._name, unexpected_key]),
                        self._instant_str,
                    )

            # If the result is not a leaf, wrap the result in a vectorial node.
            if numpy.issubdtype(result.dtype, numpy.record) or numpy.issubdtype(
                result.dtype, numpy.void
            ):
                return VectorialParameterNodeAtInstant(
                    self._name, result.view(numpy.recarray), self._instant_str
                )

            return result
