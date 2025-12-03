from __future__ import annotations
import enum
import logging
from functools import lru_cache
from typing import Tuple, Union
import numpy as np
from .config import ENUM_ARRAY_DTYPE
from .enum_array import EnumArray

log = logging.getLogger(__name__)


class Enum(enum.Enum):
    """
    Enum based on `enum34 <https://pypi.python.org/pypi/enum34/>`_, whose items
    have an index.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize an Enum item with a name and an index.

        The index is automatically assigned based on the order of the Enum items.
        """
        self.index = len(self._member_names_)

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    @classmethod
    @lru_cache(maxsize=None)
    def _get_sorted_lookup_arrays(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Build cached sorted arrays for fast searchsorted-based lookup."""
        name_to_index = {item.name: item.index for item in cls}
        sorted_names = sorted(name_to_index.keys())
        sorted_names_arr = np.array(sorted_names)
        sorted_indices = np.array(
            [name_to_index[n] for n in sorted_names], dtype=ENUM_ARRAY_DTYPE
        )
        return sorted_names_arr, sorted_indices

    @classmethod
    def encode(cls, array: Union[EnumArray, np.ndarray]) -> EnumArray:
        """
        Encode an array of enum items or string identifiers into an EnumArray.

        Args:
            array: The input array to encode. Can be an EnumArray, a NumPy array
                of enum items, or a NumPy array of string identifiers.

        Returns:
            An EnumArray containing the encoded values.

        Examples:
            >>> string_array = np.array(["ITEM_1", "ITEM_2", "ITEM_3"])
            >>> encoded_array = MyEnum.encode(string_array)
            >>> encoded_array
            EnumArray([1, 2, 3], dtype=int8)

            >>> item_array = np.array([MyEnum.ITEM_1, MyEnum.ITEM_2, MyEnum.ITEM_3])
            >>> encoded_array = MyEnum.encode(item_array)
            >>> encoded_array
            EnumArray([1, 2, 3], dtype=int8)
        """
        if isinstance(array, EnumArray):
            return array

        # Handle Enum item arrays by extracting indices directly
        if len(array) > 0 and isinstance(array[0], Enum):
            indices = np.array(
                [item.index for item in array], dtype=ENUM_ARRAY_DTYPE
            )
            return EnumArray(indices, cls)

        # Convert byte-strings or object arrays to Unicode strings
        if array.dtype.kind == "S" or array.dtype == object:
            array = array.astype(str)

        if isinstance(array, np.ndarray) and array.dtype.kind in {"U", "S"}:
            # String array - use searchsorted for O(n log m) lookup
            sorted_names, sorted_indices = cls._get_sorted_lookup_arrays()
            positions = np.searchsorted(sorted_names, array)
            # Clip positions to valid range to avoid IndexError
            positions = np.clip(positions, 0, len(sorted_names) - 1)
            # For non-matches, return 0 (first enum value) to match old np.select behaviour
            matches = sorted_names[positions] == array
            indices = np.where(matches, sorted_indices[positions], 0)
            # Raise error for invalid values to prevent silent data corruption
            invalid_mask = ~matches
            if np.any(invalid_mask):
                invalid_values = np.unique(array[invalid_mask])
                valid_names = [item.name for item in cls]
                raise ValueError(
                    f"Invalid value(s) {invalid_values.tolist()} for enum "
                    f"{cls.__name__}. Valid values are: {valid_names}"
                )
        elif array.dtype.kind in {"i", "u"}:
            # Integer array - already indices
            indices = array
        else:
            raise ValueError(f"Unsupported array dtype: {array.dtype}")

        return EnumArray(indices.astype(ENUM_ARRAY_DTYPE), cls)
