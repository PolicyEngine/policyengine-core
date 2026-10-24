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
    def _validate_integer_indices(cls, values) -> None:
        """Validate zero-based positions before a scalar or array is narrowed."""
        if isinstance(values, (int, np.integer)):
            if isinstance(values, (bool, np.bool_)):
                return
            if 0 <= values < len(cls):
                return
            invalid_values = [int(values)]
        else:
            raw = (
                values.view(np.ndarray)
                if isinstance(values, EnumArray)
                else np.asarray(values)
            )
            if raw.dtype.kind not in {"i", "u"}:
                return
            invalid = (raw < 0) | (raw >= len(cls))
            if not np.any(invalid):
                return
            invalid_values = np.unique(raw[invalid]).tolist()

        shown = invalid_values[:5]
        remainder = len(invalid_values) - len(shown)
        suffix = f" (and {remainder} more)" if remainder else ""
        raise ValueError(
            f"Invalid position(s) {shown}{suffix} for enum {cls.__name__}. "
            f"Integer positions are zero-based and must be in [0, {len(cls)})."
        )

    @classmethod
    def encode(cls, array: Union[EnumArray, np.ndarray]) -> EnumArray:
        """
        Encode enum items, names, or zero-based integer positions into an EnumArray.

        Args:
            array: The input array to encode. Can be an EnumArray, a NumPy array
                of enum items, string identifiers, or integer positions. Integer
                positions follow declaration order, from zero through one less
                than the number of members; they are not survey source codes.

        Returns:
            An EnumArray containing the encoded values.

        Raises:
            ValueError: An integer position or string identifier is invalid.

        Examples:
            >>> class MyEnum(Enum):
            ...     ITEM_1 = "First"
            ...     ITEM_2 = "Second"
            ...     ITEM_3 = "Third"
            >>> string_array = np.array(["ITEM_1", "ITEM_2", "ITEM_3"])
            >>> encoded_array = MyEnum.encode(string_array)
            >>> encoded_array.view(np.ndarray).tolist()
            [0, 1, 2]

            >>> item_array = np.array([MyEnum.ITEM_1, MyEnum.ITEM_2, MyEnum.ITEM_3])
            >>> encoded_array = MyEnum.encode(item_array)
            >>> encoded_array.view(np.ndarray).tolist()
            [0, 1, 2]
        """
        if isinstance(array, EnumArray):
            cls._validate_integer_indices(array)
            return array

        # Handle Enum item arrays by extracting indices directly
        # Use .iloc[0] for pandas Series to avoid KeyError with non-integer index
        # (pandas 3.0 uses StringDtype by default, causing array[0] to do
        # label-based lookup instead of positional access)
        if len(array) > 0:
            first_elem = array.iloc[0] if hasattr(array, "iloc") else array[0]
        else:
            first_elem = None
        if first_elem is not None and isinstance(first_elem, Enum):
            indices = np.array([item.index for item in array], dtype=ENUM_ARRAY_DTYPE)
            return EnumArray(indices, cls)

        # Convert fixed-width byte strings, as returned by h5py for string
        # datasets, to Unicode before matching enum names.
        if array.dtype.kind == "S":
            array = np.char.decode(array, "utf-8")
        elif array.dtype == object:
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
            cls._validate_integer_indices(array)
            indices = array
        else:
            raise ValueError(f"Unsupported array dtype: {array.dtype}")

        return EnumArray(indices.astype(ENUM_ARRAY_DTYPE), cls)
