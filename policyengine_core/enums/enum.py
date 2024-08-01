from __future__ import annotations
import enum
from typing import Union
import numpy as np
from .config import ENUM_ARRAY_DTYPE
from .enum_array import EnumArray


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

        # First, convert byte-string arrays to Unicode-string arrays
        # Confusingly, Numpy uses "S" to refer to byte-string arrays
        # and "U" to refer to Unicode-string arrays, which are also
        # referred to as the "str" type
        if isinstance(array[0], Enum):
            array = np.array([item.name for item in array])
        if array.dtype.kind == "S" or array.dtype == object:
            # Convert boolean array to string array
            array = array.astype(str)
        if isinstance(array, np.ndarray) and array.dtype.kind in {"U", "S"}:
            # String array
            indices = np.select(
                [array == item.name for item in cls],
                [item.index for item in cls],
            )
        elif isinstance(array, np.ndarray) and array.dtype.kind == "O":
            # Enum items array
            if len(array) > 0:
                first_item = array[0]
                if cls.__name__ == type(first_item).__name__:
                    # Use the same Enum class as the array items
                    cls = type(first_item)
            indices = np.select(
                [array == item for item in cls],
                [item.index for item in cls],
            )
        elif array.dtype.kind in {"i", "u"}:
            # Integer array
            indices = array
        else:
            raise ValueError(f"Unsupported array dtype: {array.dtype}")

        return EnumArray(indices.astype(ENUM_ARRAY_DTYPE), cls)
