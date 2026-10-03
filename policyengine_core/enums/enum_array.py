from __future__ import annotations

import importlib
import operator
import typing
from typing import Any, NoReturn, Optional, Tuple, Type

import numpy

if typing.TYPE_CHECKING:
    from policyengine_core.enums import Enum


def _restore_enum_array(
    array: numpy.ndarray, enum_name: Optional[Tuple[str, str]]
) -> EnumArray:
    """Rebuild a pickled EnumArray, with its enum if this process can find it.

    ``enum_name`` is the enum's module and qualified name, or ``None`` for an
    array that had no enum.

    Tax-benefit systems load variable files under module names that exist only
    in the process that loaded them, so an enum defined in one cannot be found
    from another process. The array then comes back with ``possible_values``
    unset (``None``), where pickling the enum by reference would fail to
    unpickle the array at all.
    """
    possible_values = None
    if enum_name is not None:
        module_name, qualified_name = enum_name
        try:
            module = importlib.import_module(module_name)
            possible_values = operator.attrgetter(qualified_name)(module)
        except (ImportError, AttributeError):
            pass
    return EnumArray(array, possible_values)


class EnumArray(numpy.ndarray):
    """
    Numpy array subclass representing an array of enum items.

    EnumArrays are encoded as ``int`` arrays to improve performance
    """

    # Subclassing ndarray is a little tricky.
    # To read more about the two following methods, see:
    # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array.
    def __new__(
        cls,
        input_array: numpy.int_,
        possible_values: Optional[Type[Enum]] = None,
    ) -> EnumArray:
        obj = numpy.asarray(input_array).view(cls)
        obj.possible_values = possible_values
        return obj

    # See previous comment
    def __array_finalize__(self, obj: Optional[numpy.int_]) -> None:
        if obj is None:
            return

        self.possible_values = getattr(obj, "possible_values", None)

    def __reduce__(self) -> tuple:
        # ndarray's own ``__reduce__`` rebuilds the array without
        # ``possible_values``, so an unpickled EnumArray could be neither
        # decoded nor compared with an enum item. The enum travels by name
        # rather than by reference; see ``_restore_enum_array``.
        enum = self.possible_values
        name = None if enum is None else (enum.__module__, enum.__qualname__)
        return _restore_enum_array, (self.view(numpy.ndarray), name)

    def __eq__(self, other: Any) -> bool:
        # When comparing to an item of self.possible_values, use the item index
        # to speed up the comparison.
        if other.__class__.__name__ is self.possible_values.__name__:
            # Use view(ndarray) so that the result is a classic ndarray, not an
            # EnumArray.
            return self.view(numpy.ndarray) == other.index

        return self.view(numpy.ndarray) == other

    def __ne__(self, other: Any) -> bool:
        return numpy.logical_not(self == other)

    def _forbidden_operation(self, other: Any) -> NoReturn:
        raise TypeError(
            "Forbidden operation. The only operations allowed on EnumArrays "
            "are '==' and '!='.",
        )

    __add__ = _forbidden_operation
    __mul__ = _forbidden_operation
    __lt__ = _forbidden_operation
    __le__ = _forbidden_operation
    __gt__ = _forbidden_operation
    __ge__ = _forbidden_operation
    __and__ = _forbidden_operation
    __or__ = _forbidden_operation

    def decode(self) -> numpy.object_:
        """
        Return the array of enum items corresponding to self.

        For instance:

        >>> enum_array = household('housing_occupancy_status', period)
        >>> enum_array[0]
        >>> 2  # Encoded value
        >>> enum_array.decode()[0]
        <HousingOccupancyStatus.free_lodger: 'Free lodger'>

        Decoded value: enum item
        """
        return numpy.select(
            [self == item.index for item in self.possible_values],
            list(self.possible_values),
        )

    def decode_to_str(self) -> numpy.str_:
        """
        Return the array of string identifiers corresponding to self.

        For instance:

        >>> enum_array = household('housing_occupancy_status', period)
        >>> enum_array[0]
        >>> 2  # Encoded value
        >>> enum_array.decode_to_str()[0]
        'free_lodger'  # String identifier
        """
        return numpy.select(
            [self == item.index for item in self.possible_values],
            [str(item.name) for item in self.possible_values],
            default="unknown",
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.decode())})"

    def __str__(self) -> str:
        return str(self.decode_to_str())
