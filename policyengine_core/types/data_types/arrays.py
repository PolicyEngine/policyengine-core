from typing import Sequence, TypeVar, Any
from numpy.typing import NDArray as ArrayType

T = TypeVar("T", bool, bytes, float, int, object, str)

ArrayLike = Sequence[T]

""":obj:`typing.Generic`: Type of any castable to :class:`numpy.ndarray`.

These include any :obj:`numpy.ndarray` and sequences (like
:obj:`list`, :obj:`tuple`, and so on).

Examples:
    >>> ArrayLike[float]
    typing.Union[numpy.ndarray, typing.Sequence[float]]

    >>> ArrayLike[str]
    typing.Union[numpy.ndarray, typing.Sequence[str]]

Note:
    `mypy`_ provides `duck type compatibility`_, so an :obj:`int` is
    considered to be valid whenever a :obj:`float` is expected.

.. versionadded:: 35.5.0

.. versionchanged:: 35.6.0
    Moved to :mod:`.types`

.. _mypy:
    https://mypy.readthedocs.io/en/stable/

.. _duck type compatibility:
    https://mypy.readthedocs.io/en/stable/duck_type_compatibility.html

.. _numpy.typing.NDArray:
    https://numpy.org/doc/stable/reference/typing.html#numpy.typing.NDArray

"""
