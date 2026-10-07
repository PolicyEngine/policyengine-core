from __future__ import annotations

import copy
from threading import RLock
from typing import Any


class StaleParameterViewError(RuntimeError):
    """Raised when a lazy dated view outlives its source policy revision."""


class ParameterTreeRevision:
    """Thread-safe monotonic revision shared by one parameter tree.

    Dated parameter views capture a numeric snapshot. Policy mutations advance
    the shared revision, allowing caches to discard obsolete entries and lazy
    views to reject access instead of mixing values from different policies.
    """

    def __init__(self, value: int = 0) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("a parameter revision must be a non-negative integer")
        self._value = value
        self._lock = RLock()

    @property
    def current(self) -> int:
        with self._lock:
            return self._value

    def advance(self) -> int:
        """Advance the revision once and return the new value."""

        with self._lock:
            self._value += 1
            return self._value

    def validate(self, captured: int) -> None:
        """Reject a snapshot that no longer describes this parameter tree."""

        current = self.current
        if captured != current:
            raise StaleParameterViewError(
                "parameter view revision "
                f"{captured} does not match current revision {current}"
            )

    def __copy__(self) -> ParameterTreeRevision:
        return type(self)(self.current)

    def __deepcopy__(self, memo: dict[int, Any]) -> ParameterTreeRevision:
        duplicate = type(self)(self.current)
        memo[id(self)] = duplicate
        return duplicate

    def __getstate__(self) -> dict[str, int]:
        return {"value": self.current}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(state["value"])

    def clone(self) -> ParameterTreeRevision:
        """Return an independent revision with the same numeric value."""

        return copy.copy(self)
