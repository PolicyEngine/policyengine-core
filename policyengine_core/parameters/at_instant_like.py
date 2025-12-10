import abc
from typing import Any

from policyengine_core import periods
from policyengine_core.periods import Instant

# Cache for instant -> string conversions used in get_at_instant
_instant_str_cache: dict = {}


class AtInstantLike(abc.ABC):
    """
    Base class for various types of parameters implementing the at instant protocol.
    """

    def __call__(self, instant: Instant) -> Any:
        return self.get_at_instant(instant)

    def get_at_instant(self, instant: Instant) -> Any:
        # Fast path for Instant objects - use their __str__ which is cached
        if isinstance(instant, Instant):
            return self._get_at_instant(str(instant))

        # For other types, use a cache to avoid repeated conversions
        # Create a hashable cache key
        cache_key = None
        if isinstance(instant, str):
            cache_key = instant
        elif isinstance(instant, tuple):
            cache_key = instant
        elif isinstance(instant, int):
            cache_key = (instant,)
        elif hasattr(instant, "year"):  # datetime.date
            cache_key = (instant.year, instant.month, instant.day)

        if cache_key is not None:
            cached_str = _instant_str_cache.get(cache_key)
            if cached_str is not None:
                return self._get_at_instant(cached_str)
            instant_obj = periods.instant(instant)
            instant_str = str(instant_obj)
            _instant_str_cache[cache_key] = instant_str
            return self._get_at_instant(instant_str)

        # Fallback for other types (Period, list, etc.)
        instant_str = str(periods.instant(instant))
        return self._get_at_instant(instant_str)

    @abc.abstractmethod
    def _get_at_instant(self, instant): ...
