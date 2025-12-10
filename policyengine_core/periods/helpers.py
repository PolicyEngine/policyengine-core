import datetime
import os
from functools import lru_cache

from policyengine_core import periods
from policyengine_core.periods import config

# Global cache for instant objects to avoid repeated tuple creation
_instant_cache: dict = {}


@lru_cache(maxsize=10000)
def _instant_from_string(instant_str: str) -> "periods.Instant":
    """Cached parsing of instant strings."""
    if not config.INSTANT_PATTERN.match(instant_str):
        raise ValueError(
            "'{}' is not a valid instant. Instants are described using the 'YYYY-MM-DD' format, for instance '2015-06-15'.".format(
                instant_str
            )
        )
    parts = instant_str.split("-", 2)[:3]
    if len(parts) == 1:
        return periods.Instant((int(parts[0]), 1, 1))
    elif len(parts) == 2:
        return periods.Instant((int(parts[0]), int(parts[1]), 1))
    else:
        return periods.Instant((int(parts[0]), int(parts[1]), int(parts[2])))


def instant(instant):
    """Return a new instant, aka a triple of integers (year, month, day).

    >>> instant(2014)
    Instant((2014, 1, 1))
    >>> instant('2014')
    Instant((2014, 1, 1))
    >>> instant('2014-02')
    Instant((2014, 2, 1))
    >>> instant('2014-3-2')
    Instant((2014, 3, 2))
    >>> instant(instant('2014-3-2'))
    Instant((2014, 3, 2))
    >>> instant(period('month', '2014-3-2'))
    Instant((2014, 3, 2))

    >>> instant(None)
    """
    if instant is None:
        return None
    if isinstance(instant, periods.Instant):
        return instant
    if isinstance(instant, str):
        return _instant_from_string(instant)

    # For other types, create a cache key and check the cache
    cache_key = None
    # Check Period before tuple since Period is a subclass of tuple
    if isinstance(instant, periods.Period):
        return instant.start
    elif isinstance(instant, datetime.date):
        cache_key = (instant.year, instant.month, instant.day)
    elif isinstance(instant, int):
        cache_key = (instant, 1, 1)
    elif isinstance(instant, (tuple, list)):
        if len(instant) == 1:
            cache_key = (instant[0], 1, 1)
        elif len(instant) == 2:
            cache_key = (instant[0], instant[1], 1)
        elif len(instant) == 3:
            cache_key = tuple(instant)

    if cache_key is not None:
        cached = _instant_cache.get(cache_key)
        if cached is not None:
            return cached
        result = periods.Instant(cache_key)
        _instant_cache[cache_key] = result
        return result

    # Fallback for unexpected types
    assert isinstance(instant, tuple), instant
    assert 1 <= len(instant) <= 3
    if len(instant) == 1:
        return periods.Instant((instant[0], 1, 1))
    if len(instant) == 2:
        return periods.Instant((instant[0], instant[1], 1))
    return periods.Instant(instant)


def instant_date(instant):
    if instant is None:
        return None
    instant_date = config.date_by_instant_cache.get(instant)
    if instant_date is None:
        config.date_by_instant_cache[instant] = instant_date = datetime.date(
            *instant
        )
    return instant_date


@lru_cache(maxsize=1024)
def _parse_simple_period(value: str):
    """Cached parsing of simple periods respecting the ISO format."""
    try:
        date = datetime.datetime.strptime(value, "%Y")
    except ValueError:
        try:
            date = datetime.datetime.strptime(value, "%Y-%m")
        except ValueError:
            try:
                date = datetime.datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                return None
            else:
                return periods.Period(
                    (
                        config.DAY,
                        periods.Instant((date.year, date.month, date.day)),
                        1,
                    )
                )
        else:
            return periods.Period(
                (
                    config.MONTH,
                    periods.Instant((date.year, date.month, 1)),
                    1,
                )
            )
    else:
        return periods.Period(
            (config.YEAR, periods.Instant((date.year, date.month, 1)), 1)
        )


def _raise_period_error(value):
    message = os.linesep.join(
        [
            "Expected a period (eg. '2017', '2017-01', '2017-01-01', ...); got: '{}'.".format(
                value
            ),
            "Learn more about legal period formats in OpenFisca:",
            "<https://openfisca.org/doc/coding-the-legislation/35_periods.html#periods-in-simulations>.",
        ]
    )
    raise ValueError(message)


@lru_cache(maxsize=1024)
def _period_from_string(value: str) -> "periods.Period":
    """Cached parsing of period strings."""
    # try to parse as a simple period
    result = _parse_simple_period(value)
    if result is not None:
        return result

    # complex period must have a ':' in their strings
    if ":" not in value:
        _raise_period_error(value)

    components = value.split(":")

    # left-most component must be a valid unit
    unit = components[0]
    if unit not in (config.DAY, config.MONTH, config.YEAR):
        _raise_period_error(value)

    # middle component must be a valid iso period
    base_period = _parse_simple_period(components[1])
    if not base_period:
        _raise_period_error(value)

    # period like year:2015-03 have a size of 1
    if len(components) == 2:
        size = 1
    # if provided, make sure the size is an integer
    elif len(components) == 3:
        try:
            size = int(components[2])
        except ValueError:
            _raise_period_error(value)
    # if there is more than 2 ":" in the string, the period is invalid
    else:
        _raise_period_error(value)

    # reject ambiguous period such as month:2014
    if unit_weight(base_period.unit) > unit_weight(unit):
        _raise_period_error(value)

    return periods.Period((unit, base_period.start, size))


def period(value):
    """Return a new period, aka a triple (unit, start_instant, size).

    >>> period('2014')
    Period((YEAR, Instant((2014, 1, 1)), 1))
    >>> period('year:2014')
    Period((YEAR, Instant((2014, 1, 1)), 1))

    >>> period('2014-2')
    Period((MONTH, Instant((2014, 2, 1)), 1))
    >>> period('2014-02')
    Period((MONTH, Instant((2014, 2, 1)), 1))
    >>> period('month:2014-2')
    Period((MONTH, Instant((2014, 2, 1)), 1))

    >>> period('year:2014-2')
    Period((YEAR, Instant((2014, 2, 1)), 1))
    """
    if isinstance(value, periods.Period):
        return value

    if isinstance(value, periods.Instant):
        return periods.Period((config.DAY, value, 1))

    if value == "ETERNITY" or value == config.ETERNITY:
        return periods.Period(
            ("eternity", instant(datetime.date.min), float("inf"))
        )

    # check the type
    if isinstance(value, int):
        return periods.Period((config.YEAR, periods.Instant((value, 1, 1)), 1))

    if isinstance(value, str):
        return _period_from_string(value)

    _raise_period_error(value)


def key_period_size(period):
    """
    Defines a key in order to sort periods by length. It uses two aspects : first unit then size

    :param period: an OpenFisca period
    :return: a string

    >>> key_period_size(period('2014'))
    '2_1'
    >>> key_period_size(period('2013'))
    '2_1'
    >>> key_period_size(period('2014-01'))
    '1_1'

    """

    unit, start, size = period

    return "{}_{}".format(unit_weight(unit), size)


def unit_weights():
    return {
        config.DAY: 100,
        config.MONTH: 200,
        config.YEAR: 300,
        config.ETERNITY: 400,
    }


def unit_weight(unit):
    return unit_weights()[unit]
