"""Unit tests for the pure period-key helpers behind ``Reform.from_dict``."""

import pytest

from policyengine_core.periods import Instant, instant, period
from policyengine_core.reforms.reform import (
    _api_period_range,
    _bounded_period_update_kwargs,
    _eternity_update_kwargs,
    _from_instant_update_kwargs,
    _period_key_update_kwargs,
    _range_update_kwargs,
    _update_start_instant,
)


# ---- individual per-format builders ---------------------------------------


def test_eternity_update_kwargs():
    assert _eternity_update_kwargs(0.5) == {"value": 0.5}


def test_range_update_kwargs():
    assert _range_update_kwargs("2026-01-01.2027-12-31", 0.5) == {
        "start": instant("2026-01-01"),
        "stop": instant("2027-12-31"),
        "value": 0.5,
    }


def test_bounded_period_update_kwargs():
    assert _bounded_period_update_kwargs("year:2026:5", 0.5) == {
        "period": period("year:2026:5"),
        "value": 0.5,
    }


def test_from_instant_update_kwargs():
    assert _from_instant_update_kwargs("2026-01-01", 0.5) == {
        "start": instant("2026-01-01"),
        "value": 0.5,
    }


# ---- dispatcher: classification + int coercion ----------------------------


@pytest.mark.parametrize(
    "period_key, expected",
    [
        ("ETERNITY", {"value": 0.5}),
        ("2026", {"start": Instant((2026, 1, 1)), "value": 0.5}),
        ("2026-01", {"start": Instant((2026, 1, 1)), "value": 0.5}),
        ("2026-01-01", {"start": Instant((2026, 1, 1)), "value": 0.5}),
        ("2026-06-15", {"start": Instant((2026, 6, 15)), "value": 0.5}),
        (
            "2026-01-01.2027-12-31",
            {
                "start": Instant((2026, 1, 1)),
                "stop": Instant((2027, 12, 31)),
                "value": 0.5,
            },
        ),
        # Non-string key is coerced to str rather than raising.
        (2026, {"start": Instant((2026, 1, 1)), "value": 0.5}),
    ],
)
def test_period_key_update_kwargs_dispatch(period_key, expected):
    assert _period_key_update_kwargs(period_key, 0.5) == expected


def test_period_key_update_kwargs_compound_period():
    assert _period_key_update_kwargs("year:2026:5", 0.5) == {
        "period": period("year:2026:5"),
        "value": 0.5,
    }


def test_period_key_update_kwargs_malformed_raises():
    with pytest.raises(ValueError):
        _period_key_update_kwargs("not-a-period", 0.5)


# ---- sort key -------------------------------------------------------------


@pytest.mark.parametrize(
    "update_kwargs, expected",
    [
        ({"start": instant("2026-01-01"), "value": 1}, Instant((2026, 1, 1))),
        (
            {
                "start": instant("2026-01-01"),
                "stop": instant("2027-12-31"),
                "value": 1,
            },
            Instant((2026, 1, 1)),
        ),
        ({"period": period("year:2026:5"), "value": 1}, Instant((2026, 1, 1))),
        # Value-only (ETERNITY) sorts first as the all-time base.
        ({"value": 1}, Instant((1, 1, 1))),
    ],
)
def test_update_start_instant(update_kwargs, expected):
    assert _update_start_instant(update_kwargs) == expected


# ---- from_api range normalization -----------------------------------------


def test_api_period_range_single_year_stays_bounded():
    assert _api_period_range("2026-01-01", "2026-12-31") == "2026-01-01.2026-12-31"


def test_api_period_range_multi_year_stays_bounded():
    assert _api_period_range("2026-01-01", "2030-12-31") == "2026-01-01.2030-12-31"


def test_api_period_range_out_of_window_raises():
    with pytest.raises(ValueError):
        _api_period_range("2200-01-01", "2300-01-01")
