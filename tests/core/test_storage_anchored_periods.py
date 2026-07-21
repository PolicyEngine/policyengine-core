"""Regression tests for storage key parsing with anchored periods (#523).

A year period anchored at a non-January month stringifies with colons
(e.g. ``"year:2027-11"``), so storage keys look like
``"default:year:2027-11"``. ``get_known_periods`` and
``get_known_branch_periods`` previously split those keys on every colon,
crashing on the literal ``"year"`` (or mis-unpacking the tuple). This is
hit in practice whenever a year-defined input variable is set at a
non-January month test period.
"""

from __future__ import annotations

import numpy as np

from policyengine_core import periods
from policyengine_core.data_storage.in_memory_storage import InMemoryStorage


def test_get_known_periods_with_anchored_year_period():
    storage = InMemoryStorage(is_eternal=False)
    anchored = periods.period("year:2027-11")
    storage.put(np.asarray([1.0]), anchored, branch_name="default")
    storage.put(np.asarray([2.0]), "2024-01", branch_name="default")
    known = storage.get_known_periods()
    assert anchored in known
    assert periods.period("2024-01") in known


def test_get_known_branch_periods_with_anchored_year_period():
    storage = InMemoryStorage(is_eternal=False)
    anchored = periods.period("year:2027-11")
    storage.put(np.asarray([1.0]), anchored, branch_name="reform")
    assert storage.get_known_branch_periods() == [("reform", anchored)]


def test_put_rejects_colon_branch_names():
    import pytest

    storage = InMemoryStorage(is_eternal=False)
    with pytest.raises(ValueError, match="526"):
        storage.put(np.asarray([1.0]), "2024-01", branch_name="my:reform")


def test_put_rejects_mid_month_anchored_periods():
    import pytest

    from policyengine_core.periods import Instant, Period, YEAR

    storage = InMemoryStorage(is_eternal=False)
    day_anchored = Period((YEAR, Instant((2027, 11, 15)), 2))
    with pytest.raises(ValueError, match="lossy"):
        storage.put(np.asarray([1.0]), day_anchored, branch_name="default")


def test_eternal_storage_round_trips():
    storage = InMemoryStorage(is_eternal=True)
    storage.put(np.asarray([1.0]), "2024-01", branch_name="default")
    from policyengine_core import periods as p

    assert storage.get_known_periods() == [p.period(p.ETERNITY)]
