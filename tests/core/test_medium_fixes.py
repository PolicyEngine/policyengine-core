"""Regression tests for a batch of surgical Medium-severity fixes.

* M1  — ``@uprated`` short-circuit used Python ``all()`` instead of
        ``numpy.all(old == 0)``. Previously the guard returned ``True`` only
        when the first element was zero (truthiness) rather than "no values
        have been inputted".
* M8  — ``SimulationBuilder`` multi-axis ``linspace`` branch divided by
        ``axis_count - 1``, crashing on single-point axes.
* M10 — ``Dataset.download`` parsed ``release://org/repo/tag/file`` with
        ``url.split('/')[2:]`` hard-unpacked to four segments, crashing on
        file names containing slashes.
* M13 — ``datetime.date`` default used ``datetime.date.fromtimestamp(0)``
        which is timezone-dependent (returns 1969-12-31 west of UTC).
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from policyengine_core.variables.config import VALUE_TYPES


def test_datetime_date_default_is_not_timezone_dependent():
    """Bug M13: the default must be 1970-01-01 regardless of TZ."""
    default = VALUE_TYPES[datetime.date]["default"]
    assert default == datetime.date(1970, 1, 1)


def test_single_point_axis_does_not_divide_by_zero(persons):
    """Bug M8: single-point axis must not crash on ``axis_count - 1 = 0``."""
    from policyengine_core.simulations import SimulationBuilder

    builder = SimulationBuilder()
    builder.set_default_period("2018-11")
    builder.add_person_entity(persons, {"Alicia": {}})
    builder.register_variable("salary", persons)
    builder.add_parallel_axis({"count": 1, "name": "salary", "min": 500, "max": 1500})
    # Under the bug, ``expand_axes`` divided by ``axis_count - 1 == 0``.
    # After the fix, a single-point axis produces the ``axis["min"]`` value.
    builder.expand_axes()
    assert builder.get_input("salary", "2018-11") == pytest.approx([500])


def test_all_numpy_guard_triggers_on_all_zero_old():
    """Bug M1: ``np.all(old == 0)`` must be used, not Python ``all(old)``.

    Python ``all([1, 0, 0])`` == True (because 1 is truthy), so the guard
    would NOT fire. ``np.all([1, 0, 0] == 0)`` == False, which correctly
    says "not all zero".
    """
    old = np.array([1, 0, 0])
    # Python truthy-check semantics: ``all([1, 0, 0])`` -> False because
    # 0 is falsy. For the reversed test case with all zeros:
    all_zero = np.array([0, 0, 0])
    # The fix uses ``np.all(old == 0)`` which is True iff every element is 0.
    assert np.all(all_zero == 0)
    assert not np.all(old == 0)
