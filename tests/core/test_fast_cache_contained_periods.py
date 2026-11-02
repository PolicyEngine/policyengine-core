"""Deleting or setting a period drops every period it contains from the fast cache.

Holder storage deletes each period the requested one contains (deleting
"2012" also deletes "2012-02"), and an ETERNITY variable keeps one value for
every period. ``Simulation._fast_cache`` used to drop only the exact
``(variable, period)`` key, so ``calculate`` kept returning values storage no
longer held. The property versions of these tests are in
``test_fast_cache_contained_periods_property.py``.
"""

from __future__ import annotations

import numpy as np

from policyengine_core import periods
from tests.fixtures.fast_cache_contained_periods import build_simulation


def test_deleting_a_year_drops_cached_months():
    """The carry-over review's repro: a deleted month must not come back from the cache."""
    simulation = build_simulation()
    simulation.set_input("monthly_input", "2012-01", [100])
    np.testing.assert_array_equal(
        simulation.calculate("monthly_input", "2012-02"), [100]
    )

    simulation.delete_arrays("monthly_input", "2012")

    assert simulation.get_array("monthly_input", "2012-01") is None
    assert simulation.get_array("monthly_input", "2012-02") is None
    simulation.set_input("monthly_input", "2012-01", [200])
    fresh = build_simulation()
    fresh.set_input("monthly_input", "2012-01", [200])
    np.testing.assert_array_equal(
        simulation.calculate("monthly_input", "2012-02"),
        fresh.calculate("monthly_input", "2012-02"),
    )
    np.testing.assert_array_equal(
        simulation.calculate("monthly_input", "2012-02"), [200]
    )


def test_deleting_a_year_keeps_cached_values_it_does_not_contain():
    """Only what storage deletes leaves the fast cache."""
    simulation = build_simulation()
    simulation.set_input("monthly_input", "2012-01", [100])
    february = simulation.calculate("monthly_input", "2012-02")
    january_next_year = simulation.calculate("monthly_input", "2013-01")
    other_variable = simulation.calculate("monthly_formula", "2012-02")

    simulation.delete_arrays("monthly_input", "2012")

    cache = simulation._fast_cache
    assert ("monthly_input", periods.period("2012-02")) not in cache
    assert cache[("monthly_input", periods.period("2013-01"))] is january_next_year
    assert cache[("monthly_formula", periods.period("2012-02"))] is other_variable
    assert february is not None


def test_deleting_a_month_keeps_the_rest_of_the_year():
    simulation = build_simulation()
    simulation.set_input("monthly_input", "2012-01", [100])
    simulation.calculate("monthly_input", "2012-02")
    march = simulation.calculate("monthly_input", "2012-03")

    simulation.delete_arrays("monthly_input", "2012-02")

    assert ("monthly_input", periods.period("2012-02")) not in simulation._fast_cache
    assert simulation._fast_cache[("monthly_input", periods.period("2012-03"))] is march


def test_deleting_an_eternity_variable_at_any_period_drops_every_cached_period():
    """Storage keeps one ETERNITY value, so deleting it at "2012" also clears "2013"."""
    simulation = build_simulation(carry_over=False)
    simulation.set_input("eternal_input", "2012", [5])
    np.testing.assert_array_equal(simulation.calculate("eternal_formula", "2013"), [10])

    simulation.delete_arrays("eternal_formula", "2012")
    simulation.set_input("eternal_input", "2012", [7])

    assert simulation.get_array("eternal_formula", "2013") is None
    np.testing.assert_array_equal(simulation.calculate("eternal_formula", "2013"), [14])


def test_setting_an_eternity_input_at_one_period_drops_it_at_every_period():
    simulation = build_simulation(carry_over=False)
    np.testing.assert_array_equal(simulation.calculate("eternal_input", "2012"), [0])

    simulation.set_input("eternal_input", "2013", [5])

    np.testing.assert_array_equal(simulation.get_array("eternal_input", "2012"), [5])
    np.testing.assert_array_equal(simulation.calculate("eternal_input", "2012"), [5])


def test_setting_a_year_split_over_months_drops_cached_months():
    """A ``set_input`` handler stores every month of an annual input.

    A variable the cache blacklist keeps out of storage still has its
    calculated months in the fast cache, so those must go too.
    """
    simulation = build_simulation(carry_over=False)
    simulation.opt_out_cache = True
    simulation.tax_benefit_system.cache_blacklist = {"monthly_input_split"}
    np.testing.assert_array_equal(
        simulation.calculate("monthly_input_split", "2012-02"), [0]
    )

    simulation.set_input("monthly_input_split", "2012", [1200])

    np.testing.assert_array_equal(
        simulation.get_array("monthly_input_split", "2012-02"), [100]
    )
    np.testing.assert_array_equal(
        simulation.calculate("monthly_input_split", "2012-02"), [100]
    )


def test_purging_an_invalidated_year_drops_cached_months():
    """``purge_cache_of_invalid_values`` deletes like ``delete_arrays`` does."""
    simulation = build_simulation(carry_over=False)
    simulation.set_input("monthly_input", "2012-02", [10])
    np.testing.assert_array_equal(
        simulation.calculate("monthly_formula", "2012-02"), [11]
    )

    simulation.invalidated_caches.add(("monthly_formula", periods.period("2012")))
    simulation.purge_cache_of_invalid_values()
    simulation.set_input("monthly_input", "2012-02", [20])

    assert simulation.get_array("monthly_formula", "2012-02") is None
    np.testing.assert_array_equal(
        simulation.calculate("monthly_formula", "2012-02"), [21]
    )
