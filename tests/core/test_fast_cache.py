"""Tests for the _fast_cache mechanism in Simulation."""

import numpy as np
from policyengine_core.simulations import SimulationBuilder


def _make_simulation(tax_benefit_system, salary=3000):
    """Build a simple simulation with one person and a salary."""
    return SimulationBuilder().build_from_entities(
        tax_benefit_system,
        {
            "persons": {
                "bill": {"salary": {"2017-01": salary}},
            },
            "households": {"household": {"parents": ["bill"]}},
        },
    )


def test_fast_cache_returns_cached_value(tax_benefit_system):
    """Second calculate() for a computed variable should return
    the cached value from _fast_cache without recomputing."""
    sim = _make_simulation(tax_benefit_system)

    # income_tax is computed via formula, so it enters _fast_cache
    result1 = sim.calculate("income_tax", "2017-01")
    assert len(sim._fast_cache) > 0

    result2 = sim.calculate("income_tax", "2017-01")
    # Must be the exact same object (identity check proves cache hit)
    assert result1 is result2


def test_fast_cache_invalidated_after_set_input(tax_benefit_system):
    """set_input() must evict the stale _fast_cache entry so the next
    calculate() returns the new value."""
    sim = _make_simulation(tax_benefit_system)

    # Populate the cache with a computed variable
    result1 = sim.calculate("income_tax", "2017-01")
    assert len(sim._fast_cache) > 0
    old_val = result1[0]

    # Overwrite income_tax with a direct value
    sim.set_input("income_tax", "2017-01", np.array([9999.0]))

    # The cache entry for income_tax must be gone
    result2 = sim.calculate("income_tax", "2017-01")
    assert np.isclose(result2[0], 9999.0), (
        f"Expected 9999.0 after set_input, got {result2[0]} (stale cache bug)"
    )


def test_fast_cache_invalidated_after_delete_arrays_with_period(
    tax_benefit_system,
):
    """delete_arrays(variable, period) must evict that specific
    _fast_cache entry."""
    sim = _make_simulation(tax_benefit_system)

    sim.calculate("income_tax", "2017-01")
    assert len(sim._fast_cache) > 0

    sim.delete_arrays("income_tax", "2017-01")

    matching = [k for k in sim._fast_cache if k[0] == "income_tax"]
    assert len(matching) == 0


def test_fast_cache_invalidated_after_delete_arrays_all_periods(
    tax_benefit_system,
):
    """delete_arrays(variable) with no period must evict ALL
    _fast_cache entries for that variable."""
    sim = _make_simulation(tax_benefit_system)

    sim.calculate("income_tax", "2017-01")
    assert len(sim._fast_cache) > 0

    sim.delete_arrays("income_tax")

    matching = [k for k in sim._fast_cache if k[0] == "income_tax"]
    assert len(matching) == 0


def test_fast_cache_empty_after_clone(tax_benefit_system):
    """clone() must produce a simulation with an empty _fast_cache."""
    sim = _make_simulation(tax_benefit_system)

    sim.calculate("income_tax", "2017-01")
    assert len(sim._fast_cache) > 0

    cloned = sim.clone()
    assert len(cloned._fast_cache) == 0


def test_fast_cache_invalidated_after_purge_cache(tax_benefit_system):
    """purge_cache_of_invalid_values() must remove entries listed in
    invalidated_caches from _fast_cache."""
    sim = _make_simulation(tax_benefit_system)

    sim.calculate("income_tax", "2017-01")
    assert len(sim._fast_cache) > 0

    # Manually mark the entry as invalidated (simulating what the
    # framework does during dependency tracking)
    from policyengine_core.periods import period as make_period

    sim.invalidated_caches.add(("income_tax", make_period("2017-01")))
    # The stack must be empty for purge to fire
    sim.tracer._stack.clear()
    sim.purge_cache_of_invalid_values()

    matching = [k for k in sim._fast_cache if k[0] == "income_tax"]
    assert len(matching) == 0


def test_fast_cache_uses_period_not_str_as_key(tax_benefit_system):
    """_fast_cache keys should use Period objects, not str(period),
    to avoid unnecessary string conversions."""
    sim = _make_simulation(tax_benefit_system)
    sim.calculate("income_tax", "2017-01")

    # All keys should have Period as second element, not str
    for key in sim._fast_cache:
        variable_name, period_key = key
        assert not isinstance(period_key, str), (
            f"Expected Period as cache key, got str: {period_key!r}"
        )
        assert isinstance(period_key, tuple), (
            f"Period should be a tuple subclass, got {type(period_key)}"
        )
