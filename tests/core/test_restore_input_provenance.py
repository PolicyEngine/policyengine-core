"""Dump restoration promotes non-derived caches without changing live provenance."""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.tools.simulation_dumper import (
    dump_simulation,
    restore_simulation,
)
from tests.fixtures.uprated_inputs import NoOp, build_simulation, build_system


@pytest.mark.parametrize("cached_value", [False, True], ids=["same", "changed"])
def test_restored_cache_is_an_input_while_live_reform_drops_it(tmp_path, cached_value):
    """Identical keys and storage marks do not imply identical input provenance."""
    period = periods.period("2012")
    key = ("eligible", "default", period)
    simulation = build_simulation(
        build_system(), [("eligible", period, [False, False])]
    )
    holder = simulation.get_holder("eligible")
    cached = np.array([cached_value, cached_value], dtype=bool)
    holder.put_in_cache(cached, period, derived=False)

    directory = str(tmp_path / "dump")
    dump_simulation(simulation, directory)
    restored = restore_simulation(directory, simulation.tax_benefit_system)
    restored_holder = restored.get_holder("eligible")

    for stored_holder in (holder, restored_holder):
        assert not stored_holder.is_derived(period)
        np.testing.assert_array_equal(stored_holder.get_array(period), cached)

    for _ in range(2):
        simulation.apply_reform(NoOp)
        restored.apply_reform(NoOp)

        assert simulation._user_input_keys == set()
        assert holder.get_array(period) is None
        assert restored._user_input_keys == {key}
        np.testing.assert_array_equal(restored_holder.get_array(period), cached)
        assert not restored_holder.is_derived(period)

        assert "eligible__2012" not in simulation.to_input_dataframe().columns
        assert (
            restored.to_input_dataframe()["eligible__2012"].tolist() == cached.tolist()
        )
