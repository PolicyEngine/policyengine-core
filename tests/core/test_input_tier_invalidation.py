"""Reform replay selects surviving input tiers, rather than cached values."""

import numpy as np
import pytest

from tests.core.test_user_input_keys import (
    JANUARY,
    _key,
    _simulation,
    _store_on_disk,
)
from tests.core.test_user_input_keys_soundness import NoopReform


@pytest.mark.parametrize("registered", [False, True])
def test_reform_keeps_disk_input_instead_of_unrecorded_memory_cache(registered):
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    if registered:
        simulation.get_holder("rent")._disk_storage.put(np.array([500.0]), JANUARY)
        simulation._user_input_keys.add(_key("rent", JANUARY))
    else:
        simulation.set_input("rent", JANUARY, [500.0])
    simulation.memory_config.max_memory_occupation_pc = 101
    holder = simulation.get_holder("rent")
    holder.put_in_cache(np.array([700.0]), JANUARY, derived=False)

    simulation.apply_reform(NoopReform)

    assert holder._memory_storage.get(JANUARY) is None
    np.testing.assert_array_equal(holder._disk_storage.get(JANUARY), [500.0])
    assert simulation.to_input_dataframe()["rent__2025-01"].tolist() == [500.0]


@pytest.mark.parametrize("registered", [False, True])
def test_reform_drops_record_when_cache_replaced_the_only_input(registered):
    simulation = _simulation()
    holder = simulation.get_holder("rent")
    if registered:
        holder._memory_storage.put(np.array([500.0]), JANUARY)
        simulation._user_input_keys.add(_key("rent", JANUARY))
    else:
        simulation.set_input("rent", JANUARY, [500.0])
    holder.put_in_cache(np.array([700.0]), JANUARY, derived=False)

    simulation.apply_reform(NoopReform)

    assert holder.get_array(JANUARY) is None
    assert _key("rent", JANUARY) not in simulation._user_input_keys
    assert "rent__2025-01" not in simulation.to_input_dataframe().columns


try:
    from hypothesis import example, given, settings, strategies as st
except ImportError:

    @pytest.mark.skip(reason="Hypothesis is only installed with the dev extra")
    def test_reform_tier_selection_matches_fresh_inputs():
        pass

else:

    @settings(max_examples=60, deadline=None)
    @example(
        disk_input=True,
        memory_input=False,
        memory_cache=True,
        clone=False,
        registered=True,
        disk_value=500,
        memory_value=700,
    )
    @given(
        disk_input=st.booleans(),
        memory_input=st.booleans(),
        memory_cache=st.booleans(),
        clone=st.booleans(),
        registered=st.booleans(),
        disk_value=st.integers(0, 1_000),
        memory_value=st.integers(0, 1_000),
    )
    def test_reform_tier_selection_matches_fresh_inputs(
        disk_input,
        memory_input,
        memory_cache,
        clone,
        registered,
        disk_value,
        memory_value,
    ):
        simulation = _simulation()
        _store_on_disk(simulation, "rent")
        if disk_input:
            if registered:
                simulation.get_holder("rent")._disk_storage.put(
                    np.array([disk_value], dtype=np.float32), JANUARY
                )
                simulation._user_input_keys.add(_key("rent", JANUARY))
            else:
                simulation.set_input("rent", JANUARY, [disk_value])
        simulation.memory_config.max_memory_occupation_pc = 101
        if memory_input:
            simulation.set_input("rent", JANUARY, [memory_value])
        if memory_cache:
            simulation.get_holder("rent").put_in_cache(
                np.array([float(memory_value)]), JANUARY, derived=False
            )
        if clone:
            simulation = simulation.clone()

        expected = (
            memory_value
            if memory_input and not memory_cache
            else disk_value
            if disk_input
            else None
        )
        fresh = _simulation()
        if expected is not None:
            fresh.set_input("rent", JANUARY, [expected])

        for _ in range(2):
            simulation.apply_reform(NoopReform)
            assert simulation._user_input_keys == fresh._user_input_keys
            assert simulation.to_input_dataframe().equals(fresh.to_input_dataframe())
            np.testing.assert_array_equal(
                simulation.calculate("rent", JANUARY),
                fresh.calculate("rent", JANUARY),
            )
