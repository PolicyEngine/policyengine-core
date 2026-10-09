"""Option caches remain derived alongside supplied-input tier records."""

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.experimental import MemoryConfig
from tests.fixtures.option_caches import (
    FLOW,
    MONTH,
    PROBE,
    YEAR,
    build,
    make_probe,
    run,
)


def _store_on_disk(simulation):
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = simulation.get_holder(PROBE)
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True


@pytest.mark.parametrize(
    "definition_period, native, target, option, expected",
    [
        (MONTH, "2012-01", "2012", "add", [12.0]),
        (YEAR, "2012", "2012-01", "divide", [1.0]),
    ],
)
@pytest.mark.parametrize("on_disk", [False, True])
@pytest.mark.parametrize("deleted_target", [False, True])
def test_option_cache_is_never_registered_exported_or_replayed_as_input(
    definition_period, native, target, option, expected, on_disk, deleted_target
):
    simulation = build(make_probe(definition_period, FLOW, set_input=None))
    if on_disk:
        _store_on_disk(simulation)
    holder = simulation.get_holder(PROBE)
    target_period = periods.period(target)
    native_period = periods.period(native)
    target_key = (PROBE, "default", target_period)
    if deleted_target:
        simulation.set_input(PROBE, target, [7])
        simulation.delete_arrays(PROBE, target)
        assert target_key not in simulation._user_input_keys
    simulation.set_input(PROBE, native, [12])

    assert run(simulation, option, target) == expected

    storage = holder._disk_storage if on_disk else holder._memory_storage
    assert storage.get(target_period).tolist() == expected
    assert storage.is_derived(target_period)
    assert not holder._stores_user_input(target_period, "default")
    assert target_key not in simulation._user_input_keys
    assert (PROBE, "default", native_period) in simulation._user_input_keys
    frame = simulation.to_input_dataframe()
    assert {
        column: frame[column].tolist()
        for column in frame.columns
        if column.startswith(f"{PROBE}__")
    } == {f"{PROBE}__{native}": [12.0]}
    assert simulation.to_input_dict()[PROBE] == {native: [12.0]}

    simulation._invalidate_all_caches()

    assert storage.get(target_period) is None
    assert storage.get(native_period).tolist() == [12.0]
    assert target_key not in simulation._user_input_keys
    assert holder._stores_user_input(native_period, "default")
    assert simulation.to_input_dataframe()[f"{PROBE}__{native}"].tolist() == [12.0]
    assert run(simulation, "calculate", target) == expected
    assert storage.is_derived(target_period)


@pytest.mark.parametrize(
    "definition_period, native, target, option, expected",
    [
        (MONTH, "2012-01", "2012", "add", [12.0]),
        (YEAR, "2012", "2012-01", "divide", [1.0]),
    ],
)
def test_option_refresh_preserves_a_supplied_disk_input_hidden_by_memory_cache(
    definition_period, native, target, option, expected
):
    simulation = build(make_probe(definition_period, FLOW, set_input=None))
    _store_on_disk(simulation)
    simulation.set_input(PROBE, native, [12])
    simulation.set_input(PROBE, target, [7])
    holder = simulation.get_holder(PROBE)
    target_period = periods.period(target)
    input_path = holder._disk_storage._files[f"default_{target}"]
    # Model an existing derived value in the higher-priority memory tier.
    # The supplied disk value remains independently recorded for replay.
    holder._memory_storage.put(
        np.array([0.0], dtype=holder.variable.dtype), target_period, derived=True
    )

    assert run(simulation, option, target) == expected
    assert run(simulation, "calculate", target) == expected
    assert holder._memory_storage.get(target_period).tolist() == expected
    assert holder._memory_storage.is_derived(target_period)
    assert holder._disk_storage.get(target_period).tolist() == [7.0]
    assert holder._user_input_storage[("default", target)] == (
        target_period,
        frozenset({"disk"}),
    )
    assert holder._stores_user_input(target_period, "default")
    assert (PROBE, "default", target_period) in simulation._user_input_keys

    simulation._invalidate_all_caches()

    assert holder._memory_storage.get(target_period) is None
    assert holder._disk_storage._files[f"default_{target}"] == input_path
    assert not holder._disk_storage.is_derived(target_period)
    assert run(simulation, "calculate", target) == [7.0]
    assert holder._stores_user_input(target_period, "default")
    assert (PROBE, "default", target_period) in simulation._user_input_keys
