"""Restored input records follow deletion even for early-year storage keys."""

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from tests.core.test_user_input_keys import _simulation, _store_on_disk
from tests.core.test_user_input_keys_soundness import (
    NoopReform,
    raw_day,
    raw_month,
    raw_year,
)


def _raw_simulation(variable_type, on_disk):
    system = CountryTaxBenefitSystem()
    system.add_variable(variable_type)
    simulation = _simulation(system)
    if on_disk:
        _store_on_disk(simulation, variable_type.__name__)
    return simulation


@pytest.mark.parametrize("year", [1, 99, 999, 1000])
@pytest.mark.parametrize("variable_type", [raw_year, raw_month, raw_day])
@pytest.mark.parametrize("on_disk", [False, True])
def test_deleted_registered_early_year_input_is_forgotten(year, variable_type, on_disk):
    """#576 restore can register an input without Holder tier metadata."""
    simulation = _raw_simulation(variable_type, on_disk)
    holder = simulation.get_holder(variable_type.__name__)
    period = periods.Period(
        (variable_type.definition_period, periods.instant((year, 1, 1)), 1)
    )
    storage = holder._disk_storage if on_disk else holder._memory_storage
    storage.put(np.array([42.0]), period)
    key = (variable_type.__name__, "default", period)
    simulation._user_input_keys.add(key)
    assert not hasattr(holder, "_user_input_storage")

    holder.delete_arrays()

    assert not storage.has(period)
    assert key not in simulation._user_input_keys
    holder.put_in_cache(np.array([99.0]), period, derived=True)
    simulation.apply_reform(NoopReform)
    assert holder.get_array(period) is None


@pytest.mark.parametrize("year", [1, 999, 1000])
@pytest.mark.parametrize(
    "variable_type, unit, month, day, size",
    [
        (raw_year, periods.YEAR, 1, 1, 2),
        (raw_year, periods.YEAR, 3, 1, 1),
        (raw_month, periods.MONTH, 3, 1, 2),
        (raw_month, periods.MONTH, 3, 1, 12),
        (raw_day, periods.DAY, 3, 17, 2),
    ],
)
@pytest.mark.parametrize("on_disk", [False, True])
def test_registered_deletion_matches_set_input_for_period_aliases(
    year, variable_type, unit, month, day, size, on_disk
):
    """Register-only and set_input paths forget the same normalized slot.

    Deletion removes the input from the record regardless of the stored
    period's unit, size, early year, alias, tier, or underscored branch.
    A write afterward remains a cache value and is cleared by a reform.
    """
    restored = _raw_simulation(variable_type, on_disk)
    tracked = _raw_simulation(variable_type, on_disk)
    period = periods.Period((unit, periods.instant((year, month, day)), size))
    branch = "restore_branch"
    holder = restored.get_holder(variable_type.__name__)
    storage = holder._disk_storage if on_disk else holder._memory_storage
    storage.put(np.array([42.0]), period, branch)
    stored_period = holder._storage_period(period)
    key = (variable_type.__name__, branch, stored_period)
    restored._user_input_keys.add(key)
    tracked.get_holder(variable_type.__name__).set_input(period, [42.0], branch)
    assert restored._user_input_keys == tracked._user_input_keys

    for simulation in (restored, tracked):
        current = simulation.get_holder(variable_type.__name__)
        current.delete_arrays(None, branch)
        current.put_in_cache(np.array([99.0]), stored_period, branch, derived=True)
        assert key not in simulation._user_input_keys
        simulation.apply_reform(NoopReform)
        assert current._get_array_from_storage(stored_period, branch) is None

    assert restored._user_input_keys == tracked._user_input_keys
