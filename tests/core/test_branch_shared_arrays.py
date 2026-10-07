"""Branch isolation through shared immutable cached-array entries."""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.data_storage.immutable_array_cache import CachedArrayEntry
from policyengine_core.enums import EnumArray
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import Simulation
from tests.fixtures.branch_shared_arrays import build_simulation

JANUARY = periods.period("2017-01")


def _entries(simulation) -> dict[tuple[str, str], CachedArrayEntry]:
    return {
        (name, key): entry
        for population in simulation.populations.values()
        for name, holder in population._holders.items()
        for key, entry in holder._memory_storage._arrays.items()
    }


def _snapshot(simulation) -> dict[tuple[str, str], np.ndarray]:
    return {key: entry.read().copy() for key, entry in _entries(simulation).items()}


def _assert_snapshot(simulation, snapshot) -> None:
    current = _entries(simulation)
    assert current.keys() == snapshot.keys()
    for key, expected in snapshot.items():
        np.testing.assert_array_equal(current[key].read(), expected)


def test_new_branch_shares_every_immutable_entry(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    simulation.calculate("housing_tax", "2017")
    parent = _entries(simulation)

    branch = simulation.get_branch("branch")
    child = _entries(branch)

    assert child.keys() == parent.keys()
    assert all(child[key] is parent[key] for key in parent)


def test_branch_storage_indexes_are_independent(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    branch = simulation.get_branch("branch")

    for population in simulation.populations.values():
        branch_population = branch.populations[population.entity.key]
        for name, holder in population._holders.items():
            branch_holder = branch_population._holders[name]
            assert branch_holder._memory_storage is not holder._memory_storage
            assert (
                branch_holder._memory_storage._arrays
                is not holder._memory_storage._arrays
            )


def test_parent_and_branch_reads_are_protected_views(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    parent_value = simulation.calculate("salary", JANUARY)
    branch = simulation.get_branch("branch")
    branch_value = branch.calculate("salary", JANUARY)

    assert np.shares_memory(parent_value, branch_value)
    assert parent_value is not branch_value
    assert not parent_value.flags.writeable
    assert not branch_value.flags.writeable
    with pytest.raises(ValueError):
        branch_value[0] = 1


def test_protected_view_cannot_be_made_writeable(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    value = simulation.calculate("salary", JANUARY)

    with pytest.raises(ValueError):
        value.flags.writeable = True


def test_set_input_on_branch_replaces_only_branch_entry(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    original = simulation.calculate("salary", JANUARY).copy()
    branch = simulation.get_branch("branch")

    branch.set_input("salary", JANUARY, np.array([10.0, 20.0, 30.0, 40.0]))

    np.testing.assert_array_equal(branch.calculate("salary", JANUARY), [10, 20, 30, 40])
    np.testing.assert_array_equal(simulation.calculate("salary", JANUARY), original)


def test_delete_on_branch_leaves_parent_entry(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    expected = simulation.calculate("income_tax", JANUARY).copy()
    branch = simulation.get_branch("branch")

    branch.delete_arrays("income_tax", JANUARY)

    assert branch.get_array("income_tax", JANUARY) is None
    np.testing.assert_array_equal(simulation.calculate("income_tax", JANUARY), expected)


def test_parent_replacement_after_branching_is_not_visible(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    expected = simulation.calculate("salary", JANUARY).copy()
    branch = simulation.get_branch("branch")

    simulation.set_input("salary", JANUARY, np.array([1.0, 2.0, 3.0, 4.0]))

    np.testing.assert_array_equal(branch.calculate("salary", JANUARY), expected)


def test_parent_deletion_after_branching_is_not_visible(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    expected = simulation.calculate("income_tax", JANUARY).copy()
    branch = simulation.get_branch("branch")
    simulation.delete_arrays("income_tax", JANUARY)

    np.testing.assert_array_equal(branch.calculate("income_tax", JANUARY), expected)


def test_nested_branch_shares_parent_snapshot(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    child = simulation.get_branch("child")
    child.set_input("rent", JANUARY, np.array([700.0, 70.0]))
    grandchild = child.get_branch("grandchild")
    entry = _entries(child)[("rent", "child:2017-01")]

    assert _entries(grandchild)[("rent", "child:2017-01")] is entry
    child.set_input("rent", JANUARY, np.array([1.0, 1.0]))
    np.testing.assert_array_equal(grandchild.calculate("rent", JANUARY), [700, 70])


def test_enum_entry_preserves_enum_metadata(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    parent = simulation.calculate("housing_occupancy_status", JANUARY)
    branch = simulation.get_branch("branch")
    value = branch.calculate("housing_occupancy_status", JANUARY)

    assert isinstance(value, EnumArray)
    assert value.possible_values is parent.possible_values
    assert list(value.decode_to_str()) == list(parent.decode_to_str())
    assert not value.flags.writeable


def test_masked_entry_protects_data_and_mask(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    masked = np.ma.array([10.0, 20.0, 30.0, 40.0], mask=[False] * 4)
    simulation.set_input("salary", "2017-03", masked)
    branch = simulation.get_branch("branch")
    value = branch.calculate("salary", "2017-03")

    with pytest.raises((ValueError, TypeError)):
        value[0] = np.ma.masked
    with pytest.raises((ValueError, TypeError)):
        value.mask[0] = True
    assert not np.ma.getmaskarray(simulation.calculate("salary", "2017-03")).any()


def test_string_entry_is_shared_and_read_only(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    simulation.set_input("postal_code", JANUARY, np.array(["75001", "69002"]))
    parent = simulation.calculate("postal_code", JANUARY)
    branch = simulation.get_branch("branch")
    value = branch.calculate("postal_code", JANUARY)

    np.testing.assert_array_equal(value, parent)
    assert np.shares_memory(value, parent)
    assert not value.flags.writeable


def test_plain_clone_shares_safe_entries(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    parent = _entries(simulation)

    clone = simulation.clone()

    assert all(_entries(clone)[key] is entry for key, entry in parent.items())
    clone.set_input("salary", JANUARY, np.zeros(4))
    assert (
        _entries(clone)[("salary", "default:2017-01")]
        is not parent[("salary", "default:2017-01")]
    )


def test_traced_branch_uses_immutable_entries(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    simulation.trace = True
    simulation.calculate("salary", JANUARY)
    branch = simulation.get_branch("branch")

    assert branch.trace and branch.tracer is simulation.tracer
    assert (
        _entries(branch)[("salary", "default:2017-01")]
        is _entries(simulation)[("salary", "default:2017-01")]
    )


def test_clearing_calculated_results_preserves_branch_inputs(
    tax_benefit_system,
) -> None:
    simulation = build_simulation(tax_benefit_system)
    branch = simulation.get_branch("branch")
    branch.set_input("rent", JANUARY, np.array([900.0, 90.0]))
    branch.calculate("disposable_income", JANUARY)

    branch.clear_calculated_results()

    np.testing.assert_array_equal(branch.calculate("rent", JANUARY), [900, 90])
    assert branch.get_array("disposable_income", JANUARY) is None


def test_branch_reads_protected_disk_value(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = simulation.get_holder("rent")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    simulation.set_input("rent", "2017-02", np.array([11.0, 22.0]))
    branch = simulation.get_branch("branch")

    value = branch.calculate("rent", "2017-02")

    np.testing.assert_array_equal(value, [11, 22])
    assert not value.flags.writeable


def test_yearly_branch_input_creates_protected_months(tax_benefit_system) -> None:
    simulation = build_simulation(tax_benefit_system)
    branch = simulation.get_branch("branch")
    branch.set_input("salary", "2018", np.full(4, 12_000.0))

    for month in periods.period("2018").get_subperiods(periods.MONTH):
        value = branch.calculate("salary", month)
        np.testing.assert_array_equal(value, np.full(4, 1_000.0))
        assert not value.flags.writeable


def test_legacy_storage_clear_on_dropped_branch_leaves_parent(
    tax_benefit_system,
) -> None:
    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    snapshot = _snapshot(simulation)
    branch = simulation.get_branch("short_lived")

    for population in branch.populations.values():
        for holder in population._holders.values():
            holder._memory_storage._arrays.clear()

    _assert_snapshot(simulation, snapshot)


def test_country_clone_override_still_gets_isolated_indexes(
    tax_benefit_system,
) -> None:
    class CountrySimulation(Simulation):
        def clone(self, debug=False, trace=False, clone_tax_benefit_system=True):
            clone = super().clone(debug, trace, clone_tax_benefit_system)
            clone.country_clone = True
            return clone

    simulation = build_simulation(tax_benefit_system)
    simulation.__class__ = CountrySimulation
    branch = simulation.get_branch("branch")

    assert branch.country_clone
    assert branch.get_holder("salary")._memory_storage._arrays is not (
        simulation.get_holder("salary")._memory_storage._arrays
    )
