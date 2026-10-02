"""A storage holds a set of shared keys only while it shares an array.

``InMemoryStorage._shared`` records which arrays a storage still shares with
the one it was cloned from (see ``InMemoryStorage.clone``). When it was added,
every storage was given an empty set of its own. A simulation has a storage
for every variable of the tax-benefit system, whether or not the variable is
ever read, so that was one 216-byte set per variable per simulation: 1.3 MB
for each simulation of a country with 6,000 variables, and the YAML test
runner keeps thousands of simulations alive.

A storage that shares nothing now refers to one empty object for all
storages, and has a set of its own only while at least one key is shared.
``test_storage_shared_index_differential.py`` checks random sequences of
operations against the storage that always had its own set.
"""

from __future__ import annotations

import numpy as np

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.data_storage.in_memory_storage import _NOTHING_SHARED
from tests.fixtures.branch_shared_arrays import build_simulation

JANUARY = periods.period("2017-01")


def _storages(simulation):
    return {
        name: holder._memory_storage
        for population in simulation.populations.values()
        for name, holder in population._holders.items()
    }


def _storage_with(*months):
    storage = InMemoryStorage(is_eternal=False)
    for index, month in enumerate(months):
        storage.put(np.array([float(index), 1.0]), month)
    return storage


def test_the_shared_nothing_index_is_empty_and_cannot_change():
    assert isinstance(_NOTHING_SHARED, frozenset)
    assert not _NOTHING_SHARED


def test_new_storages_have_no_set_of_their_own():
    first = InMemoryStorage(is_eternal=False)
    second = InMemoryStorage(is_eternal=True)

    assert first._shared is _NOTHING_SHARED
    assert second._shared is _NOTHING_SHARED

    # Storing, reading and deleting without sharing never needs one.
    first.put(np.array([1.0]), "2017-01")
    first.get("2017-01")
    first.delete("2017-01")
    first.delete()
    assert first._shared is _NOTHING_SHARED


def test_copied_storage_has_no_set_of_its_own():
    storage = _storage_with("2017-01", "2017-02")

    assert storage.clone()._shared is _NOTHING_SHARED
    assert storage.clone(share_arrays=True).clone()._shared is _NOTHING_SHARED


def test_sharing_clone_of_an_empty_storage_has_no_set_of_its_own():
    storage = InMemoryStorage(is_eternal=False)

    assert storage.clone(share_arrays=True)._shared is _NOTHING_SHARED


def test_sharing_clone_with_only_copied_arrays_has_no_set_of_its_own():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.ma.masked_array([1.0, 2.0], mask=[False, True]), "2017-01")

    clone = storage.clone(share_arrays=True)

    # A masked array is copied straight away, so nothing is shared.
    assert clone._shared is _NOTHING_SHARED
    assert not np.shares_memory(clone._arrays["default:2017-01"], storage.get(JANUARY))


def test_sharing_clones_each_have_their_own_set():
    storage = _storage_with("2017-01", "2017-02")

    first = storage.clone(share_arrays=True)
    second = storage.clone(share_arrays=True)

    assert first._shared == second._shared == {"default:2017-01", "default:2017-02"}
    assert first._shared is not second._shared
    assert storage._shared is _NOTHING_SHARED

    first.get("2017-01")
    assert first._shared == {"default:2017-02"}
    assert second._shared == {"default:2017-01", "default:2017-02"}


def test_set_is_released_once_every_shared_array_has_been_read():
    clone = _storage_with("2017-01", "2017-02").clone(share_arrays=True)

    clone.get("2017-01")
    assert clone._shared == {"default:2017-02"}
    clone.get("2017-02")

    assert clone._shared is _NOTHING_SHARED


def test_set_is_released_once_every_shared_array_has_been_replaced():
    clone = _storage_with("2017-01").clone(share_arrays=True)

    clone.put(np.array([5.0, 6.0]), "2017-01")

    assert clone._shared is _NOTHING_SHARED


def test_set_is_released_once_every_shared_array_has_been_deleted():
    by_period = _storage_with("2017-01", "2017-02").clone(share_arrays=True)
    by_period.delete("2017")
    assert by_period._shared is _NOTHING_SHARED

    by_branch = _storage_with("2017-01", "2017-02").clone(share_arrays=True)
    by_branch.delete()
    assert by_branch._shared is _NOTHING_SHARED


def test_deleting_another_branch_keeps_what_is_still_shared():
    storage = _storage_with("2017-01")
    storage.put(np.array([7.0, 8.0]), "2017-01", "other")
    clone = storage.clone(share_arrays=True)

    clone.delete(branch_name="other")

    assert clone._shared == {"default:2017-01"}


def test_storage_keeps_working_after_releasing_its_set():
    storage = _storage_with("2017-01")
    clone = storage.clone(share_arrays=True)
    clone.get("2017-01")
    assert clone._shared is _NOTHING_SHARED

    own = np.array([3.0, 4.0])
    clone.put(own, "2017-02")
    assert clone.get("2017-02") is own
    clone.delete("2017-02")
    assert clone.get("2017-02") is None

    # A clone of it shares again, with a set of its own.
    grandchild = clone.clone(share_arrays=True)
    assert grandchild._shared == {"default:2017-01"}
    assert clone._shared is _NOTHING_SHARED


def test_simulation_storages_have_no_sets_of_their_own(tax_benefit_system):
    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)

    storages = _storages(simulation)
    # One storage per variable, as in every simulation.
    assert len(storages) == len(tax_benefit_system.variables)
    assert all(storage._shared is _NOTHING_SHARED for storage in storages.values())

    copy = simulation.clone()
    assert all(
        storage._shared is _NOTHING_SHARED for storage in _storages(copy).values()
    )


def test_branch_storages_have_a_set_only_for_the_arrays_they_share(
    tax_benefit_system,
):
    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    with_arrays = {
        name for name, storage in _storages(simulation).items() if storage._arrays
    }
    assert with_arrays and with_arrays < set(_storages(simulation))

    branch = simulation.get_branch("branch")

    sets = []
    for name, storage in _storages(branch).items():
        if name in with_arrays:
            assert storage._shared == set(storage._arrays), name
            sets.append(storage._shared)
        else:
            assert storage._shared is _NOTHING_SHARED, name
    # No two storages hold the same set.
    assert len({id(shared) for shared in sets}) == len(sets)

    # Reading through the branch copies what it reads and releases the sets
    # of the storages that end up sharing nothing.
    branch.calculate("disposable_income", JANUARY)
    for name, storage in _storages(branch).items():
        if not storage._shared:
            assert storage._shared is _NOTHING_SHARED, name
    assert all(
        storage._shared is _NOTHING_SHARED for storage in _storages(simulation).values()
    )
