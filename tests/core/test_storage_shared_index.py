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

import copy
import pickle

import numpy as np
import pytest

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


def test_only_the_storage_itself_refers_to_its_set():
    import gc

    storage = _storage_with("2017-01", "2017-02")
    clones = [storage.clone(share_arrays=True) for _ in range(3)]

    for clone in clones:
        referrers = [
            referrer
            for referrer in gc.get_referrers(clone._shared)
            # Python 3.13+ may hold attributes inline in the object itself.
            if referrer is not clone and referrer is not clone.__dict__
        ]
        # Nothing else, the source included, keeps the set alive after the
        # clone releases it.
        assert referrers == [], referrers


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


def test_storage_sharing_nothing_has_no_attribute_of_its_own():
    storage = _storage_with("2017-01")
    assert "_shared" not in vars(storage)

    clone = storage.clone(share_arrays=True)
    assert "_shared" in vars(clone)
    clone.get("2017-01")

    # Released, not replaced by another empty object.
    assert "_shared" not in vars(clone)


@pytest.mark.parametrize(
    "duplicate",
    [copy.deepcopy, lambda storage: pickle.loads(pickle.dumps(storage))],
    ids=["deepcopy", "pickle"],
)
def test_copied_or_unpickled_storage_keeps_the_index_rules(duplicate):
    storage = _storage_with("2017-01", "2017-02")
    released = storage.clone(share_arrays=True)
    released.get("2017-01")
    released.get("2017-02")
    sharing = storage.clone(share_arrays=True)

    for original in (storage, released):
        twin = duplicate(original)
        assert twin._shared is _NOTHING_SHARED
        assert np.array_equal(twin.get("2017-01"), original.get("2017-01"))

    twin = duplicate(sharing)
    assert twin._shared == sharing._shared == {"default:2017-01", "default:2017-02"}
    assert twin._shared is not sharing._shared
    assert np.array_equal(twin.get("2017-01"), [0.0, 1.0])
    assert twin._shared == {"default:2017-02"}
    assert sharing._shared == {"default:2017-01", "default:2017-02"}
    twin.get("2017-02")
    assert twin._shared is _NOTHING_SHARED


def test_storage_from_a_pickle_with_an_empty_set_still_works():
    # 3.32.12 pickled every storage with an empty set of its own.
    storage = _storage_with("2017-01")
    storage.__dict__["_shared"] = set()
    restored = pickle.loads(pickle.dumps(storage))
    assert restored._shared == set()

    restored.get("2017-01")
    restored.put(np.array([2.0, 3.0]), "2017-02")
    restored.delete("2017-02")
    restored.delete()
    assert restored.get("2017-01") is None


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


# ----- Two threads making first reads at once ----- #
#
# ``InMemoryStorage.clone`` documents that a storage is not safe to read from
# several threads at once; on 3.32.12 overlapping first reads never raised.
# Releasing the set must not make them raise. The first tests run one read
# inside another at the point where a thread switch would do it, so they do
# not depend on timing.


class _ReadInsideCheck(set):
    """A set of shared keys that runs ``interleave`` inside a membership check."""

    def __init__(self, keys, interleave, on_call):
        super().__init__(keys)
        self.interleave = interleave
        self.on_call = on_call
        self.calls = 0

    def __contains__(self, key):
        found = set.__contains__(self, key)
        self.calls += 1
        if self.calls == self.on_call:
            self.interleave()
        return found


@pytest.mark.parametrize("on_call", [1, 2])
def test_another_thread_releasing_the_set_mid_read_is_harmless(on_call):
    storage = _storage_with("2017-01")
    clone = storage.clone(share_arrays=True)
    # The other thread reads the same last shared key to the end, during this
    # read's membership check in ``get`` (call 1) or in ``_stop_sharing``
    # (call 2).
    clone._shared = _ReadInsideCheck(
        clone._shared, lambda: clone.get("2017-01"), on_call
    )

    values = clone.get("2017-01")

    assert np.array_equal(values, [0.0, 1.0])
    assert values.flags.writeable
    assert clone._shared is _NOTHING_SHARED
    assert np.array_equal(storage.get("2017-01"), [0.0, 1.0])


class _ReadAfterTruthTest(set):
    """A set of shared keys that runs ``interleave`` after its first truth test."""

    def __init__(self, keys, interleave):
        super().__init__(keys)
        self.interleave = interleave
        self.tested = False

    def __bool__(self):
        found = set.__len__(self) > 0
        if not self.tested:
            self.tested = True
            self.interleave()
        return found


def test_another_thread_releasing_the_set_mid_delete_is_harmless():
    storage = _storage_with("2017-01", "2017-02")
    clone = storage.clone(share_arrays=True)
    clone.get("2017-01")
    assert clone._shared == {"default:2017-02"}
    # This thread deletes January, which it owns. Between its finding the set
    # non-empty and its dropping deleted keys from it, the other thread reads
    # February, the last shared key, and releases the set.
    clone._shared = _ReadAfterTruthTest(clone._shared, lambda: clone.get("2017-02"))

    clone.delete("2017-01")

    assert clone._shared is _NOTHING_SHARED
    assert clone.get("2017-01") is None
    assert np.array_equal(clone.get("2017-02"), [1.0, 1.0])
    assert np.array_equal(storage.get("2017-02"), [1.0, 1.0])


def test_concurrent_first_reads_never_raise():
    import sys
    import threading

    source = _storage_with("2017-01", "2017-02")
    errors = []
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(2_000):
            clone = source.clone(share_arrays=True)
            barrier = threading.Barrier(2)

            def read(month, clone=clone, barrier=barrier):
                barrier.wait()
                try:
                    clone.get(month)
                except Exception as error:  # noqa: BLE001
                    errors.append(error)

            threads = [
                threading.Thread(target=read, args=(month,))
                for month in ("2017-01", "2017-02")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert clone._shared is _NOTHING_SHARED
    finally:
        sys.setswitchinterval(interval)

    assert errors == []
