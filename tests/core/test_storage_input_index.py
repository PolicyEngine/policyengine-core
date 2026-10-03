"""A storage records which of its values are inputs, at no cost when none is.

``InMemoryStorage.is_derived`` tells a value the simulation calculated
(``put(..., derived=True)``) from an input, so that auto-carry-over and
uprating start only from inputs. The storage records the inputs, in
``_inputs``, because a simulation calculates far more values than it is
given. Like ``_shared`` (``test_storage_shared_index.py``), ``_inputs`` is
one empty object for every storage until a storage holds an input: a
simulation has a storage for every variable of the tax-benefit system, and
an empty set in each is 216 bytes per variable per simulation (#577). A
storage and its clones share one frozenset of inputs until one of them
changes its own. ``test_storage_shared_index_differential.py`` checks random
sequences of operations against a storage that always had its own sets.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.data_storage.in_memory_storage import (
    _INPUTS_RECORDED,
    _NO_INPUTS,
    _NOTHING_SHARED,
)
from policyengine_core.reforms import Reform
from tests.fixtures.branch_shared_arrays import build_simulation

JANUARY = periods.period("2017-01")


def _storages(simulation):
    return {
        name: holder._memory_storage
        for population in simulation.populations.values()
        for name, holder in population._holders.items()
    }


def test_the_no_inputs_index_is_empty_and_cannot_change():
    assert isinstance(_NO_INPUTS, frozenset)
    assert not _NO_INPUTS


def test_storing_reading_and_deleting_derived_values_never_needs_a_set():
    storage = InMemoryStorage(is_eternal=False)
    assert storage._inputs is _NO_INPUTS
    storage.put(np.array([1.0]), "2017-01", derived=True)
    assert storage.is_derived("2017-01")
    storage.get("2017-01")
    storage.delete("2017-01")
    storage.put(np.array([1.0]), "2017-02", derived=True)
    storage.delete()
    assert storage._inputs is _NO_INPUTS
    assert "_inputs" not in vars(storage)


@pytest.mark.parametrize(
    "drop",
    [
        lambda storage: storage.put(np.array([2.0]), "2017-01", derived=True),
        lambda storage: storage.delete("2017-01"),
        lambda storage: storage.delete("2017"),
        lambda storage: storage.delete(),
    ],
    ids=[
        "replaced_by_a_derived_value",
        "deleted",
        "deleted_by_a_containing_period",
        "wiped",
    ],
)
def test_an_input_gets_a_set_that_is_released_with_it(drop):
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), "2017-01")
    assert storage._inputs == {"default:2017-01"}
    assert "_inputs" in vars(storage)
    assert not storage.is_derived("2017-01")

    drop(storage)
    # Released, not replaced by another empty object.
    assert "_inputs" not in vars(storage)
    assert storage._inputs is _NO_INPUTS


def test_the_set_is_kept_while_another_value_is_an_input():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), "2017-01")
    storage.put(np.array([1.0]), "2017-02")
    storage.delete("2017-01")
    assert storage._inputs == {"default:2017-02"}
    storage.put(np.array([3.0]), "2017-02", derived=True)
    assert storage._inputs is _NO_INPUTS
    assert storage.is_derived("2017-02")


def test_only_stored_values_are_derived():
    storage = InMemoryStorage(is_eternal=False)
    assert not storage.is_derived("2017-01")
    storage.put(np.array([1.0]), "2017-01", "reform", derived=True)
    assert not storage.is_derived("2017-01")
    assert storage.is_derived("2017-01", "reform")


@pytest.mark.parametrize("share_arrays", [False, True], ids=["copy", "share"])
def test_clones_share_the_inputs_until_one_of_them_changes_its_own(share_arrays):
    derived_only = InMemoryStorage(is_eternal=False)
    derived_only.put(np.array([1.0]), "2017-01", derived=True)
    assert derived_only.clone(share_arrays)._inputs is _NO_INPUTS

    source = InMemoryStorage(is_eternal=False)
    source.put(np.array([1.0]), "2017-01")
    source.put(np.array([1.0]), "2017-02")
    first = source.clone(share_arrays)
    second = source.clone(share_arrays)
    grandchild = first.clone(share_arrays)
    # One immutable set for all four: a clone costs no set of its own.
    inputs = source._inputs
    assert isinstance(inputs, frozenset)
    assert inputs == {"default:2017-01", "default:2017-02"}
    assert first._inputs is inputs and second._inputs is inputs
    assert grandchild._inputs is inputs

    # Storing an input over an input changes nothing.
    first.put(np.array([5.0]), "2017-01")
    assert first._inputs is inputs

    # The first storage to change its inputs gets a set of its own.
    first.put(np.array([2.0]), "2017-01", derived=True)
    assert first.is_derived("2017-01") and not first.is_derived("2017-02")
    assert isinstance(first._inputs, set)
    for other in (source, second, grandchild):
        assert other._inputs is inputs and not other.is_derived("2017-01")

    source.put(np.array([3.0]), "2017-03")
    assert source._inputs == inputs | {"default:2017-03"}
    second.put(np.array([3.0]), "2017-03", derived=True)
    assert second.is_derived("2017-03") and not source.is_derived("2017-03")
    second.delete("2017-02")
    assert second._inputs == {"default:2017-01"}
    assert grandchild._inputs is inputs
    grandchild.delete()
    assert grandchild._inputs is _NO_INPUTS
    assert not source.is_derived("2017-01") and not source.is_derived("2017-02")


def test_clone_of_an_emptied_storage_gets_only_the_inputs_it_holds():
    source = InMemoryStorage(is_eternal=False)
    source.put(np.array([1.0]), "2017-01")
    source._arrays.clear()  # as policyengine-uk clears a dropped branch
    assert source.clone()._inputs is _NO_INPUTS
    source.put(np.array([1.0]), "2017-02")
    clone = source.clone()
    assert clone._inputs == {"default:2017-02"}
    assert isinstance(clone._inputs, frozenset)


@pytest.mark.parametrize(
    "duplicate",
    [copy.deepcopy, lambda storage: pickle.loads(pickle.dumps(storage))],
    ids=["deepcopy", "pickle"],
)
def test_copied_or_unpickled_storage_keeps_the_marks(duplicate):
    derived = InMemoryStorage(is_eternal=False)
    derived.put(np.array([1.0]), "2017-01", derived=True)
    twin = duplicate(derived)
    assert twin._inputs is _NO_INPUTS
    assert twin._shared is _NOTHING_SHARED
    assert twin.is_derived("2017-01")

    for cloned_first in (False, True):
        inputs = InMemoryStorage(is_eternal=False)
        inputs.put(np.array([1.0]), "2017-01")
        if cloned_first:
            inputs.clone()  # its inputs are now a frozenset it shares
        twin = duplicate(inputs)
        assert not twin.is_derived("2017-01")
        # Changing either one's inputs leaves the other's alone.
        twin.put(np.array([2.0]), "2017-01", derived=True)
        assert twin.is_derived("2017-01")
        assert not inputs.is_derived("2017-01")
        inputs.put(np.array([2.0]), "2017-02")
        assert not twin.has("2017-02") and not twin.is_derived("2017-02")


def _older_state(storage, **extra):
    """The state a version that recorded no inputs pickled."""
    state = {"_arrays": dict(storage._arrays), "is_eternal": storage.is_eternal}
    state.update(extra)
    return state


def test_values_from_an_older_pickle_count_as_inputs():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), "2017-01", derived=True)
    storage.put(np.array([1.0]), "2017-02", derived=True)
    restored = InMemoryStorage.__new__(InMemoryStorage)
    # 3.32.12 pickled an empty set of shared keys in every storage.
    restored.__setstate__(_older_state(storage, _shared=set()))
    assert not restored.is_derived("2017-01")
    assert not restored.is_derived("2017-02")
    assert restored._inputs == {"default:2017-01", "default:2017-02"}
    assert _INPUTS_RECORDED not in vars(restored)

    # A development version of this change recorded derived values instead.
    restored = InMemoryStorage.__new__(InMemoryStorage)
    restored.__setstate__(_older_state(storage, _derived={"default:2017-02"}))
    assert not restored.is_derived("2017-01")
    assert restored.is_derived("2017-02")
    assert "_derived" not in vars(restored)
    # A state with no set of shared keys gets none.
    assert "_shared" not in vars(restored)

    # An older storage with nothing stored needs no set.
    restored = InMemoryStorage.__new__(InMemoryStorage)
    restored.__setstate__(_older_state(InMemoryStorage(is_eternal=False)))
    assert restored._inputs is _NO_INPUTS
    assert restored._shared is _NOTHING_SHARED
    restored.put(np.array([2.0]), "2017-03", derived=True)
    assert restored.is_derived("2017-03")


def test_a_pickle_records_that_the_inputs_were_recorded():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), "2017-01", derived=True)
    state = storage.__getstate__()
    assert state[_INPUTS_RECORDED] is True
    assert _INPUTS_RECORDED not in vars(storage)
    assert pickle.loads(pickle.dumps(storage)).is_derived("2017-01")


def test_only_storages_holding_an_input_have_a_set(tax_benefit_system):
    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)

    storages = _storages(simulation)
    assert len(storages) == len(tax_benefit_system.variables)
    with_inputs = {name for name, storage in storages.items() if storage._inputs}
    with_derived = {
        name
        for name, storage in storages.items()
        if any(
            storage.is_derived(key.split(":", 1)[1], key.split(":", 1)[0])
            for key in storage._arrays
        )
    }
    # The situation's inputs, and the values calculated from them.
    assert with_inputs and with_derived
    assert with_inputs | with_derived < set(storages)
    for name, storage in storages.items():
        assert ("_inputs" in vars(storage)) == (name in with_inputs), name
        if name not in with_inputs:
            assert storage._inputs is _NO_INPUTS, name
            assert storage._shared is _NOTHING_SHARED, name


def test_a_branch_shares_its_parents_inputs(tax_benefit_system):
    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    branch = simulation.get_branch("branch")
    parents = _storages(simulation)
    assert any(storage._inputs for storage in parents.values())
    for name, storage in _storages(branch).items():
        assert storage._inputs is parents[name]._inputs, name


def test_apply_reform_keeps_only_the_replayed_inputs(tax_benefit_system):
    class _noop(Reform):
        def apply(self):
            pass

    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    before = {
        name: set(storage._inputs) for name, storage in _storages(simulation).items()
    }
    simulation.apply_reform(_noop)
    for name, storage in _storages(simulation).items():
        assert set(storage._inputs) == before.get(name, set()), name
        assert set(storage._arrays) == set(storage._inputs), name
