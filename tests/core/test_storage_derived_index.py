"""A storage holds a set of derived keys only while it holds a derived value.

``InMemoryStorage._derived`` records which stored values the simulation
calculated rather than took as input (``put(..., derived=True)``), so that
auto-carry-over and uprating start only from inputs. Like ``_shared``
(``test_storage_shared_index.py``), it is one empty object for every storage
until a storage holds a derived value: a simulation has a storage for every
variable of the tax-benefit system, and an empty set in each is 216 bytes
per variable per simulation. ``test_storage_shared_index_differential.py``
checks random sequences of operations against the storage that always had
both sets.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.data_storage.in_memory_storage import (
    _NOTHING_DERIVED,
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


def test_the_nothing_derived_index_is_empty_and_cannot_change():
    assert isinstance(_NOTHING_DERIVED, frozenset)
    assert not _NOTHING_DERIVED


def test_storing_reading_and_deleting_inputs_never_needs_a_set():
    storage = InMemoryStorage(is_eternal=False)
    assert storage._derived is _NOTHING_DERIVED
    storage.put(np.array([1.0]), "2017-01")
    storage.get("2017-01")
    storage.delete("2017-01")
    storage.put(np.array([1.0]), "2017-02")
    storage.delete()
    assert storage._derived is _NOTHING_DERIVED
    assert "_derived" not in vars(storage)


@pytest.mark.parametrize(
    "drop",
    [
        lambda storage: storage.put(np.array([2.0]), "2017-01"),
        lambda storage: storage.delete("2017-01"),
        lambda storage: storage.delete("2017"),
        lambda storage: storage.delete(),
    ],
    ids=["replaced_by_an_input", "deleted", "deleted_by_a_containing_period", "wiped"],
)
def test_a_derived_value_gets_a_set_that_is_released_with_it(drop):
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), "2017-01", derived=True)
    assert storage._derived == {"default:2017-01"}
    assert "_derived" in vars(storage)
    assert storage.is_derived("2017-01")

    drop(storage)
    assert not storage.is_derived("2017-01")
    # Released, not replaced by another empty object.
    assert "_derived" not in vars(storage)
    assert storage._derived is _NOTHING_DERIVED


def test_the_set_is_kept_while_another_value_is_derived():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), "2017-01", derived=True)
    storage.put(np.array([1.0]), "2017-02", derived=True)
    storage.delete("2017-01")
    assert storage._derived == {"default:2017-02"}
    storage.put(np.array([3.0]), "2017-02")
    assert storage._derived is _NOTHING_DERIVED


@pytest.mark.parametrize("share_arrays", [False, True], ids=["copy", "share"])
def test_clones_copy_the_marks_into_a_set_of_their_own(share_arrays):
    inputs = InMemoryStorage(is_eternal=False)
    inputs.put(np.array([1.0]), "2017-01")
    assert inputs.clone(share_arrays)._derived is _NOTHING_DERIVED

    derived = InMemoryStorage(is_eternal=False)
    derived.put(np.array([1.0]), "2017-01", derived=True)
    clone = derived.clone(share_arrays)
    assert clone._derived == derived._derived == {"default:2017-01"}
    assert clone._derived is not derived._derived
    clone.put(np.array([2.0]), "2017-01")
    assert clone._derived is _NOTHING_DERIVED
    assert derived.is_derived("2017-01")


@pytest.mark.parametrize(
    "duplicate",
    [copy.deepcopy, lambda storage: pickle.loads(pickle.dumps(storage))],
    ids=["deepcopy", "pickle"],
)
def test_copied_or_unpickled_storage_keeps_the_marks(duplicate):
    inputs = InMemoryStorage(is_eternal=False)
    inputs.put(np.array([1.0]), "2017-01")
    twin = duplicate(inputs)
    assert twin._derived is _NOTHING_DERIVED
    assert twin._shared is _NOTHING_SHARED
    assert not twin.is_derived("2017-01")

    derived = InMemoryStorage(is_eternal=False)
    derived.put(np.array([1.0]), "2017-01", derived=True)
    twin = duplicate(derived)
    assert twin.is_derived("2017-01")
    assert twin._derived is not derived._derived


def test_storage_pickled_with_an_empty_set_still_works():
    # Before storages shared one empty object, each pickled its own sets.
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), "2017-01")
    storage.__dict__["_derived"] = set()
    storage.__dict__["_shared"] = set()
    restored = pickle.loads(pickle.dumps(storage))
    assert not restored.is_derived("2017-01")
    restored.put(np.array([2.0]), "2017-02", derived=True)
    assert restored.is_derived("2017-02")
    restored.delete()
    assert restored.get("2017-01") is None


def test_only_storages_holding_a_derived_value_have_a_set(tax_benefit_system):
    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)

    storages = _storages(simulation)
    assert len(storages) == len(tax_benefit_system.variables)
    with_derived = {name for name, storage in storages.items() if storage._derived}
    # Calculating disposable income derives some values and not others.
    assert with_derived and with_derived < set(storages)
    for name, storage in storages.items():
        assert ("_derived" in vars(storage)) == (name in with_derived), name
        if name not in with_derived:
            assert storage._derived is _NOTHING_DERIVED, name
            assert storage._shared is _NOTHING_SHARED, name


def test_apply_reform_releases_the_marks(tax_benefit_system):
    class _noop(Reform):
        def apply(self):
            pass

    simulation = build_simulation(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    assert any(storage._derived for storage in _storages(simulation).values())
    simulation.apply_reform(_noop)
    for name, storage in _storages(simulation).items():
        assert storage._derived is _NOTHING_DERIVED, name
        assert "_derived" not in vars(storage), name
