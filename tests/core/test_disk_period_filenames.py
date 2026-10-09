"""Disk filenames remain portable without changing logical period keys."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage, OnDiskStorage


COMPOUND_PERIODS = [
    periods.period("day:2025-03-17:2"),
    periods.period("month:2025-03:2"),
    periods.period("year:2025:2"),
    periods.period("year:2025-03"),
]
EARLY_DAYS = [
    periods.Period((periods.DAY, periods.instant((year, 3, 17)), 2))
    for year in [1, 99, 999, 1000]
]
WINDOWS_RESERVED = set('<>:"/\\|?*')


@pytest.fixture
def portable_writes(monkeypatch):
    """Emulate Windows filename rejection on every CI operating system."""
    save = np.save

    def checked_save(file, *args, **kwargs):
        name = Path(file).name
        assert not WINDOWS_RESERVED.intersection(name), name
        return save(file, *args, **kwargs)

    monkeypatch.setattr(np, "save", checked_save)


@pytest.mark.parametrize("period", COMPOUND_PERIODS + EARLY_DAYS, ids=str)
def test_compound_period_has_a_portable_disk_filename(
    tmp_path, portable_writes, period
):
    storage = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    storage.put(np.array([42.0]), period)

    np.testing.assert_array_equal(storage.get(period), [42.0])
    assert storage.has(period)
    assert set(storage._files) == {f"default_{period}"}
    restored = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    restored.restore()
    np.testing.assert_array_equal(restored.get(period), [42.0])
    assert set(restored._files) == {f"default_{period}"}


@pytest.mark.parametrize("period", COMPOUND_PERIODS, ids=str)
def test_compound_period_clones_use_portable_replacement_paths(
    tmp_path, portable_writes, period
):
    storage = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    storage.put(np.array([1.0]), period)
    clone = storage.clone()

    storage.put(np.array([2.0]), period)
    clone.put(np.array([3.0]), period)
    clone.delete(period)
    clone.put(np.array([4.0]), period)

    np.testing.assert_array_equal(storage.get(period), [2.0])
    np.testing.assert_array_equal(clone.get(period), [4.0])
    files = list(tmp_path.rglob("*.npy"))
    assert len(files) == 3
    assert all(not WINDOWS_RESERVED.intersection(file.name) for file in files)
    restored = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    restored.restore()
    np.testing.assert_array_equal(restored.get(period), [1.0])


@pytest.mark.parametrize("period", COMPOUND_PERIODS, ids=str)
@pytest.mark.parametrize("branch", ["default", "policy;percent%3A"])
def test_disk_and_memory_keep_the_same_values_and_logical_periods(
    tmp_path, portable_writes, period, branch
):
    """Differential: physical encoding preserves values, keys and isolation."""
    disk = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    memory = InMemoryStorage(is_eternal=False)
    simple = periods.period("2025-01")
    for storage in [disk, memory]:
        storage.put(np.array([11.0, 12.0]), simple, branch)
        storage.put(np.array([21.0, 22.0]), period, branch, derived=True)

    assert set(disk.get_known_branch_periods()) == set(
        memory.get_known_branch_periods()
    )
    np.testing.assert_array_equal(disk.get(period, branch), memory.get(period, branch))
    assert disk.is_derived(period, branch) == memory.is_derived(period, branch)
    clones = [storage.clone() for storage in [disk, memory]]
    for clone in clones:
        clone.put(np.array([31.0, 32.0]), period, branch)
    np.testing.assert_array_equal(
        clones[0].get(period, branch), clones[1].get(period, branch)
    )
    np.testing.assert_array_equal(disk.get(period, branch), memory.get(period, branch))

    restored = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    restored.restore()
    assert set(restored.get_known_branch_periods()) == set(
        memory.get_known_branch_periods()
    )
    for known in [simple, period]:
        np.testing.assert_array_equal(
            restored.get(known, branch), memory.get(known, branch)
        )


@pytest.mark.skipif(
    os.name == "nt", reason="Windows cannot create legacy colon filenames"
)
def test_restore_still_reads_legacy_compound_period_filename(tmp_path):
    period = periods.period("day:2025-03-17:2")
    branch = "legacy;branch_with_underscore"
    np.save(tmp_path / f"{branch}_{period}.npy", np.array([42.0]))
    restored = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)

    restored.restore()

    np.testing.assert_array_equal(restored.get(period, branch), [42.0])
    assert set(restored._files) == {f"{branch}_{period}"}


@pytest.mark.parametrize("name", ["notes", "notes;percent%3A"])
def test_restore_tolerates_basenames_without_a_key_separator(tmp_path, name):
    np.save(tmp_path / f"{name}.npy", np.array([42.0]))
    restored = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)

    restored.restore()

    assert set(restored._files) == {name}
    np.testing.assert_array_equal(restored._decode_file(restored._files[name]), [42.0])


@pytest.mark.skipif(
    os.name == "nt", reason="Windows cannot create legacy colon filenames"
)
@pytest.mark.parametrize("reverse_listing", [False, True])
def test_portable_write_after_legacy_restore_wins_on_restore(
    tmp_path, monkeypatch, reverse_listing
):
    """An old file left beside its portable replacement cannot mask updates."""
    period = periods.period("month:2025-03:2")
    legacy_name = f"default_{period}.npy"
    np.save(tmp_path / legacy_name, np.array([1.0]))
    storage = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    storage.restore()
    storage.put(np.array([2.0]), period)
    names = os.listdir(tmp_path)
    assert len(names) == 2
    monkeypatch.setattr(
        os, "listdir", lambda directory: sorted(names, reverse=reverse_listing)
    )
    restored = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)

    restored.restore()

    np.testing.assert_array_equal(restored.get(period), [2.0])
    assert set(restored._files) == {f"default_{period}"}
