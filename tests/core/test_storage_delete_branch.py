"""Regression tests for InMemoryStorage.delete and OnDiskStorage.delete (C2).

Previously, ``InMemoryStorage.delete(period, branch_name)``:

* When ``period`` was ``None``, wiped every branch's data regardless of
  ``branch_name``.
* When ``period`` was given, filtered by period only (ignoring
  ``branch_name``), so deleting a period for one branch removed it for every
  branch.

Both shapes are regression-tested here.
"""

from __future__ import annotations

import numpy as np

from policyengine_core.data_storage.in_memory_storage import InMemoryStorage


def _populated():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.asarray([1.0]), "2024-01", branch_name="default")
    storage.put(np.asarray([2.0]), "2024-01", branch_name="reform")
    storage.put(np.asarray([3.0]), "2024-02", branch_name="default")
    storage.put(np.asarray([4.0]), "2024-02", branch_name="reform")
    return storage


def test_delete_period_respects_branch_name():
    storage = _populated()
    storage.delete("2024-01", branch_name="reform")
    # The reform entry for 2024-01 must be gone, but the default one must
    # still be present (previously it was also deleted).
    assert storage.get("2024-01", "default") is not None
    assert storage.get("2024-01", "reform") is None
    # Untouched periods stay intact on both branches.
    assert storage.get("2024-02", "default") is not None
    assert storage.get("2024-02", "reform") is not None


def test_delete_all_periods_respects_branch_name():
    storage = _populated()
    # Wipe the reform branch only.
    storage.delete(period=None, branch_name="reform")
    assert storage.get("2024-01", "default") is not None
    assert storage.get("2024-02", "default") is not None
    assert storage.get("2024-01", "reform") is None
    assert storage.get("2024-02", "reform") is None


def test_delete_period_covers_subperiods_only_on_requested_branch():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.asarray([1.0]), "2024-01", branch_name="default")
    storage.put(np.asarray([2.0]), "2024-02", branch_name="default")
    storage.put(np.asarray([3.0]), "2024-01", branch_name="reform")
    storage.put(np.asarray([4.0]), "2024-02", branch_name="reform")
    # Deleting the whole year on "reform" should drop both month entries on
    # reform but leave default untouched.
    storage.delete("2024", branch_name="reform")
    assert storage.get("2024-01", "default") is not None
    assert storage.get("2024-02", "default") is not None
    assert storage.get("2024-01", "reform") is None
    assert storage.get("2024-02", "reform") is None
