"""Regression test: Dataset.__getattr__ must not load files on dunder lookups (H8).

Previously ``__getattr__`` called ``self.load(name)`` for ANY missing
attribute — including introspection probes like ``copy.deepcopy(ds)``,
``pickle.dumps(ds)``, ``hasattr(ds, '_whatever')``, and IDE / debugger
``repr`` calls. Those triggered a real H5/CSV read, leaked file handles,
and could trigger a network download in the ``FLAT_FILE`` + ``url``
pipeline.
"""

from __future__ import annotations

import pytest

from policyengine_core.data.dataset import Dataset


class _FakeDataset(Dataset):
    name = "fake"
    label = "fake"
    data_format = Dataset.ARRAYS
    time_period = 2020
    file_path = "/tmp/nonexistent-policyengine-test-dataset.h5"

    def __init__(self):  # bypass the parent's file-existence check.
        self._table_cache = {}
        self.loaded_keys = []

    def load(self, key=None, mode="r"):
        # Track which keys were asked for so the tests can assert that
        # introspection DOESN'T reach here.
        self.loaded_keys.append(key)
        raise RuntimeError(
            f"Dataset.load({key!r}) should not be triggered by introspection."
        )


def test_dunder_attribute_lookup_does_not_trigger_load():
    ds = _FakeDataset()
    # hasattr does ``try: getattr; except AttributeError: False``. Before
    # the fix these lookups went through load(), which raised a
    # RuntimeError (under our fake) or opened a real H5 file (for real
    # datasets). After the fix they must raise AttributeError and never
    # touch ``load``.
    assert not hasattr(ds, "__deepcopy__")
    assert not hasattr(ds, "__setstate_ex__")
    assert not hasattr(ds, "__getinitargs__")
    # load must never have been called.
    assert ds.loaded_keys == []


def test_private_attribute_lookup_does_not_trigger_load():
    ds = _FakeDataset()
    with pytest.raises(AttributeError):
        _ = ds._not_a_real_attribute
    assert ds.loaded_keys == []
