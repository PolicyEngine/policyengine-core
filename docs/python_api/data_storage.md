# Data storage

The `policyengine_core.data_storage` module contains the classes that are used to handle the storage of data in simulations.

## InMemoryStorage

```{eval-rst}
.. autoclass:: policyengine_core.data_storage.in_memory_storage.InMemoryStorage
    :members:
    :undoc-members:
    :show-inheritance:
```

## OnDiskStorage

Each independently created temporary holder storage gets its own subdirectory,
so replacing a holder leaves files read by existing clones intact. Cloned
storages retain their source's directory and use new files for later writes.
Explicit `Holder.create_disk_storage(directory=...)` calls and preserved
storages keep the `directory/variable` layout used by simulation dumps.

`OnDiskStorage.restore()` reads direct key files in its directory. It does not
read files in the `replaced` subdirectory or select the latest write across
independently constructed storages. Clone or copy a restored storage before
writing when its existing values must remain isolated.

```{eval-rst}
.. autoclass:: policyengine_core.data_storage.on_disk_storage.OnDiskStorage
    :members:
    :undoc-members:
    :show-inheritance:
```

## TemporaryStorageDirectory

A caller-supplied `simulation._data_storage_dir` is created on first use if it
does not exist. It keeps the caller-owned lifetime described below.
Exits that skip finalizers (such as `os._exit` in multiprocessing workers) can leave uniquely named holder subdirectories in a caller-owned directory to accumulate across runs; the caller must remove them.

```{eval-rst}
.. automodule:: policyengine_core.data_storage.storage_directory
    :no-members:

.. autoclass:: policyengine_core.data_storage.storage_directory.TemporaryStorageDirectory
    :members:
    :undoc-members:
    :show-inheritance:
```
