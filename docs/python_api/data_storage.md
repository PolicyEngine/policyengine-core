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

```{eval-rst}
.. autoclass:: policyengine_core.data_storage.on_disk_storage.OnDiskStorage
    :members:
    :undoc-members:
    :show-inheritance:
```

## Storage keys

Both storage backends reject branch names containing `:` and month or year
periods anchored after the first day of a month. They also reject periods whose
string form cannot be parsed back, such as the year `0999`, before storing a
value. Existing aliases, including twelve-month periods serialized as one year,
remain supported. Disk restoration warns and skips filenames without a branch
separator and a canonical period string.

```{eval-rst}
.. automodule:: policyengine_core.data_storage.storage_keys
    :members:
```

## TemporaryStorageDirectory

```{eval-rst}
.. automodule:: policyengine_core.data_storage.storage_directory
    :no-members:

.. autoclass:: policyengine_core.data_storage.storage_directory.TemporaryStorageDirectory
    :members:
    :undoc-members:
    :show-inheritance:
```
