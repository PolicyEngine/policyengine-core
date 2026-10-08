# Data storage

The `policyengine_core.data_storage` module contains the classes that are used to handle the storage of data in simulations.

Both storage classes provide `delete_exact(period, branch_name, derived_only)`
to remove a single stored period, leaving contained periods in place.
With `derived_only=True`, an input at that key remains stored; this is
used to discard provisional results from cut period recursions.

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

## TemporaryStorageDirectory

```{eval-rst}
.. automodule:: policyengine_core.data_storage.storage_directory
    :no-members:

.. autoclass:: policyengine_core.data_storage.storage_directory.TemporaryStorageDirectory
    :members:
    :undoc-members:
    :show-inheritance:
```
