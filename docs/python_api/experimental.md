# Experimental features

`policyengine_core.experimental` houses new features that aren't fully stable yet, and may be removed in future versions. 

## MemoryConfig

A simulation with a `MemoryConfig` stores values on disk in `Simulation.data_storage_dir`: a temporary folder it makes, removed once the simulation and every clone, branch or disk storage that reads files in it have been garbage-collected, or at interpreter exit. A clone or branch, or a process forked from the one that made the folder, makes the disk storages it needs itself in a folder of its own; those it copied or inherited keep storing in the folder they came from, without writing over files their source reads. A disk storage made with `preserve_storage_dir=True` (`Holder.create_disk_storage(preserve=True)`, say), or set to it later, keeps its subfolder: the folder is then left holding that subfolder, for you to remove. To choose the folder instead, set `simulation._data_storage_dir` before anything is stored on disk. That folder is never removed, though what simulations make in it is, unless it is in a temporary folder a simulation made: that one is then kept alive by this simulation too, and removed, with everything in it, once nothing keeps it. Between processes the guarantees are narrower: nothing in a forked process, or one a disk storage is unpickled in, keeps the folder alive, and when the process that made it removes it, everything in it goes, including a folder a forked process made inside it. The [data storage reference](data_storage.md) lists what holds for each way of sharing the files.

```{eval-rst}
.. autoclass:: policyengine_core.experimental.memory_config.MemoryConfig
    :members:
    :undoc-members:
    :show-inheritance:
```