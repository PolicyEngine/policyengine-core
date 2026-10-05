# Experimental features

`policyengine_core.experimental` houses new features that aren't fully stable yet, and may be removed in future versions. 

## MemoryConfig

A simulation with a `MemoryConfig` stores values on disk in `Simulation.data_storage_dir`: a temporary folder it makes, removed once the simulation and every clone, branch or copied disk storage that reads files in it have been garbage-collected, or at interpreter exit. A clone or branch stores its own new values in a folder of its own. To choose the folder instead, set `simulation._data_storage_dir` before anything is stored on disk. That folder is never removed, though what simulations make in it is.

```{eval-rst}
.. autoclass:: policyengine_core.experimental.memory_config.MemoryConfig
    :members:
    :undoc-members:
    :show-inheritance:
```