"""The temporary folder a simulation stores values on disk in.

A simulation with a ``MemoryConfig`` stores values on disk, each variable's
in a subfolder (an ``OnDiskStorage``) of the simulation's folder,
``Simulation.data_storage_dir``. Unless the caller chose that folder, the
simulation makes it, as a ``TemporaryStorageDirectory``, which removes it once
nothing that can read the files in it is left. What shares them:

* ``Simulation.clone`` and ``get_branch`` (a clone) copy the holders. Each
  copied disk storage is an ``OnDiskStorage.clone``: it reads the source's
  files, and keeps alive the storage that removes their subfolder and the
  folder object, so the files last while it does, whatever is collected
  first. A branch also keeps its parent (``parent_branch``). The disk
  storages a clone makes itself go in a folder of its own, made in the
  caller's folder if the source had one: had they gone in the source's, a
  variable both store after cloning would have one subfolder, files and all,
  that each wrote over and removed when collected.
* Pickling or copying a disk storage gives one that reads the same files and
  never removes them. In the process that made the folder, the copy keeps
  alive the storage that removes their subfolder, and this object (a copy of
  it is the object itself), as a clone does. Anywhere else nothing removes
  them: they belong to the process that made them.
* A process forked from the one that made the folder has copies of all these
  objects. Only the process that made a folder or subfolder removes it.
* Nothing removes a folder the caller chose (``_data_storage_dir``), only the
  subfolders disk storages made in it.
"""

import os
import shutil
import tempfile
import weakref
from typing import Optional

# The directory each path names, among those created in this process and not
# yet removed, so that unpickling one in this process gives back the same
# object (see ``TemporaryStorageDirectory.__reduce__``).
_LIVE: "weakref.WeakValueDictionary[str, TemporaryStorageDirectory]" = (
    weakref.WeakValueDictionary()
)


def _remove_directory(path: str, creator_pid: int) -> None:
    # A process forked after the directory was created has a copy of this
    # object, and runs this when its copy is collected or it exits; the
    # directory still belongs to the process that created it.
    if os.getpid() == creator_pid:
        shutil.rmtree(path, ignore_errors=True)


class TemporaryStorageDirectory:
    """A temporary directory for values a simulation stores on disk.

    The directory is removed once this object is garbage-collected, or at
    interpreter exit, whichever comes first, and only by the process that
    created it. So whatever reads or writes files in the directory keeps a
    reference to this object: the simulation that created it and every disk
    storage in it (see ``Holder.create_disk_storage``), which the storages
    cloned or copied from those keep alive in turn. The directory then lasts
    exactly as long as something can read it.

    Copying this object gives the same object. Unpickling it in the process
    that created it, while the directory exists, also gives the same object;
    anywhere else it gives one that never removes the directory, which belongs
    to the process that created it.
    """

    def __init__(self, parent: Optional[str] = None):
        """Create the directory, in ``parent`` or the default temporary directory."""
        self.path = tempfile.mkdtemp(prefix="openfisca_", dir=parent)
        self._finalizer = weakref.finalize(
            self, _remove_directory, self.path, os.getpid()
        )
        _LIVE[self.path] = self

    @property
    def removes_directory(self) -> bool:
        """Whether this object removes the directory when it is collected."""
        finalizer = getattr(self, "_finalizer", None)
        return finalizer is not None and finalizer.alive

    def __reduce__(self):
        return _attach, (self.path,)

    def __copy__(self) -> "TemporaryStorageDirectory":
        return self

    def __deepcopy__(self, memo: dict) -> "TemporaryStorageDirectory":
        return self

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path!r}>"


def _attach(path: str) -> TemporaryStorageDirectory:
    """The live directory object for ``path``, or one that never removes it."""
    directory = _LIVE.get(path)
    if directory is None:
        directory = TemporaryStorageDirectory.__new__(TemporaryStorageDirectory)
        directory.path = path
        directory._finalizer = None
    return directory
