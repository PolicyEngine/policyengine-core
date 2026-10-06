"""The temporary folder a simulation stores values on disk in.

A simulation with a ``MemoryConfig`` stores values on disk, each variable's
in a subfolder (an ``OnDiskStorage``) of the simulation's folder,
``Simulation.data_storage_dir``. Unless the caller chose that folder, the
simulation makes it, as a ``TemporaryStorageDirectory``. That object removes
the folder, with everything in it but the subfolders of preserved storages
(below), when it is garbage-collected, or at interpreter exit, and only in
the process that made it. What reads files in the folder, or stores values
in a folder inside it, keeps the object alive. What each way of sharing the
files guarantees:

* **Within the process that made the folder.** The simulation keeps the
  folder, and so does every disk storage anywhere under it, whether a holder
  made it or the caller constructed it there. ``Simulation.clone``,
  ``get_branch`` and ``derivative`` copy the holders: each copied disk
  storage is an ``OnDiskStorage.clone`` that reads the source's files and
  keeps alive the folder and the storage that removes their subfolder. A
  pickled or copied disk storage does the same. The disk storages a clone
  makes itself go in a folder of its own.

  Folders nest: a simulation given the folder, or a folder in it, as its
  own (``_data_storage_dir``) keeps the folder alive from then on, and so
  does a clone of that simulation, which makes its own folder there. A
  folder made inside another live one keeps that one alive, and has it
  leave the subfolders it preserves, however deep the nesting.

  So nothing removes a file any of these reads while it is alive, whatever
  is collected first, with one exception, as before: a disk storage removes
  everything in its subfolder when collected, including files there that
  another storage reads without keeping it alive. Those are the files of a
  second storage made to remove the same subfolder, and anything made
  inside the subfolder. Nor does a storage write over a file another reads:
  storages cloned or copied from one another write a new file instead (see
  ``OnDiskStorage._path_to_write``), for the files any of them read back
  before (``OnDiskStorage.restore``) too. Storages each made for one
  subfolder still write the same file for a key, as before, and ``restore``
  reads whatever files are in the subfolder when it runs. A ``restore`` run
  after a storage was cloned, with no fork since, is not covered either:
  another storage in its family may still write over a file it read back.
* **Explicitly preserved storages.** A disk storage made with
  ``preserve_storage_dir=True`` (``Holder.create_disk_storage(preserve=True)``,
  say), or whose ``preserve_storage_dir`` is later set to ``True``, keeps its
  subfolder: removing the folder, or a folder containing it that a
  simulation made, leaves that subfolder, every file in it and the folders
  leading to it, for the caller to remove. Exceptions: a subfolder inside
  another disk storage's goes when that storage removes its own, as above;
  and two kinds of storage read another storage's subfolder rather than
  keep one. Those are a clone or copy, whose flag is ``True`` only because
  it never removes its source's subfolder, and a storage made with the flag
  for a subfolder that a live storage made without it removes. Each keeps
  that storage alive while it reads the files, which that storage still
  removes once collected, as before.
* **Another process.** A process forked from the one that made the folder
  has copies of all these objects; so does a process a disk storage is
  unpickled in. Each process removes only the folders and disk storage
  subfolders it made, but removing a folder removes everything in it, the
  folder a forked process made inside it included. Disk storages copied
  into another process, by forking or unpickling, or cloned from one there,
  write only new files of their own, and after forking, the disk storages
  of the process that made the folder write over no file they wrote or read
  back before. So neither process writes over a file the other reads
  through these storages. A forked process stores what new holders put on
  disk in a folder of its own, made inside the folder it inherited. But
  nothing in another process keeps the folder alive: once the process that
  made it collects the simulation and its storages, or exits, the folder is
  gone, with everything in it, the forked process's own folder included,
  even while a forked or unpickling process still reads them. Explicit
  preservation is likewise kept only by the process that made the folder.
* **A folder the caller chose** (``_data_storage_dir``). Nothing removes it,
  only the subfolders disk storages made in it, as before, unless it is in a
  folder a simulation made: that one removes it with everything else in it,
  once nothing keeps that one alive (above). A clone or forked process makes
  its own folder inside it.
* **A process that ends without running the exit handlers that remove its
  folders** (killed, or ended with ``os._exit``, as a process
  ``multiprocessing`` forks is) leaves the folders it made that were still
  alive, except those a forked process made inside a folder it inherited,
  which go with that folder.
"""

import os
import shutil
import tempfile
import weakref
from typing import Optional


def path_key(path: str) -> str:
    """The key ``path`` is looked up by: absolute, with links resolved."""
    return os.path.normcase(os.path.realpath(path))


# The directory each path names (by ``path_key``), among those created in this
# process and not yet removed, so that unpickling one in this process gives
# back the same object (see ``TemporaryStorageDirectory.__reduce__``), and a
# disk storage made in one finds it (see ``directory_containing``).
_LIVE: "weakref.WeakValueDictionary[str, TemporaryStorageDirectory]" = (
    weakref.WeakValueDictionary()
)


def directory_containing(path: str) -> Optional["TemporaryStorageDirectory"]:
    """The live directory made in this process that is ``path`` or contains
    it, the innermost if several do; ``None`` if none does."""
    key = path_key(path)
    while True:
        directory = _LIVE.get(key)
        if directory is not None:
            return directory
        parent = os.path.dirname(key)
        if parent == key:
            return None
        key = parent


def _remove_directory(path: str, key: str, creator_pid: int, kept: set) -> None:
    # A process forked after the directory was created has a copy of this
    # object, and runs this when its copy is collected or it exits; the
    # directory still belongs to the process that created it.
    if os.getpid() != creator_pid:
        return
    kept = {kept_key for kept_key in kept if os.path.lexists(kept_key)}
    if key in kept:
        return
    if kept:
        _remove_all_but(key, kept)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _remove_all_but(folder: str, kept: set) -> None:
    """Remove everything in ``folder`` (a ``path_key``) but the paths in
    ``kept`` and the folders leading to them."""
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return
    for entry in entries:
        key = os.path.join(folder, os.path.normcase(entry.name))
        if key in kept:
            continue
        is_folder = entry.is_dir(follow_symlinks=False)
        if any(kept_key.startswith(key + os.sep) for kept_key in kept):
            if is_folder:
                _remove_all_but(key, kept)
            continue
        if is_folder:
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            try:
                os.unlink(entry.path)
            except OSError:
                pass


class TemporaryStorageDirectory:
    """A temporary directory for values a simulation stores on disk.

    The directory is removed once this object is garbage-collected, or at
    interpreter exit, whichever comes first, and only by the process that
    created it. The simulation that created it keeps a reference to this
    object, and so does every disk storage made in it in this process (see
    ``OnDiskStorage``), which the storages cloned or copied from those keep
    alive in turn. So do a simulation given this directory or a folder in it
    (``Simulation._data_storage_dir``), and a directory of this class made
    in it. Subfolders of disk storages that preserve theirs are left, with
    the folders leading to them, here or in a directory made in this one.
    What this guarantees for each way of sharing the files is in the
    docstring of this module.

    Copying this object gives the same object. Unpickling it in the process
    that created it, while the directory exists, also gives the same object;
    anywhere else it gives one that never removes the directory, which belongs
    to the process that created it.
    """

    # The live directory this one was made in, if any (see ``__init__``).
    _container = None

    def __init__(self, parent: Optional[str] = None):
        """Create the directory, in ``parent`` or the default temporary directory."""
        self.path = tempfile.mkdtemp(prefix="openfisca_", dir=parent)
        self._creator_pid = os.getpid()
        key = path_key(self.path)
        # The subfolders (by ``path_key``) left when the directory is removed
        # (see ``keep``); the finalizer reads this set, not this object.
        self._kept = set()
        # Removing a live directory this one was made in (a simulation given
        # another's folder made this one in it, say) would remove this one
        # with it, so this one keeps that one alive, and has it leave what
        # this one keeps (see ``keep``). That one may keep another, and so on.
        self._container = directory_containing(os.path.dirname(self.path))
        self._finalizer = weakref.finalize(
            self, _remove_directory, self.path, key, self._creator_pid, self._kept
        )
        _LIVE[key] = self

    @property
    def removes_directory(self) -> bool:
        """Whether this object removes the directory when it is collected."""
        finalizer = getattr(self, "_finalizer", None)
        return (
            finalizer is not None
            and finalizer.alive
            and self._creator_pid == os.getpid()
        )

    def keep(self, path: str) -> None:
        """Leave ``path``, a folder in the directory, when removing it, or
        any live directory this one was made in."""
        key = path_key(path)
        directory = self
        while directory is not None:
            directory._kept.add(key)
            directory = directory._container

    def release(self, path: str) -> None:
        """Remove ``path`` with the directory again (see ``keep``)."""
        key = path_key(path)
        directory = self
        while directory is not None:
            directory._kept.discard(key)
            directory = directory._container

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
    directory = _LIVE.get(path_key(path))
    if directory is None:
        directory = TemporaryStorageDirectory.__new__(TemporaryStorageDirectory)
        directory.path = path
        directory._creator_pid = None
        directory._kept = set()
        directory._finalizer = None
    return directory
