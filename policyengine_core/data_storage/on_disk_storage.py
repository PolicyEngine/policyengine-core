import os
import shutil
import tempfile
import weakref

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.data_storage.storage_directory import (
    directory_containing,
    path_key,
)
from policyengine_core.enums import EnumArray
from policyengine_core.periods import Period

# Subdirectory of a storage directory for the files ``put`` writes when a
# key's usual file is shared with a clone (see ``OnDiskStorage._path_to_write``).
# ``restore`` reads only the files directly in the directory, so it never
# reads these.
REPLACEMENTS_DIR = "replaced"

# The storage that removes each storage directory when collected (the last
# one made without ``preserve_storage_dir`` or set to remove it), by
# ``path_key``, among those alive in this process: what a copy of a storage
# keeps alive (see ``__setstate__``).
_DIRECTORY_OWNERS: "weakref.WeakValueDictionary[str, OnDiskStorage]" = (
    weakref.WeakValueDictionary()
)

# How many processes this one has forked. A forked process may read any file
# a storage wrote before the fork, so from then on the storage writes over
# none of them (see ``_epoch``).
_forks = 0


def _count_fork() -> None:
    global _forks
    _forks += 1


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_parent=_count_fork)


def _epoch() -> tuple:
    """Changes in a process when it forks, and differs in the forked one."""
    return os.getpid(), _forks


class OnDiskStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors on disk
    """

    # Defaults for ``__del__``, which also runs on an instance whose
    # ``__init__`` did not: such an instance removes nothing.
    _preserve_storage_dir = True
    _creator_pid = None
    # The simulation's temporary directory this storage's directory is in, if
    # any, kept alive while this storage (or a clone or copy of it) reads
    # files in it (see ``TemporaryStorageDirectory``).
    _parent_directory = None
    # Whether this storage writes only new files (see ``_path_to_write``).
    _detached = False
    _own_epoch = None

    def __init__(
        self,
        storage_dir: str,
        is_eternal: bool = False,
        preserve_storage_dir: bool = False,
    ):
        self._start(storage_dir, is_eternal, directory_containing(storage_dir))
        if preserve_storage_dir:
            # Made to read a directory a live storage removes: it keeps that
            # storage alive while it reads the files, as a clone does.
            owner = _DIRECTORY_OWNERS.get(path_key(storage_dir))
            if owner is not None:
                self._storage_dir_owner = owner
        self.preserve_storage_dir = preserve_storage_dir

    def _start(self, storage_dir: str, is_eternal: bool, parent_directory) -> None:
        self._files = {}
        self._enums = {}
        # File keys stored with ``put(..., derived=True)``; see
        # ``InMemoryStorage``.
        self._derived = set()
        # For each key, the file this storage last wrote for it, while it has
        # not shared that file since: the only files of its family ``put``
        # writes over (see ``_path_to_write``). Kept when the key is deleted,
        # so that writing it again reuses the file. Only valid in the epoch
        # (``_epoch``) it was recorded in.
        self._own_paths = {}
        self._own_epoch = _epoch()
        # Paths of every file this storage, the storage it was cloned from or
        # any other clone of either wrote or stored when cloned: one set,
        # shared by all of them.
        self._family_files = set()
        # Paths ``restore`` read back, by this storage or any clone in its
        # family: one set, shared like ``_family_files``. They join the
        # family once something else may read them (see ``restore``): those
        # this storage reads, when it is cloned or copied; all of them, once
        # this process forks (see ``_catch_up_after_fork``).
        self._restored_paths = set()
        self.is_eternal = is_eternal
        self.storage_dir = storage_dir
        self._creator_pid = os.getpid()
        self._parent_directory = parent_directory

    @property
    def preserve_storage_dir(self) -> bool:
        """Whether this storage leaves its directory when it is collected.

        A storage made with ``False`` removes its directory when collected,
        in the process that made it. One made with ``True``, or set to
        ``True`` later, preserves the directory: neither it nor the removal of
        a simulation's temporary directory containing it removes it (see
        ``TemporaryStorageDirectory``). Two kinds of storage read another
        storage's directory instead, and preserve nothing: a clone or a copied
        storage, for which this is ``True``; and a storage made with ``True``
        for a directory that a live storage made with ``False`` removes. Each
        keeps that storage alive, so the files last while it reads them.
        """
        return self._preserve_storage_dir

    @preserve_storage_dir.setter
    def preserve_storage_dir(self, preserve: bool) -> None:
        self._preserve_storage_dir = preserve
        key = path_key(self.storage_dir)
        directory = self._parent_directory
        if not preserve:
            _DIRECTORY_OWNERS[key] = self
            if directory is not None:
                directory.release(key)
            return
        owner = _DIRECTORY_OWNERS.get(key)
        if directory is not None and (owner is None or owner.preserve_storage_dir):
            directory.keep(key)

    def _writes_only_new_files(self) -> bool:
        """Whether storages this one cannot know of write in its directory:
        it is copied (``__setstate__``), cloned from a copy, or in a process
        forked from the one that made it. So ``put`` writes only new files of
        its own (see ``_path_to_write``)."""
        return self._detached or self._creator_pid != os.getpid()

    def _catch_up_after_fork(self) -> None:
        """If this process forked since this storage last wrote, was cloned
        or was copied, the forked process may read any file this storage, or
        any clone in its family, wrote or read back before:
        those it wrote are in the family, and those it read back join it (all
        it ever read back, not only those it reads now, since the forked
        process may still read one this storage has since dropped). Called
        before every write, and before a clone or copy shares the family, so
        a clone made after the fork writes over none of them either."""
        epoch = _epoch()
        if self._own_epoch != epoch:
            self._family_files.update(self._restored_paths, self._files.values())
            self._own_paths = {}
            self._own_epoch = epoch

    def __getstate__(self) -> dict:
        # Whatever this state is read into reads the same files, so from now
        # on this storage writes over none of them, as after ``clone``:
        # those ``restore`` read back included.
        self._catch_up_after_fork()
        self._family_files.update(self._files.values())
        self._own_paths = {}
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        # Containers of its own: ``copy.copy`` passes the source's.
        state["_files"] = dict(state.get("_files", {}))
        state["_enums"] = dict(state.get("_enums", {}))
        # A storage pickled before derived marks existed has none: its values
        # count as inputs.
        state["_derived"] = set(state.get("_derived", ()))
        # Nor did it record the files of its family.
        state["_family_files"] = set(
            state.get("_family_files", state["_files"].values())
        )
        # The flag was a plain attribute.
        state.pop("preserve_storage_dir", None)
        # A copy reads the files of the storage it was copied from, so it
        # never removes their directory, which that storage, its clones or
        # another copy may still read; it keeps alive instead the storage in
        # this process that removes it, as a clone does, and the directory
        # containing it. Elsewhere (another process) nothing here removes the
        # directory. Other storages may write in it without knowing of this
        # one, so it writes only new files.
        state["_preserve_storage_dir"] = True
        state["_detached"] = True
        state["_own_paths"] = {}
        state["_own_epoch"] = _epoch()
        state["_restored_paths"] = set()
        owner = _DIRECTORY_OWNERS.get(path_key(state["storage_dir"]))
        if owner is not None:
            state["_storage_dir_owner"] = owner
        if state.get("_parent_directory") is None:
            state["_parent_directory"] = directory_containing(state["storage_dir"])
        self.__dict__.update(state)

    def clone(self) -> "OnDiskStorage":
        """Create a private metadata view over this storage directory.

        The file and enum mappings are copied so deleting or rewiring entries
        through the clone does not mutate the source storage. The underlying
        ``.npy`` files are shared, so neither storage writes over them: a later
        ``put`` of one of their keys, through either storage, writes a new
        file (see ``_path_to_write``). Clones retain the original cleanup owner
        so the shared directory stays alive, but never own cleanup themselves,
        nor preserve the directory (see ``preserve_storage_dir``).
        """
        self._catch_up_after_fork()
        clone = OnDiskStorage.__new__(OnDiskStorage)
        clone._start(self.storage_dir, self.is_eternal, self._parent_directory)
        clone._preserve_storage_dir = True
        clone._detached = self._writes_only_new_files()
        clone._files = self._files.copy()
        clone._enums = self._enums.copy()
        clone._derived = set(self._derived)
        # One set of read-back paths for the family, so a clone made before
        # ``restore`` still writes over none of them after a fork.
        clone._restored_paths = self._restored_paths
        clone._storage_dir_owner = getattr(self, "_storage_dir_owner", self)
        # Both storages now read every file stored so far, including any this
        # family did not write (read back by ``restore``, say).
        self._family_files.update(self._files.values())
        clone._family_files = self._family_files
        self._own_paths = {}
        return clone

    def _decode_file(self, file: str) -> ArrayLike:
        enum = self._enums.get(file)
        if enum is not None:
            return EnumArray(numpy.load(file), enum)
        else:
            return numpy.load(file)

    def get(self, period: Period, branch_name: str = "default") -> ArrayLike:
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)

        values = self._files.get(f"{branch_name}_{period}")
        if values is None:
            return None
        return self._decode_file(values)

    def has(self, period: Period, branch_name: str = "default") -> bool:
        """Whether a value is stored for ``period`` under ``branch_name``.

        Unlike ``get``, this reads no file.
        """
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        return f"{branch_name}_{periods.period(period)}" in self._files

    def is_derived(self, period: Period, branch_name: str = "default") -> bool:
        """Whether the value stored for ``period`` under ``branch_name`` was
        stored with ``derived=True``; ``False`` if none is stored."""
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        key = f"{branch_name}_{periods.period(period)}"
        return key in self._derived and key in self._files

    def put(
        self,
        value: ArrayLike,
        period: Period,
        branch_name: str = "default",
        derived: bool = False,
    ) -> None:
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)

        filename = f"{branch_name}_{period}"
        path = self._path_to_write(filename)
        if isinstance(value, EnumArray):
            self._enums[path] = value.possible_values
            value = value.view(numpy.ndarray)
        numpy.save(path, value)
        self._files[filename] = path
        self._own_paths[filename] = path
        self._family_files.add(path)
        if derived:
            self._derived.add(filename)
        else:
            self._derived.discard(filename)

    def _path_to_write(self, filename: str) -> str:
        """The path ``put`` writes the value for key ``filename`` to.

        Storages cloned from one another share a directory and the files
        stored before cloning, and each can store the same key afterwards,
        which names the same path. Writing over a file another of them reads
        would change that storage's value. So ``put`` writes over one of
        their files only if this storage wrote it and has not shared it since;
        otherwise the value goes to a new file of its own, in
        ``REPLACEMENTS_DIR``. A file only storages outside this family wrote
        is written over, as before.

        A copied storage, or one in a process forked from the one that made
        it, shares the directory with storages it cannot know of: it writes
        every value it has not written itself in this process to a new file.
        And after this process forks, the forked process may read any file
        this storage wrote or read back (``restore``) before, so this storage
        writes over none of them.
        """
        self._catch_up_after_fork()
        own = self._own_paths.get(filename)
        if own is not None:
            return own
        path = os.path.join(self.storage_dir, filename) + ".npy"
        if path not in self._family_files and not self._writes_only_new_files():
            return path
        directory = os.path.join(self.storage_dir, REPLACEMENTS_DIR)
        os.makedirs(directory, exist_ok=True)
        descriptor, path = tempfile.mkstemp(
            prefix=f"{filename}.", suffix=".npy", dir=directory
        )
        os.close(descriptor)
        return path

    def delete(self, period: Period = None, branch_name: str = "default") -> None:
        if period is None:
            # Only wipe files belonging to the requested branch (previously
            # this wiped every branch regardless of ``branch_name`` — same
            # class of bug as C2 in InMemoryStorage).
            branch_prefix = f"{branch_name}_"
            self._files = {
                period_item: value
                for period_item, value in self._files.items()
                if not period_item.startswith(branch_prefix)
            }
            self._derived.intersection_update(self._files)
            return

        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)

        if period is not None:
            self._files = {
                period_item: value
                for period_item, value in self._files.items()
                if not period_item == f"{branch_name}_{period}"
            }
            self._derived.intersection_update(self._files)

    def get_known_periods(self) -> list:
        return list([periods.period(x.split("_")[1]) for x in self._files.keys()])

    def get_known_branch_periods(self) -> list:
        return [
            (branch_name, periods.period(period))
            for branch_name, period in map(lambda x: x.split("_"), self._files.keys())
        ]

    def restore(self) -> None:
        """Read back the values stored in this storage's directory: for each
        key, the file named for it (not those in ``REPLACEMENTS_DIR``).

        This storage writes over those files, as over those it wrote, until
        it is cloned or copied, or this process forks: from then on it writes
        a new file instead, since the clone, copy or forked process reads
        them. A storage made separately for the directory, the one that wrote
        them say, still writes over them, as before. So may another storage
        in this one's family, when ``restore`` runs after it was cloned and
        this process has not forked since.
        """
        self._files = files = {}
        # Files read back from a directory carry no derived marks.
        self._derived = set()
        # Restore self._files from content of storage_dir.
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".npy"):
                continue
            path = os.path.join(self.storage_dir, filename)
            filename_core = filename.rsplit(".", 1)[0]
            files[filename_core] = path
        self._restored_paths.update(files.values())

    def __del__(self, _rmtree=shutil.rmtree, _getpid=os.getpid) -> None:
        # (The defaults keep the two functions reachable while the
        # interpreter shuts down.) Only in the process that created this
        # storage: a process forked from it has a copy, and the directory is
        # still the creator's.
        if self._preserve_storage_dir or self._creator_pid != _getpid():
            return
        # Remove this storage's directory, with the files of its clones (each
        # keeps this storage alive). Not the directory containing it: that is
        # a simulation's, removed with the simulation (see
        # ``TemporaryStorageDirectory``), or one the caller chose.
        _rmtree(self.storage_dir, ignore_errors=True)
