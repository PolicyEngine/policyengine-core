import os
import shutil
import tempfile
import weakref
from typing import Dict, Optional, Tuple, Union

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.enums import EnumArray
from policyengine_core.periods import Period

# Incremented in the parent and the child at every fork. A file created before
# a fork may be read by the other process, so neither overwrites it in place.
_fork_generation = 0


def _after_fork() -> None:
    global _fork_generation
    _fork_generation += 1


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_parent=_after_fork, after_in_child=_after_fork)

# The live handle for each directory path, so that every storage in this
# process that uses a directory shares one handle, and so one set of readers.
_live_directories: "weakref.WeakValueDictionary[str, StorageDirectory]" = (
    weakref.WeakValueDictionary()
)


def _remove_directory(path: str, pid: int) -> None:
    # A forked child inherits the finalizer, but the directory belongs to the
    # process that created it.
    if os.getpid() == pid:
        shutil.rmtree(path, ignore_errors=True)


def _create_exclusively(path: str) -> bool:
    """Create an empty file at ``path``, or return False if one exists."""
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def _find_directory(path: str, pid: int) -> "StorageDirectory":
    """Unpickle a directory handle.

    In the process that made it, this is the live handle, which keeps the
    directory. Elsewhere it is a handle that never removes the directory.
    """
    if os.getpid() == pid:
        directory = _live_directories.get(path)
        if directory is not None:
            return directory
    directory = StorageDirectory(path)
    directory.preserve = True
    return directory


class StorageDirectory:
    """A directory removed, with its contents, once nothing references it.

    Whatever may read a file in the directory holds a reference to this
    object, so the directory stays on disk exactly as long as one of them is
    alive. A directory created inside another holds that one as ``parent``,
    and preserving a directory preserves its parents.

    ``readers`` records, for each file, the storages that may read it other
    than the one that created it, so that storage knows when it may
    overwrite the file. Copying a handle, with ``copy`` or ``deepcopy``,
    returns the handle itself.
    """

    def __init__(self, path: str, parent: Optional["StorageDirectory"] = None):
        self.path = os.path.normpath(path)
        self.parent = parent
        self.readers: Dict[str, "weakref.WeakSet[OnDiskStorage]"] = {}
        self._pid = os.getpid()
        self._finalizer = weakref.finalize(
            self, _remove_directory, self.path, self._pid
        )
        _live_directories[self.path] = self

    @classmethod
    def for_path(cls, path: str) -> "StorageDirectory":
        """Return this process's handle for ``path``, creating one if needed."""
        directory = _live_directories.get(os.path.normpath(path))
        if directory is not None and directory._pid == os.getpid():
            return directory
        return cls(path)

    @classmethod
    def create(
        cls,
        prefix: str,
        parent: Union["StorageDirectory", str, None] = None,
    ) -> "StorageDirectory":
        """Create a new, uniquely named directory inside ``parent``.

        ``parent`` is another :class:`StorageDirectory`, a path, or ``None``
        for the system temporary directory.
        """
        if isinstance(parent, StorageDirectory):
            parent_path = parent.path
        else:
            parent_path = parent
            parent = None
        if parent_path is not None:
            os.makedirs(parent_path, exist_ok=True)
        return cls(tempfile.mkdtemp(prefix=prefix, dir=parent_path), parent=parent)

    @property
    def preserve(self) -> bool:
        """Whether the directory stays on disk after nothing references it."""
        return not self._finalizer.alive

    @preserve.setter
    def preserve(self, preserve: bool) -> None:
        if preserve:
            self._finalizer.detach()
            if self.parent is not None:
                self.parent.preserve = True
        elif not self._finalizer.alive:
            self._finalizer = weakref.finalize(
                self, _remove_directory, self.path, self._pid
            )

    def add_reader(self, path: str, storage: "OnDiskStorage") -> None:
        self.readers.setdefault(path, weakref.WeakSet()).add(storage)

    def is_read(self, path: str) -> bool:
        """Whether a live storage other than its creator may read ``path``."""
        readers = self.readers.get(path)
        if readers is None:
            return False
        if not readers:
            del self.readers[path]
            return False
        return True

    def __copy__(self) -> "StorageDirectory":
        return self

    def __deepcopy__(self, memo: dict) -> "StorageDirectory":
        return self

    def __reduce__(self):
        return _find_directory, (self.path, self._pid)


def _parse_file_name(file_name: str) -> Tuple[str, int]:
    """Split a stored file's name, without ``.npy``, into its key and version.

    A key's first file is ``{key}.npy`` and later ones ``{key}.{n}.npy``. A
    key is ``{branch_name}_{period}`` and a period's string form contains no
    ``.``, so a final ``.`` followed only by digits is a version.
    """
    key, separator, version = file_name.rpartition(".")
    if separator and version.isdigit():
        return key, int(version)
    return file_name, 0


class OnDiskStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors on disk.

    Each storage writes only into its own directory, ``storage_dir``. A
    storage created with ``storage_dir=None`` makes that directory on its
    first write, inside the directory given by :meth:`temporary` (or the
    system temporary directory).

    A clone (:meth:`clone`) reads the files its source had stored when it was
    cloned, and writes into a new directory of its own. A storage overwrites
    a file in place only if it created the file itself, since the last fork,
    the file is still the key's newest, and no live clone, copy or storage
    that restored the directory may read it. Otherwise it writes a new file,
    created exclusively and numbered above every existing file for the key,
    so :meth:`restore` finds each key's latest value.

    Unless ``preserve_storage_dir`` is set, a storage's directory is removed
    once the storage and every storage that may read a file in it are gone.
    """

    def __init__(
        self,
        storage_dir: Optional[str],
        is_eternal: bool = False,
        preserve_storage_dir: bool = False,
    ):
        self._files: Dict[str, str] = {}
        self._enums: Dict[str, type] = {}
        self.is_eternal = is_eternal
        self.storage_dir = storage_dir
        self._preserve_storage_dir = preserve_storage_dir
        self._directory: Optional[StorageDirectory] = None
        self._name = "storage"
        if storage_dir is not None:
            self._directory = StorageDirectory.for_path(storage_dir)
            if preserve_storage_dir:
                self._directory.preserve = True
            self.storage_dir = self._directory.path
            self._name = os.path.basename(self._directory.path)
        # Where ``storage_dir`` is created on the first write, if it is None.
        self._parent_directory: Union[StorageDirectory, str, None] = None
        # Other storages' directories that files in ``_files`` may be in.
        self._read_directories: Tuple[StorageDirectory, ...] = ()
        # The newest file this storage created for each key, and the fork
        # generation it was created in: the only file it may overwrite.
        self._created: Dict[str, Tuple[str, int]] = {}
        # Whether the directory is this storage's own temporary one, which
        # nothing restores.
        self._temporary = False

    @classmethod
    def temporary(
        cls,
        name: str,
        directory: Union[StorageDirectory, str, None] = None,
        is_eternal: bool = False,
    ) -> "OnDiskStorage":
        """Create a storage whose directory is made on its first write.

        The directory is created inside ``directory`` (a
        :class:`StorageDirectory`, which the storage keeps alive, a path, or
        ``None`` for the system temporary directory) with a unique name
        starting with ``name``, and removed once nothing reads it.
        """
        storage = cls(None, is_eternal=is_eternal)
        storage._name = name
        storage._parent_directory = directory
        storage._temporary = True
        return storage

    @property
    def preserve_storage_dir(self) -> bool:
        """Whether this storage's directory stays on disk after it is gone."""
        return self._preserve_storage_dir

    @preserve_storage_dir.setter
    def preserve_storage_dir(self, preserve: bool) -> None:
        self._preserve_storage_dir = preserve
        if self._directory is not None:
            self._directory.preserve = preserve

    def clone(self) -> "OnDiskStorage":
        """Create a storage that starts with this storage's values.

        The clone gets copies of the file and enum mappings, so deleting or
        rewiring entries through one storage does not change the other. It
        reads the same ``.npy`` files, which stay on disk while it may read
        them and are not overwritten while it is alive, and writes into a
        directory of its own, created on its first write next to this
        storage's.
        """
        clone = OnDiskStorage.temporary(
            self._name, self._parent_directory, is_eternal=self.is_eternal
        )
        clone._files = self._files.copy()
        clone._enums = self._enums.copy()
        clone._read_directories = self._read_directories
        if self._directory is not None:
            clone._read_directories += (self._directory,)
        clone._register_as_reader()
        return clone

    def _register_as_reader(self) -> None:
        """Record this storage as a reader of every file it maps."""
        directories = {
            directory.path: directory for directory in self._read_directories
        }
        if self._directory is not None:
            directories[self._directory.path] = self._directory
        for path in self._files.values():
            directory = directories.get(os.path.dirname(path))
            if directory is not None:
                directory.add_reader(path, self)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_files"] = dict(self._files)
        state["_enums"] = dict(self._enums)
        # A copy never overwrites a file the original created.
        state["_created"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._register_as_reader()

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

    def _get_storage_dir(self) -> str:
        if self.storage_dir is None:
            self._directory = StorageDirectory.create(
                prefix=f"{self._name}_", parent=self._parent_directory
            )
            self._directory.preserve = self._preserve_storage_dir
            self.storage_dir = self._directory.path
        return self.storage_dir

    def _newest_version(self, key: str) -> int:
        """The highest version of any file for ``key`` in ``storage_dir``."""
        versions = [
            version
            for file_key, version in (
                _parse_file_name(file_name[: -len(".npy")])
                for file_name in os.listdir(self.storage_dir)
                if file_name.endswith(".npy")
            )
            if file_key == key
        ]
        return max(versions, default=-1)

    def _create_file(self, key: str) -> str:
        """Create an empty file for a new value of ``key`` and return its path.

        The file is ``{key}.npy`` if there is none yet, and otherwise
        ``{key}.{n}.npy`` numbered above every file for the key.
        """
        storage_dir = self._get_storage_dir()
        path = os.path.join(storage_dir, f"{key}.npy")
        if _create_exclusively(path):
            return path
        version = max(self._newest_version(key), 0)
        while True:
            version += 1
            path = os.path.join(storage_dir, f"{key}.{version}.npy")
            if _create_exclusively(path):
                return path

    def _may_overwrite(self, key: str, path: str, generation: int) -> bool:
        if generation != _fork_generation or self._directory.is_read(path):
            return False
        if self._temporary:
            return True
        # In a directory another storage may write and restore, the file
        # must still be the key's newest.
        version = _parse_file_name(os.path.basename(path)[: -len(".npy")])[1]
        return version >= self._newest_version(key)

    def _get_path_to_write(self, key: str) -> str:
        """Return the file to store a new value for ``key`` in."""
        created = self._created.get(key)
        if created is not None and self._may_overwrite(key, *created):
            return created[0]
        path = self._create_file(key)
        self._created[key] = (path, _fork_generation)
        return path

    def put(
        self, value: ArrayLike, period: Period, branch_name: str = "default"
    ) -> None:
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)

        filename = f"{branch_name}_{period}"
        path = self._get_path_to_write(filename)
        if isinstance(value, EnumArray):
            self._enums[path] = value.possible_values
            value = value.view(numpy.ndarray)
        else:
            self._enums.pop(path, None)
        numpy.save(path, value)
        self._files[filename] = path

    def delete(self, period: Period = None, branch_name: str = "default") -> None:
        # Deleting forgets the mapping only. The file stays on disk until the
        # directory is removed, since another storage may still read it.
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

    def get_known_periods(self) -> list:
        return list([periods.period(x.split("_")[1]) for x in self._files.keys()])

    def get_known_branch_periods(self) -> list:
        return [
            (branch_name, periods.period(period))
            for branch_name, period in map(lambda x: x.split("_"), self._files.keys())
        ]

    def restore(self) -> None:
        """Map each key to its highest-numbered, and so latest, file in
        ``storage_dir``.

        The storage becomes a reader of those files, and stores new values in
        new files, since other storages may read the restored ones.
        """
        self._files = files = {}
        self._created = {}
        if self.storage_dir is None:
            return
        versions = {}
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".npy"):
                continue
            key, version = _parse_file_name(filename[: -len(".npy")])
            if version >= versions.get(key, -1):
                versions[key] = version
                files[key] = os.path.join(self.storage_dir, filename)
        self._register_as_reader()
