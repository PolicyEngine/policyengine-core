import os
import shutil
import tempfile
import weakref
from typing import Dict, Optional, Set, Tuple, Union

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.enums import EnumArray
from policyengine_core.periods import Period


def _remove_directory(path: str, pid: int) -> None:
    # A forked child inherits the finalizer, but the directory belongs to the
    # process that created it.
    if os.getpid() == pid:
        shutil.rmtree(path, ignore_errors=True)


class StorageDirectory:
    """A directory removed, with its contents, once nothing references it.

    Whatever may read a file in the directory holds a reference to this
    object, so the directory stays on disk exactly as long as one of them is
    alive. A directory created inside another holds that one as ``parent``.
    """

    def __init__(self, path: str, parent: Optional["StorageDirectory"] = None):
        self.path = path
        self.parent = parent
        self._pid = os.getpid()
        self._finalizer = weakref.finalize(self, _remove_directory, path, self._pid)

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
        elif not self._finalizer.alive:
            self._finalizer = weakref.finalize(
                self, _remove_directory, self.path, self._pid
            )


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
    cloned, and writes into a new directory of its own, so neither storage
    can change or remove a value the other reads. Once a file has been
    shared with a clone, storing its key again writes a new file instead of
    overwriting it.

    Unless ``preserve_storage_dir`` is set, a storage's directory is removed
    once the storage and every clone that may read a file in it are gone.
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
            self._directory = StorageDirectory(storage_dir)
            self._directory.preserve = preserve_storage_dir
            self._name = os.path.basename(os.path.normpath(storage_dir))
        # Where ``storage_dir`` is created on the first write, if it is None.
        self._parent_directory: Union[StorageDirectory, str, None] = None
        # Other storages' directories that files in ``_files`` may be in.
        self._read_directories: Tuple[StorageDirectory, ...] = ()
        # Files this storage wrote or restored, and those of them a clone
        # may read, which must not be overwritten.
        self._written_files: Set[str] = set()
        self._shared_files: Set[str] = set()
        # The last number given to a ``{key}.{n}.npy`` file. A plain integer,
        # not ``itertools.count``, so storages still copy and pickle on Python 3.14.
        self._last_version = 0

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
        them, and writes into a directory of its own, created on its first
        write next to this storage's. From now on, this storage writes a new
        file for a key rather than overwrite one the clone reads.
        """
        clone = OnDiskStorage.temporary(
            self._name, self._parent_directory, is_eternal=self.is_eternal
        )
        clone._files = self._files.copy()
        clone._enums = self._enums.copy()
        clone._read_directories = self._read_directories
        if self._directory is not None:
            clone._read_directories += (self._directory,)
        self._shared_files.update(self._files.values())
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

    def _get_storage_dir(self) -> str:
        if self.storage_dir is None:
            self._directory = StorageDirectory.create(
                prefix=f"{self._name}_", parent=self._parent_directory
            )
            self._directory.preserve = self._preserve_storage_dir
            self.storage_dir = self._directory.path
        return self.storage_dir

    def _get_path_to_write(self, key: str) -> str:
        """Return the file to store a new value for ``key`` in.

        That is the key's current file or ``{key}.npy`` in this storage's
        directory, unless a clone may read it; then a new numbered file.
        """
        current = self._files.get(key)
        if current in self._written_files and current not in self._shared_files:
            return current
        storage_dir = self._get_storage_dir()
        path = os.path.join(storage_dir, f"{key}.npy")
        while path in self._shared_files:
            self._last_version += 1
            path = os.path.join(storage_dir, f"{key}.{self._last_version}.npy")
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
        self._written_files.add(path)
        self._files[filename] = path

    def delete(self, period: Period = None, branch_name: str = "default") -> None:
        # Deleting forgets the mapping only. The file stays on disk until the
        # directory is removed, since a clone may still read it.
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
        """Map each key to its latest file in ``storage_dir``.

        Other storages, including ones in other processes, may read the
        restored files, so storing one of their keys again writes a new file.
        """
        self._files = files = {}
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
        self._written_files.update(files.values())
        self._shared_files.update(files.values())
