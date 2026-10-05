import os
import shutil
import tempfile
import weakref

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.enums import EnumArray
from policyengine_core.periods import Period

# Subdirectory of a storage directory for the files ``put`` writes when a
# key's usual file is shared with a clone (see ``OnDiskStorage._path_to_write``).
# ``restore`` reads only the files directly in the directory, so it never
# reads these.
REPLACEMENTS_DIR = "replaced"

# The storage that removes each storage directory when collected (one created
# without ``preserve_storage_dir``), by path, among those alive in this
# process: what a copy of a storage keeps alive (see ``__setstate__``).
_DIRECTORY_OWNERS: "weakref.WeakValueDictionary[str, OnDiskStorage]" = (
    weakref.WeakValueDictionary()
)


class OnDiskStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors on disk
    """

    # Defaults for ``__del__``, which also runs on an instance whose
    # ``__init__`` did not: such an instance removes nothing.
    preserve_storage_dir = True
    _creator_pid = None
    # The simulation's temporary directory this storage's directory is in, if
    # any, kept alive while this storage (or a clone or copy of it) reads
    # files in it (see ``Holder.create_disk_storage``).
    _parent_directory = None

    def __init__(
        self,
        storage_dir: str,
        is_eternal: bool = False,
        preserve_storage_dir: bool = False,
    ):
        self._files = {}
        self._enums = {}
        # File keys stored with ``put(..., derived=True)``; see
        # ``InMemoryStorage``.
        self._derived = set()
        # For each key, the file this storage last wrote for it, while it has
        # not shared that file since: the only files of its family ``put``
        # writes over (see ``_path_to_write``). Kept when the key is deleted,
        # so that writing it again reuses the file.
        self._own_paths = {}
        # Paths of every file this storage, the storage it was cloned from or
        # any other clone of either wrote or stored when cloned: one set,
        # shared by all of them.
        self._family_files = set()
        self.is_eternal = is_eternal
        self.preserve_storage_dir = preserve_storage_dir
        self.storage_dir = storage_dir
        self._creator_pid = os.getpid()
        if not preserve_storage_dir:
            _DIRECTORY_OWNERS[storage_dir] = self

    def __getstate__(self) -> dict:
        # Whatever this state is read into reads the same files, so from now
        # on this storage writes over none of them, as after ``clone``.
        self._own_paths = {}
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        # A storage pickled before derived marks existed has none: its values
        # count as inputs.
        state.setdefault("_derived", set())
        # Nor did it record the files it wrote: it writes over none it stores.
        state.setdefault("_own_paths", {})
        if "_family_files" not in state:
            state["_family_files"] = set(state.get("_files", {}).values())
        # A copy reads the files of the storage it was copied from, so it
        # never removes their directory, which that storage, its clones or
        # another copy may still read; it keeps alive instead the storage in
        # this process that removes it, as a clone does. Elsewhere (another
        # process) nothing here removes the directory.
        state["preserve_storage_dir"] = True
        owner = _DIRECTORY_OWNERS.get(state.get("storage_dir"))
        if owner is not None:
            state["_storage_dir_owner"] = owner
        self.__dict__.update(state)

    def clone(self) -> "OnDiskStorage":
        """Create a private metadata view over this storage directory.

        The file and enum mappings are copied so deleting or rewiring entries
        through the clone does not mutate the source storage. The underlying
        ``.npy`` files are shared, so neither storage writes over them: a later
        ``put`` of one of their keys, through either storage, writes a new
        file (see ``_path_to_write``). Clones retain the original cleanup owner
        so the shared directory stays alive, but never own cleanup themselves.
        """
        clone = OnDiskStorage(
            self.storage_dir,
            is_eternal=self.is_eternal,
            preserve_storage_dir=True,
        )
        clone._files = self._files.copy()
        clone._enums = self._enums.copy()
        clone._derived = set(self._derived)
        clone._storage_dir_owner = getattr(self, "_storage_dir_owner", self)
        clone._parent_directory = self._parent_directory
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
        """
        own = self._own_paths.get(filename)
        if own is not None:
            return own
        path = os.path.join(self.storage_dir, filename) + ".npy"
        if path not in self._family_files:
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

    def __del__(self, _rmtree=shutil.rmtree, _getpid=os.getpid) -> None:
        # (The defaults keep the two functions reachable while the
        # interpreter shuts down.) Only in the process that created this
        # storage: a process forked from it has a copy, and the directory is
        # still the creator's.
        if self.preserve_storage_dir or self._creator_pid != _getpid():
            return
        # Remove this storage's directory, with the files of its clones (each
        # keeps this storage alive). Not the directory containing it: that is
        # a simulation's, removed with the simulation (see
        # ``TemporaryStorageDirectory``), or one the caller chose.
        _rmtree(self.storage_dir, ignore_errors=True)
