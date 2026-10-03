import os
import shutil

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.enums import EnumArray
from policyengine_core.periods import Period


class OnDiskStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors on disk
    """

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
        self.is_eternal = is_eternal
        self.preserve_storage_dir = preserve_storage_dir
        self.storage_dir = storage_dir

    def __setstate__(self, state: dict) -> None:
        # A storage pickled before derived marks existed has none: its values
        # count as inputs.
        state.setdefault("_derived", set())
        self.__dict__.update(state)

    def clone(self) -> "OnDiskStorage":
        """Create a private metadata view over this storage directory.

        The file and enum mappings are copied so deleting or rewiring entries
        through the clone does not mutate the source storage. The underlying
        ``.npy`` files remain shared: writing the same ``{branch}_{period}``
        key from two views targets the same path and can overwrite the file.
        Clones retain the original cleanup owner so the shared directory stays
        alive, but never own cleanup themselves.
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
        path = os.path.join(self.storage_dir, filename) + ".npy"
        if isinstance(value, EnumArray):
            self._enums[path] = value.possible_values
            value = value.view(numpy.ndarray)
        numpy.save(path, value)
        self._files[filename] = path
        if derived:
            self._derived.add(filename)
        else:
            self._derived.discard(filename)

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

    def __del__(self) -> None:
        if self.preserve_storage_dir:
            return
        shutil.rmtree(self.storage_dir)  # Remove the holder temporary files
        # If the simulation temporary directory is empty, remove it
        parent_dir = os.path.abspath(os.path.join(self.storage_dir, os.pardir))
        if not os.listdir(parent_dir):
            shutil.rmtree(parent_dir)
