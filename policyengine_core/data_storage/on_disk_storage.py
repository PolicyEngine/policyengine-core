import os
import shutil
from typing import Dict, Optional, Set

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.data_storage.store_history import next_sequence_number
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
        # As in ``InMemoryStorage``: when each file was stored, and which were
        # stored as inputs.
        self._sequence_numbers: Dict[str, int] = {}
        self._input_keys: Set[str] = set()
        self.is_eternal = is_eternal
        self.preserve_storage_dir = preserve_storage_dir
        self.storage_dir = storage_dir

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
        clone._sequence_numbers = dict(self._sequence_numbers)
        clone._input_keys = set(self._input_keys)
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

    def put(
        self,
        value: ArrayLike,
        period: Period,
        branch_name: str = "default",
        sequence_number: Optional[int] = None,
        is_input: bool = False,
    ) -> None:
        """Store ``value`` for ``period`` on ``branch_name``.

        ``sequence_number`` and ``is_input`` are as in
        :meth:`InMemoryStorage.put`.
        """
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
        self._sequence_numbers[filename] = (
            next_sequence_number() if sequence_number is None else sequence_number
        )
        if is_input:
            self._input_keys.add(filename)
        else:
            self._input_keys.discard(filename)

    def drop_computed(self, since: Optional[int] = None) -> int:
        """Forget stored values that are not inputs, and return how many.

        As :meth:`InMemoryStorage.drop_computed`. The files stay on disk
        (other views of this directory may still use them) until the storage
        directory is removed.
        """
        dropped = [
            key
            for key in self._files
            if key not in self._input_keys
            and (since is None or self._sequence_numbers.get(key, since) >= since)
        ]
        for key in dropped:
            del self._files[key]
            self._sequence_numbers.pop(key, None)
        return len(dropped)

    def _forget_deleted_keys(self) -> None:
        self._sequence_numbers = {
            key: number
            for key, number in self._sequence_numbers.items()
            if key in self._files
        }
        self._input_keys.intersection_update(self._files)

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
            self._forget_deleted_keys()
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
            self._forget_deleted_keys()

    def get_known_periods(self) -> list:
        return list([periods.period(x.split("_")[1]) for x in self._files.keys()])

    def get_known_branch_periods(self) -> list:
        return [
            (branch_name, periods.period(period))
            for branch_name, period in map(lambda x: x.split("_"), self._files.keys())
        ]

    def restore(self) -> None:
        self._files = files = {}
        self._sequence_numbers = {}
        self._input_keys = set()
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
