import os
import shutil

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.enums import EnumArray
from policyengine_core.periods import Period


def _split_key(key: str) -> tuple:
    """Split a ``f"{branch_name}_{period}"`` file key into its two parts.

    Branch names often contain ``_`` (policyengine-us uses ``no_salt`` and
    ``mtr_for_adult_1``) but a period's string form never does, so the key
    splits on its last ``_``.
    """
    branch_name, period = key.rsplit("_", 1)
    return branch_name, period


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
        self, value: ArrayLike, period: Period, branch_name: str = "default"
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

    def delete(self, period: Period = None, branch_name: str = "default") -> None:
        if period is None:
            # Only wipe files belonging to the requested branch (previously
            # this wiped every branch regardless of ``branch_name`` — same
            # class of bug as C2 in InMemoryStorage). Compare the parsed
            # branch name, not a prefix: deleting ``pre_tcja`` must not
            # also wipe ``pre_tcja_ctc``.
            self._files = {
                period_item: value
                for period_item, value in self._files.items()
                if _split_key(period_item)[0] != branch_name
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
        return [period for _, period in self.get_known_branch_periods()]

    def get_known_branch_periods(self) -> list:
        return [
            (branch_name, periods.period(period))
            for branch_name, period in map(_split_key, self._files.keys())
        ]

    def restore(self) -> None:
        self._files = files = {}
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
