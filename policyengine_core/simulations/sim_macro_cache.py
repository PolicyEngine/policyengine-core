import shutil
from pathlib import Path
import h5py
from numpy.typing import ArrayLike
import importlib.metadata

from policyengine_core.taxbenefitsystems import TaxBenefitSystem


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class SimulationMacroCache(metaclass=Singleton):
    def __init__(self, tax_benefit_system: TaxBenefitSystem):
        self.core_version = importlib.metadata.version("policyengine-core")
        self.country_package_metadata = (
            tax_benefit_system.get_package_metadata()
        )
        self.country_version = self.country_package_metadata["version"]
        self.cache_folder_path = None
        self.cache_file_path = None

    def set_cache_path(
        self,
        parent_path: [Path, str],
        dataset_name: str,
        variable_name: str,
        period: str,
        branch_name: str,
    ):
        storage_folder = Path(parent_path) / f"{dataset_name}_variable_cache"
        self.cache_folder_path = storage_folder
        storage_folder.mkdir(exist_ok=True)
        self.cache_file_path = (
            storage_folder / f"{variable_name}_{period}_{branch_name}.h5"
        )

    def set_cache_value(self, cache_file_path: Path, value: ArrayLike):
        with h5py.File(cache_file_path, "w") as f:
            f.create_dataset(
                "metadata:core_version",
                data=self.core_version,
            )
            f.create_dataset(
                "metadata:country_version",
                data=self.country_version,
            )
            f.create_dataset("values", data=value)

    def get_cache_path(self):
        return self.cache_file_path

    def get_cache_value(self, cache_file_path: Path):
        with h5py.File(cache_file_path, "r") as f:
            # Validate both core version and country package metadata are up-to-date, otherwise flush the cache
            if (
                "metadata:core_version" in f
                and "metadata:country_version" in f
            ):
                if (
                    f["metadata:core_version"][()].decode("utf-8")
                    != self.core_version
                    or f["metadata:country_version"][()].decode("utf-8")
                    != self.country_version
                ):
                    f.close()
                    self.clear_cache(self.cache_folder_path)
                    return None
            else:
                f.close()
                self.clear_cache(self.cache_folder_path)
                return None
            return f["values"][()]

    def clear_cache(self, cache_folder_path: Path):
        shutil.rmtree(cache_folder_path)
