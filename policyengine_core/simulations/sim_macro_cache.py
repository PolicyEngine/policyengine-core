import shutil
from pathlib import Path
import h5py
from numpy.typing import ArrayLike


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SimulationMacroCache(metaclass=Singleton):
    def __init__(self):
        self.core_version = None
        self.country_version = None
        self.cache_file_path = None

    def set_version(self, core_version: str, country_version: {}):
        self.core_version = core_version
        self.country_version = country_version

    def set_cache_path(self, parent_path: Path, dataset_name: str):
        self.cache_file_path = parent_path / f"{dataset_name}_variable_cache"

    def get_cache_value(self, version: str, country_version: {}, cache_file_path: Path):
        with h5py.File(cache_file_path, "r") as f:
            if "metadata:core_version" in f and "metadata:country_version" in f:
                # Validate version is correct, otherwise flush the cache
                if f["metadata:core_version"][()] != version or f["metadata:country_version"][()] != country_version:
                    self.clear_cache(cache_file_path)
                    return None
            else:
                self.clear_cache(cache_file_path)
                return None
            return f["values"][()]

    def set_cache_value(self, core_version: str, country_version: {}, cache_file_path: Path, value: ArrayLike):
        self.set_version(core_version, country_version)
        with h5py.File(cache_file_path, "w") as f:
            f.create_dataset("values", data=value)
            f.create_dataset("metadata:core_version", data=self.core_version)
            f.create_dataset("metadata:country_version", data=self.country_version)
        return "cache set successfully"

    def clear_cache(self, cache_file_path: Path):
        shutil.rmtree(cache_file_path)
