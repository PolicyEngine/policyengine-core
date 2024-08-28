from pathlib import Path
from typing import Dict, Union, List
import h5py
import numpy as np
import pandas as pd
import shutil
import requests
import os


class Dataset:
    """The `Dataset` class is a base class for datasets used directly or indirectly for microsimulation models.
    A dataset defines a generation function to create it from other data, and this class provides common features
    like storage, metadata and loading."""

    name: str = None
    """The name of the dataset. This is used to generate filenames and is used as the key in the `datasets` dictionary."""
    label: str = None
    """The label of the dataset. This is used for logging and is used as the key in the `datasets` dictionary."""
    data_format: str = None
    """The format of the dataset. This can be either `Dataset.ARRAYS`, `Dataset.TIME_PERIOD_ARRAYS` or `Dataset.TABLES`. If `Dataset.ARRAYS`, the dataset is stored as a collection of arrays. If `Dataset.TIME_PERIOD_ARRAYS`, the dataset is stored as a collection of arrays, with one array per time period. If `Dataset.TABLES`, the dataset is stored as a collection of tables (DataFrames)."""
    file_path: Path = None
    """The path to the dataset file. This is used to load the dataset from a file."""
    time_period: str = None
    """The time period of the dataset. This is used to automatically enter the values in the correct time period if the data type is `Dataset.ARRAYS`."""
    url: str = None
    """The URL to download the dataset from. This is used to download the dataset if it does not exist."""

    # Data formats
    TABLES = "tables"
    ARRAYS = "arrays"
    TIME_PERIOD_ARRAYS = "time_period_arrays"
    FLAT_FILE = "flat_file"

    _table_cache: Dict[str, pd.DataFrame] = None

    def __init__(self, require: bool = False):
        # Setup dataset
        if self.file_path is None:
            raise ValueError(
                "Dataset file_path must be specified in the dataset class definition."
            )
        elif isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        assert (
            self.name
        ), "You tried to instantiate a Dataset object, but no name has been provided."
        assert (
            self.label
        ), "You tried to instantiate a Dataset object, but no label has been provided."

        assert self.data_format in [
            Dataset.TABLES,
            Dataset.ARRAYS,
            Dataset.TIME_PERIOD_ARRAYS,
            Dataset.FLAT_FILE,
        ], f"You tried to instantiate a Dataset object, but your data_format attribute is invalid ({self.data_format})."

        self._table_cache = {}

        if not self.exists and require:
            if self.url is not None:
                self.download()
            else:
                self.generate()

    def load(
        self, key: str = None, mode: str = "r"
    ) -> Union[h5py.File, np.array, pd.DataFrame, pd.HDFStore]:
        """Loads the dataset for a given year, returning a H5 file reader. You can then access the
        dataset like a dictionary (e.g.e Dataset.load(2022)["variable"]).

        Args:
            key (str, optional): The key to load. Defaults to None.
            mode (str, optional): The mode to open the file with. Defaults to "r".

        Returns:
            Union[h5py.File, np.array, pd.DataFrame, pd.HDFStore]: The dataset.
        """
        file = self.file_path
        if self.data_format in (Dataset.ARRAYS, Dataset.TIME_PERIOD_ARRAYS):
            if key is None:
                # If no key provided, return the basic H5 reader.
                return h5py.File(file, mode=mode)
            else:
                # If key provided, return only the values requested.
                with h5py.File(file, mode=mode) as f:
                    values = np.array(f[key])
                return values
        elif self.data_format == Dataset.TABLES:
            if key is None:
                # Non-openfisca datasets are assumed to be of the format (table name: [table], ...).
                return pd.HDFStore(file)
            else:
                if key in self._table_cache:
                    return self._table_cache[key]
                # If a table name is provided, return that table.
                with pd.HDFStore(file) as f:
                    values = f[key]
                self._table_cache[key] = values
                return values
        elif self.data_format == Dataset.FLAT_FILE:
            if key is None:
                return pd.read_csv(file)
            else:
                raise ValueError(
                    "You tried to load a key from a flat file dataset, but flat file datasets do not support keys."
                )
        else:
            raise ValueError(
                f"Invalid data format {self.data_format} for dataset {self.label}."
            )

    def save(self, key: str, values: Union[np.array, pd.DataFrame]):
        """Overwrites the values for `key` with `values`.

        Args:
            key (str): The key to save.
            values (Union[np.array, pd.DataFrame]): The values to save.
        """
        file = self.file_path
        if self.data_format in (Dataset.ARRAYS, Dataset.TIME_PERIOD_ARRAYS):
            with h5py.File(file, "a") as f:
                # Overwrite if existing
                if key in f:
                    del f[key]
                f.create_dataset(key, data=values)
        elif self.data_format == Dataset.TABLES:
            with pd.HDFStore(file, "a") as f:
                f.put(key, values)
            self._table_cache = {}
        elif self.data_format == Dataset.FLAT_FILE:
            values.to_csv(file, index=False)
        else:
            raise ValueError(
                f"Invalid data format {self.data_format} for dataset {self.label}."
            )

    def save_dataset(self, data, file_path: str = None) -> None:
        """Writes a complete dataset to disk.

        Args:
            data: The data to save.

        >>> example_data: Dict[str, Dict[str, Sequence]] = {
        ...     "employment_income": {
        ...         "2022": np.array([25000, 25000, 30000, 30000]),
        ...     },
        ... }
        >>> example_data["employment_income"]["2022"] = [25000, 25000, 30000, 30000]
        """
        if file_path is not None:
            file = Path(file_path)
        elif not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)
        file = self.file_path
        if self.data_format == Dataset.TABLES:
            for table_name, dataframe in data.items():
                self.save(table_name, dataframe)
        elif self.data_format == Dataset.TIME_PERIOD_ARRAYS:
            with h5py.File(file, "w") as f:
                for variable, values in data.items():
                    for time_period, value in values.items():
                        key = f"{variable}/{time_period}"
                        # Overwrite if existing
                        if key in f:
                            del f[key]
                        try:
                            f.create_dataset(key, data=value)
                        except:
                            raise ValueError(
                                f"Could not save {key} to {file}. The value is {value}."
                            )
        elif self.data_format == Dataset.ARRAYS:
            with h5py.File(file, "a" if file.exists() else "w") as f:
                for variable, value in data.items():
                    # Overwrite if existing
                    if variable in f:
                        del f[variable]
                    try:
                        f.create_dataset(variable, data=value)
                    except:
                        raise ValueError(
                            f"Could not save {variable} to {file}. The value is {value}."
                        )
        elif self.data_format == Dataset.FLAT_FILE:
            data.to_csv(file, index=False)

    def load_dataset(
        self,
    ):
        """Loads a complete dataset from disk.

        Returns:
            Dict[str, Dict[str, Sequence]]: The dataset.
        """
        file = self.file_path
        if self.data_format == Dataset.TABLES:
            with pd.HDFStore(file) as f:
                data = {table_name: f[table_name] for table_name in f.keys()}
        elif self.data_format == Dataset.TIME_PERIOD_ARRAYS:
            with h5py.File(file, "r") as f:
                data = {}
                for variable in f.keys():
                    data[variable] = {}
                    for time_period in f[variable].keys():
                        key = f"{variable}/{time_period}"
                        data[variable][time_period] = np.array(f[key])
        elif self.data_format == Dataset.ARRAYS:
            with h5py.File(file, "r") as f:
                data = {
                    variable: np.array(f[variable]) for variable in f.keys()
                }
        return data

    def generate(self):
        """Generates the dataset for a given year (all datasets should implement this method).

        Raises:
            NotImplementedError: If the function has not been overriden.
        """
        raise NotImplementedError(
            f"You tried to generate the dataset for {self.label}, but no dataset generation implementation has been provided for {self.label}."
        )

    @property
    def exists(self) -> bool:
        """Checks whether the dataset exists.

        Returns:
            bool: Whether the dataset exists.
        """
        return self.file_path.exists()

    @property
    def variables(self) -> List[str]:
        """Returns the variables in the dataset.

        Returns:
            List[str]: The variables in the dataset.
        """
        if self.data_format == Dataset.TABLES:
            with pd.HDFStore(self.file_path) as f:
                return list(f.keys())
        elif self.data_format in (Dataset.ARRAYS, Dataset.TIME_PERIOD_ARRAYS):
            with h5py.File(self.file_path, "r") as f:
                return list(f.keys())
        elif self.data_format == Dataset.FLAT_FILE:
            return pd.read_csv(self.file_path, nrows=0).columns.tolist()
        else:
            raise ValueError(
                f"Invalid data format {self.data_format} for dataset {self.label}."
            )

    def __getattr__(self, name):
        """Allows the dataset to be accessed like a dictionary.

        Args:
            name (str): The key to access.

        Returns:
            Union[np.array, pd.DataFrame]: The dataset.
        """
        return self.load(name)

    def store_file(self, file_path: str):
        """Moves a file to the dataset's file path.

        Args:
            file_path (str): The file path to move.
        """

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        shutil.move(file_path, self.file_path)

    def download(self, url: str = None):
        """Downloads a file to the dataset's file path.

        Args:
            url (str): The url to download.
        """

        if url is None:
            url = self.url

        if "POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN" not in os.environ:
            auth_headers = {}
        else:
            auth_headers = {
                "Authorization": f"token {os.environ['POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN']}",
            }

        # "release://" is a special protocol for downloading from GitHub releases
        # e.g. release://policyengine/policyengine-us/cps-2023/cps_2023.h5
        # release://org/repo/release_tag/file_path
        # Use the GitHub API to get the download URL for the release asset

        if url.startswith("release://"):
            org, repo, release_tag, file_path = url.split("/")[2:]
            url = f"https://api.github.com/repos/{org}/{repo}/releases/tags/{release_tag}"
            response = requests.get(url, headers=auth_headers)
            if response.status_code != 200:
                raise ValueError(
                    f"Invalid response code {response.status_code} for url {url}."
                )
            assets = response.json()["assets"]
            for asset in assets:
                if asset["name"] == file_path:
                    url = asset["url"]
                    break
            else:
                raise ValueError(
                    f"File {file_path} not found in release {release_tag} of {org}/{repo}."
                )
        else:
            url = url

        response = requests.get(
            url,
            headers={
                "Accept": "application/octet-stream",
                **auth_headers,
            },
        )

        if response.status_code != 200:
            raise ValueError(
                f"Invalid response code {response.status_code} for url {url}."
            )

        with open(self.file_path, "wb") as f:
            f.write(response.content)

    def remove(self):
        """Removes the dataset from disk."""
        if self.exists:
            self.file_path.unlink()

    @staticmethod
    def from_file(file_path: str, time_period: str = None):
        """Creates a dataset from a file.

        Args:
            file_path (str): The file path to create the dataset from.

        Returns:
            Dataset: The dataset.
        """
        file_path = Path(file_path)
        dataset = type(
            "Dataset",
            (Dataset,),
            {
                "name": file_path.stem,
                "label": file_path.stem,
                "data_format": Dataset.FLAT_FILE,
                "file_path": file_path,
                "time_period": time_period,
            },
        )()

        return dataset

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame, time_period: str = None):
        """Creates a dataset from a DataFrame.

        Returns:
            Dataset: The dataset.
        """
        dataset = type(
            "Dataset",
            (Dataset,),
            {
                "name": "dataframe",
                "label": "DataFrame",
                "data_format": Dataset.FLAT_FILE,
                "file_path": "dataframe",
                "time_period": time_period,
                "load": lambda self: dataframe,
            },
        )()

        return dataset
