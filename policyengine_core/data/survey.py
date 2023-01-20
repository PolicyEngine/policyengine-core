"""
This module contains the main definition for the in-memory representation of survey datasets.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any


class Survey:
    """
    A `Survey` instance is an in-memory representation of a weighted survey.
    """

    name: str = None
    """A Python-safe name for the survey."""

    label: str = None
    """The label of the survey."""

    data_file: Path = None
    """The folder where the survey data is stored."""

    public_data_url: str = None
    """The URL where the public data is stored, if it is publicly available."""

    tables: Dict[str, pd.DataFrame] = None

    def __init__(self):
        if self.data_file is None:
            self.data_file = (
                Path(__file__).parent / "data" / f"{self.name}.pkl"
            )
        if self.data_file.exists():
            self.tables = self.load_all()

    def generate(self):
        """Generate the survey data."""
        raise NotImplementedError(
            f"Survey {self.name} does not have a `generate` method."
        )

    def load(self, table_name: str) -> pd.DataFrame:
        """Load a table from disk."""
        if self.tables is not None:
            return self.tables[table_name]
        tables = self.load_all()
        return tables[table_name]

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all tables from disk."""
        if self.tables is not None:
            return self.tables
        with open(self.data_file, "rb") as f:
            tables: Dict[str, pd.DataFrame] = pickle.load(f)
        return tables

    def save(self, tables: Dict[str, pd.DataFrame]):
        if not self.data_file.parent.exists():
            self.data_file.parent.mkdir(parents=True)
        with open(self.data_file, "wb") as f:
            pickle.dump(tables, f)

    def __getattr__(self, __name: str) -> Any:
        return self.load(__name)

    @property
    def exists(self) -> bool:
        return self.data_file.exists()

    def __repr__(self) -> str:
        return f"Survey({self.name!r}, {self.label!r})"
