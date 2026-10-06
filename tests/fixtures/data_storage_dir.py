"""Simulations that store every value on disk, for the tests of the folder
they store values in (``Simulation.data_storage_dir``).

Shared by ``tests/core/test_data_storage_dir.py`` (examples) and
``tests/core/test_data_storage_dir_property.py`` (properties).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import warnings

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.enums import Enum
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable
from policyengine_core.warnings import MemoryConfigWarning

PEOPLE = 3
YEARS = ["2014", "2015", "2016"]


class Level(Enum):
    low = "Low"
    middle = "Middle"
    high = "High"


class disk_amount(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    label = "A yearly amount with no formula"


class disk_level(Variable):
    value_type = Enum
    possible_values = Level
    default_value = Level.low
    entity = entities.Person
    definition_period = periods.YEAR
    label = "A yearly level with no formula"


VARIABLES = ("disk_amount", "disk_level")

SYSTEM = CountryTaxBenefitSystem()
SYSTEM.add_variables(disk_amount, disk_level)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", MemoryConfigWarning)
    STORE_ON_DISK = MemoryConfig(max_memory_occupation=0)


def disk_simulation(data_storage_dir: str = None):
    """A simulation of ``PEOPLE`` people that stores every value on disk.

    With ``data_storage_dir``, the simulation stores them in that folder.
    """
    simulation = SimulationBuilder().build_default_simulation(SYSTEM, count=PEOPLE)
    # A system keeps the last simulation built with it alive. These tests
    # decide when simulations are collected.
    SYSTEM.simulation = None
    if data_storage_dir is not None:
        simulation._data_storage_dir = str(data_storage_dir)
    simulation.memory_config = STORE_ON_DISK
    # Holders read the memory configuration when created; none has a value
    # yet, so start again without them.
    for population in simulation.populations.values():
        population._holders = {}
    return simulation


def clone(simulation):
    """``simulation.clone()``, with nothing in the clone keeping
    ``simulation`` alive.

    ``Simulation.clone`` copies the source's ``calc`` and ``df``, bound
    methods of the source (so a clone's ``calc`` calculates on its source, a
    separate bug), and so a clone kept its source alive. These tests decide
    when simulations are collected.
    """
    new = simulation.clone()
    new.calc = new.calculate
    new.df = new.calculate_dataframe
    return new


def values(variable: str, seed: int) -> np.ndarray:
    """Values of ``variable`` for ``PEOPLE`` people, different for each seed
    (each of ``len(Level) ** PEOPLE`` seeds in a row, for ``disk_level``)."""
    if variable == "disk_level":
        digits = seed // len(Level) ** np.arange(PEOPLE)
        return (digits % len(Level)).astype(np.int16)
    return (np.arange(PEOPLE) + seed).astype(np.float32)


def read(simulation, variable: str, period: str):
    """The value stored for ``variable`` at ``period``, read from disk.

    Read through the holder, past the simulation's in-memory caches, so a
    value whose file is gone fails to read.
    """
    holder = simulation.persons._holders[variable]
    return holder.get_array(periods.period(period), simulation.branch_name)


@contextlib.contextmanager
def temporary_folders():
    """Make temporary folders (``tempfile``) in a new, empty folder.

    Yields that folder, which is removed afterwards.
    """
    root = tempfile.mkdtemp(prefix="core-data-storage-dir-test-")
    previous = tempfile.tempdir
    tempfile.tempdir = root
    try:
        yield root
    finally:
        tempfile.tempdir = previous
        shutil.rmtree(root, ignore_errors=True)


def storage_folders(folder: str) -> list:
    """The folders simulations made (``openfisca_*``) in ``folder``."""
    return sorted(name for name in os.listdir(folder) if name.startswith("openfisca_"))
