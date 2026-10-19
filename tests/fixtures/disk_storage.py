"""Helpers for tests of disk-backed holder storage."""

import contextlib
import sys
from typing import List

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import Simulation, SimulationBuilder


def build_simulation(on_disk: bool = True, count: int = 1) -> Simulation:
    # A tax-benefit system keeps the last simulation built on it, so give
    # each simulation its own for the tests that release one.
    simulation = SimulationBuilder().build_default_simulation(
        CountryTaxBenefitSystem(), count=count
    )
    if on_disk:
        simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    # Holders made before ``memory_config`` is set have no disk storage, so
    # start without any: each holder is created where it is first used.
    for population in simulation.populations.values():
        population._holders = {}
    return simulation


@contextlib.contextmanager
def unraisable_exceptions():
    """Collect exceptions Python can only print, such as those in finalizers."""
    errors: List[str] = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = lambda unraisable: errors.append(repr(unraisable.exc_value))
    try:
        yield errors
    finally:
        sys.unraisablehook = previous_hook
