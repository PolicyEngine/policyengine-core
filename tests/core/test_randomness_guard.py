"""Runtime behaviour of the (legacy) randomness guard.

Formula-time randomness is now rejected statically, at variable registration
(see ``tests/core/variables/test_variable_formula_determinism.py`` and
``test_formula_randomness.py``). What remains here are the runtime-guard
mechanics that still apply while the guard exists; they are removed when the
guard itself is removed.
"""

import random

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.simulations.randomness_guard import (
    NonDeterministicFormulaError,
    forbid_randomness,
)
from policyengine_core.variables import Variable

PERIOD = "2013-01"


def _simulation_with(*variable_classes):
    system = CountryTaxBenefitSystem()
    system.add_variables(*variable_classes)
    return SimulationBuilder().build_default_simulation(system)


class deterministic(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.MONTH
    label = "deterministic formula"

    def formula(person, period):
        return person.count


class raises_value_error(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that raises a non-RNG exception"

    def formula(person, period):
        raise ValueError("boom")


def test_deterministic_formula_is_unaffected():
    simulation = _simulation_with(deterministic)
    result = simulation.calculate("deterministic", PERIOD)
    assert (result == 1).all()


def test_randomness_restored_after_non_rng_exception_in_formula():
    # If a formula raises a normal exception, _run_formula's guarded block must
    # still restore the randomness namespaces (no leak of the patched state).
    simulation = _simulation_with(raises_value_error)
    with pytest.raises(ValueError, match="boom"):
        simulation.calculate("raises_value_error", PERIOD)
    assert isinstance(float(np.random.random()), float)
    assert isinstance(random.random(), float)


def test_guard_is_reentrant():
    # Entering twice and leaving the inner context must not restore the
    # originals while the outer context is still active.
    with forbid_randomness("outer"):
        with forbid_randomness("inner"):
            with pytest.raises(NonDeterministicFormulaError, match="inner"):
                np.random.random()
        # Still guarded: the outer context owns the patch.
        with pytest.raises(NonDeterministicFormulaError, match="outer"):
            np.random.random()
    # Fully restored once the outermost context exits.
    assert isinstance(float(np.random.random()), float)
