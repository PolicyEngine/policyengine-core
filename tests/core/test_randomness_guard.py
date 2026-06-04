"""A variable formula must not invoke a random number generator.

Rules-engine formulas have to be deterministic functions of their inputs, so
calling ``numpy.random`` or the stdlib ``random`` module inside a formula raises
:class:`NonDeterministicFormulaError`. These tests pin that behaviour and verify
the guard restores the randomness namespaces afterwards.
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


class uses_numpy_random(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that draws from numpy.random"

    def formula(person, period):
        return np.random.random(person.count)


class uses_stdlib_random(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that draws from the random module"

    def formula(person, period):
        return random.random()


class uses_seeded_generator(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that builds a seeded generator"

    def formula(person, period):
        # Seeding does not make randomness acceptable inside a formula.
        return np.random.default_rng(0).random(person.count)


class deterministic(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.MONTH
    label = "deterministic formula"

    def formula(person, period):
        return person.count


def test_numpy_random_in_formula_raises():
    simulation = _simulation_with(uses_numpy_random)
    with pytest.raises(NonDeterministicFormulaError, match="uses_numpy_random"):
        simulation.calculate("uses_numpy_random", PERIOD)


def test_stdlib_random_in_formula_raises():
    simulation = _simulation_with(uses_stdlib_random)
    with pytest.raises(NonDeterministicFormulaError, match="random"):
        simulation.calculate("uses_stdlib_random", PERIOD)


def test_seeded_generator_in_formula_still_raises():
    simulation = _simulation_with(uses_seeded_generator)
    with pytest.raises(NonDeterministicFormulaError):
        simulation.calculate("uses_seeded_generator", PERIOD)


def test_deterministic_formula_is_unaffected():
    simulation = _simulation_with(deterministic)
    result = simulation.calculate("deterministic", PERIOD)
    assert (result == 1).all()


def test_randomness_restored_after_guarded_formula():
    simulation = _simulation_with(uses_numpy_random)
    with pytest.raises(NonDeterministicFormulaError):
        simulation.calculate("uses_numpy_random", PERIOD)
    # Outside any formula, numpy and stdlib randomness work normally again.
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
