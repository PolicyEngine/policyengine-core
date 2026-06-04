"""A variable formula must not invoke a random number generator.

Rules-engine formulas have to be deterministic functions of their inputs, so
calling ``numpy.random`` or the stdlib ``random`` module inside a formula raises
:class:`NonDeterministicFormulaError`. These tests pin that behaviour and verify
the guard restores the randomness namespaces afterwards.
"""

import random
from numpy.random import random as _bare_random_import

import numpy as np
import pytest

# A generator hoisted to module scope, built before any formula runs. Its bound
# methods cannot be patched (numpy generator classes are immutable C types), so
# using it inside a formula is a known, pinned gap in the guard.
_PREBUILT_RNG = np.random.default_rng(0)

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


class raises_value_error(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that raises a non-RNG exception"

    def formula(person, period):
        raise ValueError("boom")


class uses_prebuilt_generator(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that uses a module-scope generator (known gap)"

    def formula(person, period):
        return _PREBUILT_RNG.random(person.count)


class uses_bare_imported_function(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that uses a by-name imported drawing function (known gap)"

    def formula(person, period):
        return _bare_random_import()


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


def test_randomness_restored_after_non_rng_exception_in_formula():
    # If a formula raises a normal exception, _run_formula's guarded block must
    # still restore the randomness namespaces (no leak of the patched state).
    simulation = _simulation_with(raises_value_error)
    with pytest.raises(ValueError, match="boom"):
        simulation.calculate("raises_value_error", PERIOD)
    assert isinstance(float(np.random.random()), float)
    assert isinstance(random.random(), float)


def test_constructing_generator_inside_formula_is_caught():
    # Building a generator inside the formula hits the patched np.random
    # constructor, so even this seeded form raises (covered by
    # uses_seeded_generator too; kept explicit for the boundary).
    simulation = _simulation_with(uses_seeded_generator)
    with pytest.raises(NonDeterministicFormulaError):
        simulation.calculate("uses_seeded_generator", PERIOD)


def test_prebuilt_generator_is_a_known_gap():
    # Documented limitation: a generator built before the formula runs cannot be
    # intercepted (numpy generator classes are immutable). Pin the behaviour so a
    # future change to it is noticed.
    simulation = _simulation_with(uses_prebuilt_generator)
    result = simulation.calculate("uses_prebuilt_generator", PERIOD)
    assert result is not None


def test_by_name_imported_function_is_a_known_gap():
    # Documented limitation: a drawing function imported by name before the guard
    # installs is not intercepted. Pin the behaviour.
    simulation = _simulation_with(uses_bare_imported_function)
    result = simulation.calculate("uses_bare_imported_function", PERIOD)
    assert result is not None


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
