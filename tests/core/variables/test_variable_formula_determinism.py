"""Registration-time enforcement of formula determinism (policyengine-core#518).

A variable whose formula references randomness must be rejected when it is
*registered* with a tax-benefit system -- i.e. when it is instantiated by
``add_variables`` -- so a bad formula fails fast at load, with the variable
named, instead of intermittently at calculation time. A deterministic variable
must still register and compute normally.

These integration tests prove the wiring into ``Variable.__init__`` (the pure
detector is unit-tested in ``test_formula_randomness``).
"""

import random
from numpy.random import random as _bare_random_import

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.reforms import Reform
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable
from policyengine_core.variables.formula_randomness import (
    NonDeterministicFormulaError,
)

# A generator hoisted to module scope, built before any formula runs.
_PREBUILT_RNG = np.random.default_rng(0)

PERIOD = "2013-01"


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
    label = "formula that builds a seeded generator inside the formula"

    def formula(person, period):
        return np.random.default_rng(0).random(person.count)


class uses_prebuilt_generator(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that uses a module-scope generator"

    def formula(person, period):
        return _PREBUILT_RNG.random(person.count)


class uses_bare_imported_function(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "formula that uses a by-name imported drawing function"

    def formula(person, period):
        return _bare_random_import()


class deterministic(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.MONTH
    label = "deterministic formula"

    def formula(person, period):
        return person.count


class uses_random_in_dated_formula(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "clean base formula but randomness in a later dated formula"

    def formula(person, period):
        return person.count

    def formula_2020(person, period):
        return np.random.random(person.count)


RANDOMNESS_VARIABLES = [
    uses_numpy_random,
    uses_stdlib_random,
    uses_seeded_generator,
    uses_prebuilt_generator,
    uses_bare_imported_function,
]


def _register(*variable_classes):
    system = CountryTaxBenefitSystem()
    system.add_variables(*variable_classes)
    return system


@pytest.mark.parametrize(
    "variable_class", RANDOMNESS_VARIABLES, ids=lambda c: c.__name__
)
def test_randomness_variable_rejected_at_registration(variable_class):
    with pytest.raises(NonDeterministicFormulaError, match=variable_class.__name__):
        _register(variable_class)


def test_deterministic_variable_registers_and_computes():
    system = _register(deterministic)
    simulation = SimulationBuilder().build_default_simulation(system)
    result = simulation.calculate("deterministic", PERIOD)
    assert (result == 1).all()


def test_randomness_in_a_dated_formula_is_rejected():
    # check_formula_determinism scans every dated formula, not just the first.
    with pytest.raises(
        NonDeterministicFormulaError, match="uses_random_in_dated_formula"
    ):
        _register(uses_random_in_dated_formula)


def test_reform_update_variable_with_randomness_is_rejected():
    # The reform / update_variable path re-runs Variable.__init__, so a reform
    # that introduces a randomness-using formula is rejected when applied.
    class disposable_income(Variable):
        def formula(person, period):
            return np.random.random(person.count)

    class reform(Reform):
        def apply(self):
            self.update_variable(disposable_income)

    with pytest.raises(NonDeterministicFormulaError, match="disposable_income"):
        reform(CountryTaxBenefitSystem())
