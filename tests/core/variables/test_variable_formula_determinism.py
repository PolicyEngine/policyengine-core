"""Registration-time enforcement of formula determinism (policyengine-core#518).

A variable whose formula references randomness must be rejected when it is
*registered* with a tax-benefit system -- i.e. when it is instantiated by
``add_variables`` -- so a bad formula fails fast at load, with the variable
named, instead of intermittently at calculation time. A deterministic variable
must still register and compute normally.

These integration tests prove the wiring into ``Variable.__init__`` (the pure
detector is unit-tested in ``test_formula_randomness``). Until that wiring lands
they are strict xfails.
"""

import random
from numpy.random import random as _bare_random_import

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
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


@pytest.mark.xfail(
    strict=True,
    reason="#518 static check not yet wired into Variable.__init__",
)
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
