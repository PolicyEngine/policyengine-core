"""A reform simulation's baseline branch uses the baseline system throughout.

``Simulation.__init__`` builds the baseline branch with ``get_branch``, under
the reform's system, and then gives it the baseline's. The branch's
populations kept the reform system's entities, so they looked variables up in
the reform's system, and its holders kept the reform's ``Variable`` objects:
a variable the reform neutralized read as its default in the baseline too,
and an input variable whose default the reform changed took the reform's
default there. ``subsample``, which rebuilds the branch the same way, did the
same, and also gave the rebuilt branch a ``baseline`` of its own: the branch
it replaced, with the old population.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core.country_template import (
    CountryTaxBenefitSystem,
    Microsimulation,
    Simulation,
)
from policyengine_core.country_template.entities import Person
from policyengine_core.errors import VariableNotFoundError
from policyengine_core.periods import MONTH
from policyengine_core.reforms import Reform
from policyengine_core.simulations import Simulation as CoreSimulation
from policyengine_core.variables import Variable

PERIOD = "2022-01"
BASELINE_RATE = 0.15
SALARY = [4000.0, 2500.0]
SITUATION = {
    "persons": {
        "a": {"salary": {PERIOD: SALARY[0]}},
        "b": {"salary": {PERIOD: SALARY[1]}},
        # No salary given: it takes the variable's default.
        "c": {},
    },
    "households": {"h": {"parents": ["a", "b"], "children": ["c"]}},
}
REFORM_DEFAULT_SALARY = 1000.0


class neutralize_income_tax(Reform):
    def apply(self):
        self.neutralize_variable("income_tax")


class salary(Variable):
    default_value = REFORM_DEFAULT_SALARY


class change_salary_default(Reform):
    def apply(self):
        self.update_variable(salary)


class doubled_salary(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Salary times two"

    def formula(person, period, parameters):
        return person("salary", period) * 2


class add_doubled_salary(Reform):
    def apply(self):
        self.update_variable(doubled_salary)


def _baseline_of_new_simulation(reform):
    return Simulation(situation=SITUATION, reform=reform).baseline


def _baseline_after_subsample(reform):
    simulation = Microsimulation(reform=reform)
    simulation.subsample(n=1, seed="baseline-system", time_period="2022")
    return simulation.baseline


BASELINES = [
    pytest.param(_baseline_of_new_simulation, id="new-simulation"),
    pytest.param(_baseline_after_subsample, id="after-subsample"),
]


def _salary(baseline):
    return np.array(baseline.calculate("salary", PERIOD))


@pytest.mark.parametrize("make_baseline", BASELINES)
def test_a_variable_the_reform_neutralizes_is_calculated_in_the_baseline(
    make_baseline,
):
    baseline = make_baseline(neutralize_income_tax)
    reform = baseline.parent_branch

    np.testing.assert_allclose(
        baseline.calculate("income_tax", PERIOD), _salary(baseline) * BASELINE_RATE
    )
    assert baseline.calculate("income_tax", PERIOD).sum() > 0
    np.testing.assert_array_equal(reform.calculate("income_tax", PERIOD), 0)


def test_an_input_default_the_reform_changes_is_the_baselines_in_the_baseline():
    # No one's salary is given. (Given for some, the situation builder would
    # store the reform's default for the others, as an input.)
    situation = {
        "persons": {"a": {}, "b": {}},
        "households": {"h": {"parents": ["a", "b"]}},
    }
    reform = Simulation(situation=situation, reform=change_salary_default)
    baseline = reform.baseline

    np.testing.assert_array_equal(
        reform.calculate("salary", PERIOD), [REFORM_DEFAULT_SALARY] * 2
    )
    np.testing.assert_array_equal(baseline.calculate("salary", PERIOD), [0, 0])
    np.testing.assert_array_equal(baseline.calculate("income_tax", PERIOD), [0, 0])
    np.testing.assert_allclose(
        reform.calculate("income_tax", PERIOD),
        [REFORM_DEFAULT_SALARY * BASELINE_RATE] * 2,
    )


@pytest.mark.parametrize("make_baseline", BASELINES)
def test_the_baseline_has_nothing_for_a_variable_only_the_reform_has(
    make_baseline,
):
    baseline = make_baseline(add_doubled_salary)
    reform = baseline.parent_branch

    np.testing.assert_array_equal(
        reform.calculate("doubled_salary", PERIOD),
        2 * np.array(reform.calculate("salary", PERIOD)),
    )
    assert "doubled_salary" not in baseline.persons._holders
    assert "doubled_salary" not in baseline.input_variables
    assert all(key[0] != "doubled_salary" for key in baseline._user_input_keys)
    with pytest.raises(VariableNotFoundError):
        baseline.get_holder("doubled_salary")
    with pytest.raises(ValueError, match="does not exist"):
        baseline.calculate("doubled_salary", PERIOD)
    # The reform simulation keeps its own.
    assert "doubled_salary" in reform.persons._holders


@pytest.mark.parametrize("make_baseline", BASELINES)
@pytest.mark.parametrize(
    "reform",
    [neutralize_income_tax, change_salary_default, add_doubled_salary],
    ids=lambda reform: reform.__name__,
)
def test_the_baseline_uses_its_systems_entities_and_variables(make_baseline, reform):
    baseline = make_baseline(reform)
    system = baseline.tax_benefit_system
    entities = {
        entity.key: entity for entity in [system.person_entity, *system.group_entities]
    }

    assert system is not baseline.parent_branch.tax_benefit_system
    for key, population in baseline.populations.items():
        assert population.entity is entities[key]
        assert population.entity._tax_benefit_system is system
        for name, holder in population._holders.items():
            assert holder.variable is system.variables[name]


def test_the_baselines_record_of_inputs_is_its_own():
    reform = Simulation(situation=SITUATION, reform=add_doubled_salary)

    assert reform.baseline.input_variables is not reform.input_variables
    assert reform.baseline._user_input_keys is not reform._user_input_keys
    assert "doubled_salary" in reform.tax_benefit_system.variables


def test_subsample_leaves_the_baseline_without_a_baseline():
    simulation = Microsimulation(reform=neutralize_income_tax)
    assert simulation.baseline.baseline is None

    simulation.subsample(n=1, seed="baseline-system", time_period="2022")

    assert simulation.baseline.baseline is None
    assert simulation.baseline.parent_branch is simulation


def test_a_reform_simulation_without_a_default_system_instance_still_builds():
    """With no instance to use, the baseline branch gets no system."""

    class SimulationWithoutInstance(CoreSimulation):
        default_tax_benefit_system = CountryTaxBenefitSystem
        default_role = "parent"
        default_calculation_period = PERIOD
        default_input_period = PERIOD

    simulation = SimulationWithoutInstance(
        situation=SITUATION, reform=neutralize_income_tax
    )

    assert simulation.baseline.tax_benefit_system is None
    np.testing.assert_array_equal(simulation.calculate("income_tax", PERIOD), 0)
