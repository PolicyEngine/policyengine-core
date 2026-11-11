"""What a clone or branch takes from its source must be its own.

``Simulation.clone`` copies the source's instance dictionary, and that held
``calc`` and ``df``, bound methods of the source. So ``clone.calc(...)`` and
``clone.df(...)`` calculated on the source, and the clone kept the source
alive. A clone given a copy of the tax-benefit system also kept the source's
system: its copy still named the source as its simulation, and the clone's
populations still looked variables up in the source's system, so a clone
could not calculate a variable a reform added to its own system.

Properties over chains of clones and branches are in
``test_clone_rebinding_property.py``.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pandas as pd
import pytest

from policyengine_core.country_template import (
    CountryTaxBenefitSystem,
    Simulation,
)
from policyengine_core.country_template.entities import Person
from policyengine_core.country_template.situation_examples import couple
from policyengine_core.periods import MONTH
from policyengine_core.reforms import Reform
from policyengine_core.variables import Variable

PERIOD = "2017-01"
SOURCE_SALARY = [4000.0, 2500.0]  # from ``couple``
CLONE_SALARY = [1000.0, 3000.0]


def _source() -> Simulation:
    """A simulation of ``couple`` with a system of its own.

    A shared system (the country's default instance, or a module fixture)
    names the last simulation built with it, which would keep that one
    alive whatever its clones do.
    """
    return Simulation(tax_benefit_system=CountryTaxBenefitSystem(), situation=couple)


def _income_tax(simulation, salary):
    rate = simulation.tax_benefit_system.parameters(PERIOD).taxes.income_tax_rate
    return np.array(salary) * rate


class doubled_salary(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Salary times two"

    def formula(person, period, parameters):
        return person("salary", period) * 2


class add_doubled_salary(Reform):
    def apply(self):
        # ``update_variable`` adds it, and can be applied again: a reform
        # simulation applies its reform to a system built with it.
        self.update_variable(doubled_salary)


MAKERS = [
    pytest.param(lambda s: s.clone(), id="clone"),
    pytest.param(
        lambda s: s.clone(clone_tax_benefit_system=False),
        id="clone-sharing-system",
    ),
    pytest.param(lambda s: s.get_branch("branch"), id="branch"),
    pytest.param(
        lambda s: s.get_branch("branch", clone_system=True),
        id="branch-with-own-system",
    ),
    pytest.param(lambda s: s.clone().clone(), id="clone-of-clone"),
    pytest.param(lambda s: s.get_branch("branch").clone(), id="clone-of-branch"),
    pytest.param(lambda s: s.clone().get_branch("branch"), id="branch-of-clone"),
]


@pytest.mark.parametrize("make", MAKERS)
def test_calc_uses_the_clones_inputs(make):
    source = _source()
    clone = make(source)
    clone.set_input("salary", PERIOD, CLONE_SALARY)

    np.testing.assert_array_equal(clone.calc("salary", PERIOD), CLONE_SALARY)
    np.testing.assert_allclose(
        clone.calc("income_tax", PERIOD), _income_tax(clone, CLONE_SALARY)
    )
    # The source is unchanged, through its own alias.
    np.testing.assert_array_equal(source.calc("salary", PERIOD), SOURCE_SALARY)


@pytest.mark.parametrize("make", MAKERS)
def test_df_uses_the_clones_inputs(make):
    source = _source()
    clone = make(source)
    clone.set_input("salary", PERIOD, CLONE_SALARY)

    frame = clone.df(["salary", "income_tax"], PERIOD)
    assert list(frame.columns) == ["salary", "income_tax"]
    np.testing.assert_array_equal(frame["salary"], CLONE_SALARY)
    np.testing.assert_allclose(frame["income_tax"], _income_tax(clone, CLONE_SALARY))
    pd.testing.assert_frame_equal(
        frame, clone.calculate_dataframe(["salary", "income_tax"], PERIOD)
    )
    # The source is unchanged, through its own alias.
    np.testing.assert_array_equal(
        source.df(["salary"], PERIOD)["salary"], SOURCE_SALARY
    )


def test_calc_and_df_are_bound_to_the_clone():
    source = _source()
    for clone in (source.clone(), source.get_branch("branch")):
        assert clone.calc.__self__ is clone
        assert clone.calc.__func__ is source.calc.__func__
        assert clone.df.__self__ is clone
        assert clone.df.__func__ is source.df.__func__
    assert source.calc.__self__ is source
    assert source.df.__self__ is source


@pytest.mark.parametrize("depth", [1, 3])
def test_a_clone_does_not_keep_its_source_alive(depth):
    simulation = _source()
    ancestors = []
    for _ in range(depth):
        ancestors.append(weakref.ref(simulation))
        simulation = simulation.clone()
    clone = simulation
    del simulation
    gc.collect()

    assert [ancestor() for ancestor in ancestors] == [None] * depth
    # The clone still works on its own.
    clone.set_input("salary", PERIOD, CLONE_SALARY)
    np.testing.assert_allclose(
        clone.calc("income_tax", PERIOD), _income_tax(clone, CLONE_SALARY)
    )
    np.testing.assert_array_equal(clone.df(["salary"], PERIOD)["salary"], CLONE_SALARY)


def test_a_clone_of_a_reform_simulation_is_bound_to_itself():
    """A reform simulation's clone shares the source's ``baseline`` branch,
    so it keeps the source alive through it; its aliases are still its own."""
    source = Simulation(situation=couple, reform=add_doubled_salary)
    clone = source.clone()
    clone.set_input("salary", PERIOD, CLONE_SALARY)
    assert clone.calc.__self__ is clone
    np.testing.assert_array_equal(
        clone.calc("doubled_salary", PERIOD), 2 * np.array(CLONE_SALARY)
    )
    np.testing.assert_array_equal(
        source.calc("doubled_salary", PERIOD), 2 * np.array(SOURCE_SALARY)
    )


def test_bound_methods_a_subclass_keeps_are_rebound_too():
    """Every alias of a method of the source, not only ``calc`` and ``df``,
    and only those: a bound method of another object stays as it is."""

    class Helper:
        def describe(self):
            return "helper"

    helper = Helper()

    class AliasingSimulation(Simulation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.compute = self.calculate
            self.adds = self.calculate_add
            self.describe = helper.describe

    source = AliasingSimulation(
        tax_benefit_system=CountryTaxBenefitSystem(), situation=couple
    )
    clone = source.clone()
    clone.set_input("salary", PERIOD, CLONE_SALARY)

    assert clone.compute.__self__ is clone
    assert clone.adds.__self__ is clone
    np.testing.assert_array_equal(clone.compute("salary", PERIOD), CLONE_SALARY)
    np.testing.assert_array_equal(source.compute("salary", PERIOD), SOURCE_SALARY)
    assert clone.describe.__self__ is helper


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda s: s.clone(), id="clone"),
        pytest.param(
            lambda s: s.get_branch("branch", clone_system=True),
            id="branch-with-own-system",
        ),
    ],
)
def test_a_clone_calculates_a_variable_a_reform_adds_to_its_own_system(make):
    source = _source()
    clone = make(source)
    clone.apply_reform(add_doubled_salary)

    np.testing.assert_array_equal(
        clone.calculate("doubled_salary", PERIOD), 2 * np.array(SOURCE_SALARY)
    )
    assert "doubled_salary" not in source.tax_benefit_system.variables


def test_a_clone_with_its_own_system_looks_variables_up_in_it():
    source = _source()
    clone = source.clone()
    assert clone.tax_benefit_system is not source.tax_benefit_system
    assert clone.tax_benefit_system.simulation is clone
    for key, population in clone.populations.items():
        assert population.entity._tax_benefit_system is clone.tax_benefit_system
        assert population.entity.key == key
        assert population.entity is not source.populations[key].entity
    # The source keeps its own.
    assert source.tax_benefit_system.simulation is source
    for population in source.populations.values():
        assert population.entity._tax_benefit_system is source.tax_benefit_system


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda s: s.clone(clone_tax_benefit_system=False),
            id="clone-sharing-system",
        ),
        pytest.param(lambda s: s.get_branch("branch"), id="branch"),
    ],
)
def test_a_clone_sharing_its_sources_system_leaves_it_alone(make):
    """A shared system still names, and its entities still belong to, the
    simulation that built it."""
    source = _source()
    clone = make(source)
    assert clone.tax_benefit_system is source.tax_benefit_system
    assert source.tax_benefit_system.simulation is source
    for key, population in clone.populations.items():
        assert population.entity is source.populations[key].entity
