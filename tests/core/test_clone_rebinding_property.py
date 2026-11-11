"""Properties of clones and branches: each one is a simulation of its own.

For any chain of ``clone`` and ``get_branch`` calls (with or without a copy
of the tax-benefit system), any inputs set along the way and any reform
applied to one of them, every simulation in the chain:

* has its method aliases (``calc``, ``df``) bound to itself;
* returns from ``calc`` and ``df`` what ``calculate`` and
  ``calculate_dataframe`` return (two paths, one meaning);
* has populations whose entities are its own system's, and, when it was
  given a copy of the system, is the simulation that copy names;
* returns what a new simulation given the same inputs returns.

Examples, including the ones that failed before clones were rebound, are in
``test_clone_rebinding.py``.
"""

from __future__ import annotations

import copy
import types

import numpy as np
import pandas as pd
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.country_template import (  # noqa: E402
    CountryTaxBenefitSystem,
    Simulation,
)
from policyengine_core.country_template.situation_examples import couple  # noqa: E402
from policyengine_core.simulations import Simulation as CoreSimulation  # noqa: E402
from tests.core.test_clone_rebinding import add_doubled_salary  # noqa: E402

PERIOD = "2017-01"
PEOPLE = ["Alicia", "Javier"]
SOURCE_SALARY = [4000.0, 2500.0]  # from ``couple``
OUTPUTS = [
    "salary",
    "income_tax",
    "social_security_contribution",
    "disposable_income",
    "household_income",
    "total_benefits",
    "total_taxes",
]
PERSON_OUTPUTS = ["salary", "income_tax", "disposable_income"]

MAKERS = {
    "clone": lambda s, name: s.clone(),
    "clone sharing the system": lambda s, name: s.clone(clone_tax_benefit_system=False),
    "branch": lambda s, name: s.get_branch(name),
    "branch with its own system": lambda s, name: s.get_branch(name, clone_system=True),
}
OWN_SYSTEM = {"clone", "branch with its own system"}

SALARY = st.lists(
    st.integers(min_value=0, max_value=10_000).map(float),
    min_size=len(PEOPLE),
    max_size=len(PEOPLE),
)


@st.composite
def operations(draw):
    """Steps on a growing list of simulations, the first being the source.

    Each step names the simulation it acts on by its position in the list.
    """
    steps = []
    count = 1
    for _ in range(draw(st.integers(min_value=1, max_value=6))):
        target = draw(st.integers(min_value=0, max_value=count - 1))
        kind = draw(st.sampled_from([*MAKERS, "set salary", "reform"]))
        if kind == "set salary":
            steps.append((kind, target, draw(SALARY)))
        elif kind == "reform":
            steps.append((kind, target, None))
        else:
            steps.append((kind, target, None))
            count += 1
    return steps


def _fresh(salary):
    """A new simulation of ``couple`` with these salaries."""
    situation = copy.deepcopy(couple)
    for person, amount in zip(PEOPLE, salary):
        situation["persons"][person]["salary"] = {PERIOD: amount}
    return Simulation(tax_benefit_system=CountryTaxBenefitSystem(), situation=situation)


def _run(steps):
    """Apply ``steps``; return each simulation, how it was made and the
    salaries it should read."""
    source = Simulation(tax_benefit_system=CountryTaxBenefitSystem(), situation=couple)
    simulations = [source]
    kinds = ["source"]
    salaries = [list(SOURCE_SALARY)]
    for index, (kind, target, salary) in enumerate(steps):
        simulation = simulations[target]
        if kind == "set salary":
            simulation.set_input("salary", PERIOD, salary)
            salaries[target] = list(salary)
        elif kind == "reform":
            simulation.apply_reform(add_doubled_salary)
        else:
            # A branch named after the step, so no name is ever reused.
            simulations.append(MAKERS[kind](simulation, f"branch_{index}"))
            kinds.append(kind)
            # A clone or branch starts from what its source holds then;
            # later inputs on either stay out of the other.
            salaries.append(list(salaries[target]))
    return simulations, kinds, salaries


def _system_entities(system):
    return {
        entity.key: entity for entity in [system.person_entity, *system.group_entities]
    }


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(operations())
# A clone of a clone, each with its own inputs.
@example([("clone", 0, None), ("clone", 1, None), ("set salary", 2, [1.0, 2.0])])
# A clone of a branch, and a branch of a clone.
@example(
    [
        ("branch", 0, None),
        ("clone", 1, None),
        ("set salary", 2, [5.0, 6.0]),
        ("branch", 2, None),
    ]
)
# A reform on a clone with its own system, which its source must not see.
@example([("clone", 0, None), ("reform", 1, None), ("set salary", 1, [7.0, 0.0])])
def test_every_clone_and_branch_is_a_simulation_of_its_own(steps):
    simulations, kinds, salaries = _run(steps)
    for simulation, kind, salary in zip(simulations, kinds, salaries):
        # Its aliases are its own: no bound method of another simulation is
        # left on it.
        assert simulation.calc.__self__ is simulation
        assert simulation.df.__self__ is simulation
        for value in vars(simulation).values():
            if isinstance(value, types.MethodType) and isinstance(
                value.__self__, CoreSimulation
            ):
                assert value.__self__ is simulation

        # Its populations use its own system.
        system = simulation.tax_benefit_system
        entities = _system_entities(system)
        for key, population in simulation.populations.items():
            assert population.entity is entities[key]
            assert population.entity._tax_benefit_system is system
        if kind in OWN_SYSTEM:
            assert system.simulation is simulation

        # ``calc`` and ``df`` are ``calculate`` and ``calculate_dataframe``.
        for variable in OUTPUTS:
            np.testing.assert_array_equal(
                simulation.calc(variable, PERIOD),
                simulation.calculate(variable, PERIOD),
            )
        pd.testing.assert_frame_equal(
            simulation.df(PERSON_OUTPUTS, PERIOD),
            simulation.calculate_dataframe(PERSON_OUTPUTS, PERIOD),
        )

        # It returns what a new simulation with its inputs returns.
        np.testing.assert_array_equal(simulation.calc("salary", PERIOD), salary)
        fresh = _fresh(salary)
        for variable in OUTPUTS:
            np.testing.assert_allclose(
                simulation.calc(variable, PERIOD),
                fresh.calculate(variable, PERIOD),
                rtol=1e-6,
                err_msg=variable,
            )

        # A variable a reform added is there exactly when its system has it.
        if "doubled_salary" in system.variables:
            np.testing.assert_array_equal(
                simulation.calc("doubled_salary", PERIOD), 2 * np.array(salary)
            )
        else:
            with pytest.raises(ValueError, match="does not exist"):
                simulation.calc("doubled_salary", PERIOD)
