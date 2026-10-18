"""Properties of a reform simulation's baseline across clones and branches.

For any chain of ``clone`` and ``get_branch`` calls from a reform simulation
(under any of four reforms), any salaries set along the way on any member or
on its baseline, any calculations and any cache invalidations:

* A member made by ``clone`` owns a baseline branch: it is the member's
  branch, registered under ``"baseline"``, and ``get_branch("baseline")``
  returns it. A member made by ``get_branch`` shares its parent's baseline.
* Every baseline's populations use its system's entities, and its holders
  that system's variables.
* An operation on one simulation (a member, or a baseline) changes no other
  simulation's stored values, and copying one changes none.
* No two simulations share an ``invalidated_caches`` set.
* Every member returns what a new reform simulation with its inputs returns,
  and every baseline what a new simulation without the reform, with the
  baseline's inputs, returns (differential).

Examples, including the ones that failed before, are in
``test_clone_reform_baseline.py``, ``test_reform_baseline_system.py`` and
``test_clone_invalidated_caches.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.country_template import Simulation  # noqa: E402
from policyengine_core.periods import ETERNITY  # noqa: E402
from tests.core.test_reform_baseline_system import (  # noqa: E402
    add_doubled_salary,
    change_salary_default,
    neutralize_income_tax,
)

PERIOD = "2022-01"
PEOPLE = ["a", "b"]
REFORMS = {
    "rate": {"taxes.income_tax_rate": {"2022-01-01": 0.42}},
    "neutralize income tax": neutralize_income_tax,
    "salary default": change_salary_default,
    "added variable": add_doubled_salary,
}
OUTPUTS = [
    "salary",
    "income_tax",
    "disposable_income",
    "total_taxes",
    "household_income",
]
FORMULA_OUTPUTS = [variable for variable in OUTPUTS if variable != "salary"]
SALARY = st.lists(
    st.integers(min_value=0, max_value=10_000).map(float),
    min_size=len(PEOPLE),
    max_size=len(PEOPLE),
)


def _situation(salary):
    """``salary`` for each person, or, with ``None``, no one's salary."""
    persons = {person: {} for person in PEOPLE}
    if salary is not None:
        for person, amount in zip(PEOPLE, salary):
            persons[person]["salary"] = {PERIOD: amount}
    return {"persons": persons, "households": {"h": {"parents": PEOPLE}}}


@st.composite
def steps(draw):
    """Steps on a growing list of members, the first being the source.

    Each step names the member it acts on by its position in the list, and
    whether it acts on the member or on the member's baseline.
    """
    result = []
    count = 1
    for _ in range(draw(st.integers(min_value=1, max_value=7))):
        target = draw(st.integers(min_value=0, max_value=count - 1))
        kind = draw(
            st.sampled_from(
                [
                    "clone",
                    "clone sharing the system",
                    "branch",
                    "set salary",
                    "calculate",
                    "invalidate",
                ]
            )
        )
        on_baseline = draw(st.booleans())
        if kind == "set salary":
            argument = draw(SALARY)
        elif kind == "calculate":
            argument = draw(st.sampled_from(OUTPUTS))
        elif kind == "invalidate":
            # Formula values only: invalidating an input deletes it.
            argument = draw(st.sampled_from(FORMULA_OUTPUTS))
        else:
            argument = None
            count += 1
        result.append((kind, target, on_baseline, argument))
    return result


def _stored(simulation):
    """Every value ``simulation``'s holders store, by variable, branch and
    period."""
    stored = {}
    for population in simulation.populations.values():
        for name, holder in population._holders.items():
            for branch, period in holder.get_known_branch_periods():
                value = holder._memory_storage.get(period, branch)
                stored[(name, branch, str(period))] = np.array(value, copy=True)
    return stored


def _assert_same_storage(before, after, label):
    assert before.keys() == after.keys(), label
    for key, value in before.items():
        np.testing.assert_array_equal(after[key], value, err_msg=f"{label} {key}")


class Family:
    """The members, their baselines, and the salaries each one reads."""

    def __init__(self, reform, salary):
        source = Simulation(situation=_situation(salary), reform=REFORMS[reform])
        self.members = [source]
        self.parents = [None]
        self.made_by = ["source"]
        # Salary each simulation reads, by identity (``None``: no one's).
        self.salary = {id(source): salary, id(source.baseline): salary}
        # Simulations holding calculated values, their own or copied. An input
        # set on one would leave them stale (``set_input`` does not
        # invalidate what was calculated from the old input), so the model
        # sets inputs only on the others.
        self.calculated = set()

    def simulations(self):
        """Each member and each baseline, once."""
        seen = {}
        for member in self.members:
            seen[id(member)] = member
            seen[id(member.baseline)] = member.baseline
        return list(seen.values())

    def apply(self, step, index):
        kind, target, on_baseline, argument = step
        member = self.members[target]
        acted_on = member.baseline if on_baseline else member
        if kind in ("clone", "clone sharing the system", "branch"):
            if kind == "branch":
                # A branch named after the step, so no name is reused.
                copy = member.get_branch(f"branch_{index}")
            else:
                copy = member.clone(
                    clone_tax_benefit_system=kind == "clone",
                )
            self.members.append(copy)
            self.parents.append(member)
            self.made_by.append(kind)
            self.salary[id(copy)] = self.salary[id(member)]
            if id(member) in self.calculated:
                self.calculated.add(id(copy))
            if copy.baseline is not member.baseline:
                self.salary[id(copy.baseline)] = self.salary[id(member.baseline)]
                if id(member.baseline) in self.calculated:
                    self.calculated.add(id(copy.baseline))
            return None
        if kind == "set salary":
            if id(acted_on) not in self.calculated:
                acted_on.set_input("salary", PERIOD, argument)
                self.salary[id(acted_on)] = list(argument)
        elif kind == "calculate":
            if argument in acted_on.tax_benefit_system.variables:
                acted_on.calculate(argument, PERIOD)
                self.calculated.add(id(acted_on))
        elif kind == "invalidate":
            acted_on.invalidate_cache_entry(argument, PERIOD)
        return acted_on


def _run_and_check_independence(reform, salary, steps_):
    family = Family(reform, salary)
    for index, step in enumerate(steps_):
        before = {id(s): (s, _stored(s)) for s in family.simulations()}
        sets_before = {id(s): set(s.invalidated_caches) for s in family.simulations()}
        acted_on = family.apply(step, index)
        for key, (simulation, stored) in before.items():
            if simulation is acted_on:
                continue
            _assert_same_storage(stored, _stored(simulation), f"step {index} {step}")
            assert simulation.invalidated_caches == sets_before[key]
    return family


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    reform=st.sampled_from(list(REFORMS)),
    salary=st.one_of(st.none(), SALARY),
    steps_=steps(),
)
# Reform values cached before cloning; the clone's baseline must not read
# them as its own.
@example(
    reform="rate",
    salary=[4000.0, 2500.0],
    steps_=[("calculate", 0, False, "income_tax"), ("clone", 0, False, None)],
)
# Inputs set on a clone's baseline, then a clone of that clone.
@example(
    reform="neutralize income tax",
    salary=[4000.0, 2500.0],
    steps_=[
        ("clone", 0, False, None),
        ("set salary", 1, True, [100.0, 200.0]),
        ("clone", 1, False, None),
    ],
)
# A clone of a branch shares the branch's (its parent's) baseline.
@example(
    reform="salary default",
    salary=None,
    steps_=[
        ("branch", 0, False, None),
        ("clone", 1, False, None),
        ("set salary", 2, True, [5.0, 6.0]),
    ],
)
# An invalidation recorded on a branch stays out of its parent's purge.
@example(
    reform="added variable",
    salary=[1.0, 2.0],
    steps_=[
        ("branch", 0, False, None),
        ("invalidate", 1, False, "income_tax"),
        ("calculate", 0, False, "income_tax"),
    ],
)
def test_every_clone_has_a_baseline_of_its_own(reform, salary, steps_):
    family = _run_and_check_independence(reform, salary, steps_)
    simulations = family.simulations()

    # No two share a set of invalidations.
    sets = [id(s.invalidated_caches) for s in simulations]
    assert len(set(sets)) == len(sets)

    for member, parent, made_by in zip(family.members, family.parents, family.made_by):
        baseline = member.baseline
        if made_by == "branch":
            assert baseline is parent.baseline
        elif made_by == "source" or baseline.parent_branch is member:
            # Its own branch.
            assert baseline.parent_branch is member
            assert member.branches["baseline"] is baseline
            assert member.get_branch("baseline") is baseline
            assert baseline.branch_name == "baseline"
            assert baseline.tracer is member.tracer
        else:
            # A clone of a member whose baseline is not its own shares it.
            assert made_by != "branch"
            assert baseline is parent.baseline
            assert parent.baseline.parent_branch is not parent
        if made_by in ("clone", "clone sharing the system"):
            if parent.baseline.parent_branch is parent:
                assert baseline is not parent.baseline
                assert baseline.tax_benefit_system is parent.baseline.tax_benefit_system

    # Every baseline uses its own system. (A member made by ``clone`` with a
    # copy of the system is policyengine-core#586's and its follow-ups'.)
    baselines = {id(member.baseline): member.baseline for member in family.members}
    for baseline in baselines.values():
        system = baseline.tax_benefit_system
        entities = {
            entity.key: entity
            for entity in [system.person_entity, *system.group_entities]
        }
        for key, population in baseline.populations.items():
            assert population.entity is entities[key]
            for name, holder in population._holders.items():
                assert holder.variable is system.variables[name]
                assert holder.variable.entity.key == key
                assert holder._memory_storage.is_eternal == (
                    holder.variable.definition_period == ETERNITY
                )

    # Differential: each returns what a new simulation with its inputs and
    # policy returns.
    for simulation in simulations:
        situation = _situation(family.salary[id(simulation)])
        if id(simulation) in baselines:
            fresh = Simulation(situation=situation)
        else:
            fresh = Simulation(situation=situation, reform=REFORMS[reform])
        for variable in OUTPUTS + ["doubled_salary"]:
            if variable not in fresh.tax_benefit_system.variables:
                assert variable not in simulation.tax_benefit_system.variables
                continue
            np.testing.assert_allclose(
                simulation.calculate(variable, PERIOD),
                fresh.calculate(variable, PERIOD),
                rtol=1e-6,
                err_msg=f"{variable} on {simulation.branch_name}",
            )
