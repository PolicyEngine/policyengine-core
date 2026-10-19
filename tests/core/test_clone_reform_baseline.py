"""A clone of a reform simulation has a baseline of its own.

``Simulation.__init__`` gives a reform simulation a baseline branch,
``baseline``, under the baseline policy. ``Simulation.clone`` copies the
instance dictionary, so a clone shared that branch with its source:
calculations on the clone's baseline filled the source's, an input set on
either one's baseline reached the other, ``clone.get_branch("baseline")``
made a new branch under the clone's own (reform) policy, ``subsample`` of
the clone left its baseline at the old size, and the clone kept its source
alive through the branch's ``parent_branch``.

A branch still shares its parent's baseline, which formulas running in the
branch read through it, and so does a clone of a simulation whose baseline
is not its own branch (a branch, or a country package's separate baseline
simulation).

Properties over chains of clones and branches are in
``test_clone_reform_baseline_property.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core.country_template import Microsimulation, Simulation

PERIOD = "2022-01"
INSTANT = "2022-01-01"
BASELINE_RATE = 0.15
REFORM_RATE = 0.42
REFORM = {"taxes.income_tax_rate": {INSTANT: REFORM_RATE}}
SALARY = [4000.0, 2500.0]
OTHER_SALARY = [1000.0, 3000.0]
SITUATION = {
    "persons": {
        "a": {"salary": {PERIOD: SALARY[0]}},
        "b": {"salary": {PERIOD: SALARY[1]}},
    },
    "households": {"h": {"parents": ["a", "b"]}},
}
OUTPUTS = ["income_tax", "disposable_income", "total_taxes", "household_income"]


def _reform_simulation() -> Simulation:
    return Simulation(situation=SITUATION, reform=REFORM)


def _income_tax(salary, rate):
    return np.array(salary) * rate


MAKERS = [
    pytest.param(lambda s: s.clone(), id="clone"),
    pytest.param(
        lambda s: s.clone(clone_tax_benefit_system=False),
        id="clone-sharing-system",
    ),
    pytest.param(lambda s: s.clone().clone(), id="clone-of-clone"),
    pytest.param(lambda s: s.clone(trace=True), id="traced-clone"),
]


@pytest.mark.parametrize("make", MAKERS)
def test_a_clone_has_its_own_baseline_branch(make):
    source = _reform_simulation()
    clone = make(source)

    assert clone.baseline is not source.baseline
    assert clone.baseline.parent_branch is clone
    assert clone.baseline.branch_name == "baseline"
    assert clone.branches == {"baseline": clone.baseline}
    assert clone.get_branch("baseline") is clone.baseline
    # The baseline policy is shared, as ``__init__`` shares it.
    assert clone.baseline.tax_benefit_system is source.baseline.tax_benefit_system
    # Traced in its simulation, as ``__init__`` makes it.
    assert clone.baseline.tracer is clone.tracer
    assert clone.baseline.trace == clone.trace
    # The source keeps its own.
    assert source.baseline.parent_branch is source
    assert source.branches["baseline"] is source.baseline


@pytest.mark.parametrize("make", MAKERS)
def test_a_clones_baseline_calculates_under_the_baseline_policy(make):
    clone = make(_reform_simulation())

    np.testing.assert_allclose(
        clone.calculate("income_tax", PERIOD), _income_tax(SALARY, REFORM_RATE)
    )
    np.testing.assert_allclose(
        clone.baseline.calculate("income_tax", PERIOD),
        _income_tax(SALARY, BASELINE_RATE),
    )
    # Formulas reach the baseline through ``get_branch("baseline")`` too
    # (policyengine-us's behavioural responses do). On a clone this made a
    # new branch under the clone's own policy, so it returned reform values.
    np.testing.assert_allclose(
        clone.get_branch("baseline").calculate("income_tax", PERIOD),
        _income_tax(SALARY, BASELINE_RATE),
    )


def test_a_clone_made_after_reform_calculations_keeps_baseline_values():
    """The clone's baseline copies the source's baseline branch.

    A new branch of the clone would start from the clone's cached arrays,
    which hold the values the source calculated under the reform, and read
    them as its own.
    """
    source = _reform_simulation()
    for variable in OUTPUTS:
        source.calculate(variable, PERIOD)
    clone = source.clone()

    fresh_baseline = Simulation(situation=SITUATION)
    for variable in OUTPUTS:
        np.testing.assert_allclose(
            clone.baseline.calculate(variable, PERIOD),
            fresh_baseline.calculate(variable, PERIOD),
            err_msg=variable,
        )


def test_a_clone_copies_what_its_sources_baseline_holds():
    source = _reform_simulation()
    source.baseline.set_input("salary", PERIOD, OTHER_SALARY)
    source.baseline.calculate("income_tax", PERIOD)

    clone = source.clone()

    np.testing.assert_array_equal(
        clone.baseline.calculate("salary", PERIOD), OTHER_SALARY
    )
    np.testing.assert_allclose(
        clone.baseline.calculate("income_tax", PERIOD),
        _income_tax(OTHER_SALARY, BASELINE_RATE),
    )


@pytest.mark.parametrize("calculate_on", ["clone", "source"])
def test_calculations_on_one_baseline_stay_in_it(calculate_on):
    source = _reform_simulation()
    clone = source.clone()
    used, other = (
        (clone.baseline, source.baseline)
        if calculate_on == "clone"
        else (source.baseline, clone.baseline)
    )

    used.calculate("income_tax", PERIOD)

    assert used.get_holder("income_tax").get_known_periods()
    assert other.get_holder("income_tax").get_known_periods() == []


@pytest.mark.parametrize("set_on", ["clone", "source"])
def test_an_input_set_on_one_baseline_stays_in_it(set_on):
    source = _reform_simulation()
    clone = source.clone()
    used, other = (
        (clone.baseline, source.baseline)
        if set_on == "clone"
        else (source.baseline, clone.baseline)
    )

    used.set_input("salary", PERIOD, OTHER_SALARY)

    np.testing.assert_array_equal(used.calculate("salary", PERIOD), OTHER_SALARY)
    np.testing.assert_array_equal(other.calculate("salary", PERIOD), SALARY)
    np.testing.assert_allclose(
        other.calculate("income_tax", PERIOD), _income_tax(SALARY, BASELINE_RATE)
    )


def test_subsample_of_a_clone_subsamples_its_baseline():
    source = Microsimulation(reform=REFORM)
    people = source.persons.count
    clone = source.clone()

    clone.subsample(n=1, seed="clone-baseline", time_period="2022")

    assert clone.persons.count < people
    assert clone.baseline.persons.count == clone.persons.count
    assert clone.baseline is clone.branches["baseline"]
    assert clone.baseline.parent_branch is clone
    reform_tax = clone.calculate("income_tax").values
    baseline_tax = clone.baseline.calculate("income_tax").values
    # The template's income tax is flat-rate.
    np.testing.assert_allclose(
        baseline_tax, reform_tax * BASELINE_RATE / REFORM_RATE, rtol=1e-6
    )
    # The source and its baseline keep every person.
    assert source.persons.count == people
    assert source.baseline.persons.count == people


def test_a_clone_does_not_refer_to_its_source_through_its_baseline():
    source = _reform_simulation()
    clone = source.clone()

    # Bound methods are left out: ``calc`` and ``df`` are bound to the
    # source on a core without policyengine-core#586.
    def references(simulation):
        return [
            value
            for value in vars(simulation).values()
            if not callable(getattr(value, "__func__", None))
        ]

    for value in references(clone) + references(clone.baseline):
        assert value is not source
        assert value is not source.baseline
    assert all(branch is not source.baseline for branch in clone.branches.values())


def test_a_traced_clone_traces_its_baseline():
    clone = _reform_simulation().clone(trace=True)

    clone.baseline.calculate("income_tax", PERIOD)

    traced = [(node.name, node.branch_name) for node in clone.tracer.browse_trace()]
    assert ("income_tax", "baseline") in traced


@pytest.mark.parametrize("clone_system", [False, True])
def test_a_branch_shares_its_parents_baseline(clone_system):
    source = _reform_simulation()

    branch = source.get_branch("branch", clone_system=clone_system)

    assert branch.baseline is source.baseline
    assert "baseline" not in branch.branches


def test_a_clone_of_a_branch_shares_the_branchs_baseline():
    source = _reform_simulation()
    branch = source.get_branch("branch")

    clone = branch.clone()

    # The branch's baseline is its parent's, so the clone shares it, as it
    # shares the branch's ``parent_branch``.
    assert clone.baseline is source.baseline
    assert clone.parent_branch is source


def test_a_separate_baseline_simulation_is_shared():
    """A baseline a country package builds as a simulation of its own.

    (policyengine-uk does.) Core does not know how it was built, so a clone
    shares it; the package's own ``clone`` can copy it.
    """
    source = Simulation(situation=SITUATION)
    source.baseline = Simulation(situation=SITUATION)

    clone = source.clone()

    assert clone.baseline is source.baseline
    assert clone.branches == {}


def test_a_simulation_without_a_baseline_clones_without_one():
    clone = Simulation(situation=SITUATION).clone()

    assert clone.baseline is None
    assert clone.branches == {}
