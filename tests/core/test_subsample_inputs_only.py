"""``subsample`` rebuilds a simulation from its inputs, not its calculations.

``subsample`` used to export every stored value, calculated ones included,
and load them all back as inputs. A formula result calculated before
subsampling then replaced its formula for good: it was carried over past the
formula's end, and survived ``apply_reform`` and later ``set_input`` calls,
so results after subsampling depended on what had been calculated before.
"""

import numpy as np
import pytest

from policyengine_core.reforms import Reform
from policyengine_core.variables import Variable
from tests.fixtures.subsample_inputs import (
    DATASET_YEAR,
    build,
    doubled,
    stored,
)

SEED = "inputs-only"


def subsampled(prior=(), n=4):
    """A simulation subsampled after calculating ``prior`` (variable, period)."""
    simulation = build()
    for variable, period in prior:
        simulation.calculate(variable, period)
    simulation.subsample(n=n, seed=SEED, time_period=DATASET_YEAR)
    return simulation


def test_formula_result_is_not_carried_past_its_end_after_subsample():
    fresh = subsampled()
    used = subsampled(prior=[("ended", DATASET_YEAR)])
    assert used.calculate("ended", "2023").tolist() == [0.0] * used.persons.count
    assert (
        used.calculate("ended", "2023").tolist()
        == fresh.calculate("ended", "2023").tolist()
    )


def test_subsample_stores_the_same_values_whatever_was_calculated_before():
    fresh = subsampled()
    used = subsampled(
        prior=[
            ("doubled", DATASET_YEAR),
            ("household_total", DATASET_YEAR),
            ("person_weight", DATASET_YEAR),
            ("ended", DATASET_YEAR),
            ("base", "2024"),
        ]
    )
    assert stored(used) == stored(fresh)
    assert not any(name == "doubled" for name, _, _ in stored(used))


def test_calculated_value_follows_a_new_input_after_subsample():
    used = subsampled(prior=[("doubled", DATASET_YEAR)])
    count = used.persons.count
    used.set_input("base", DATASET_YEAR, np.full(count, 5.0))
    assert used.calculate("doubled", DATASET_YEAR).tolist() == [10.0] * count


def test_calculated_value_follows_a_reform_after_subsample():
    class tripled(Variable):
        value_type = float
        entity = doubled.entity
        definition_period = doubled.definition_period
        label = "Three times the input amount"

        def formula(person, period):
            return 3 * person("base", period)

    tripled.__name__ = "doubled"

    class reform(Reform):
        def apply(self):
            self.update_variable(tripled)

    used = subsampled(prior=[("doubled", DATASET_YEAR)])
    used.apply_reform(reform)
    base = used.calculate("base", DATASET_YEAR)
    assert used.calculate("doubled", DATASET_YEAR).tolist() == (3 * base).tolist()


def test_dataset_values_for_a_formula_variable_survive_subsample():
    # ``overridden`` has a formula returning -1, and dataset values that
    # replace it: person ``i`` (1-based ID) has ``i - 1``.
    simulation = subsampled()
    ids = simulation.calculate("person_id", DATASET_YEAR)
    assert (
        simulation.calculate("overridden", DATASET_YEAR).tolist()
        == (ids - 1).astype(float).tolist()
    )
    assert ("overridden", "default", DATASET_YEAR) in stored(simulation)


def test_formulas_run_on_the_subsample_after_subsample():
    simulation = subsampled(prior=[("household_total", DATASET_YEAR)])
    assert (
        simulation.calculate("doubled", DATASET_YEAR).tolist()
        == (2 * simulation.calculate("base", DATASET_YEAR)).tolist()
    )
    household_total = simulation.calculate("household_total", DATASET_YEAR)
    assert household_total.sum() == simulation.calculate("doubled", DATASET_YEAR).sum()


@pytest.mark.parametrize("with_baseline", [False, True])
def test_subsample_recreates_formula_branches_on_the_new_population(with_baseline):
    fresh = build()
    used = build()
    if with_baseline:
        for simulation in (fresh, used):
            simulation.baseline = simulation.get_branch("baseline", clone_system=True)
        baseline_system = used.baseline.tax_benefit_system

    used.calculate("via_branch", DATASET_YEAR)
    stale_branch = used.get_branch("probe")
    for simulation in (fresh, used):
        simulation.subsample(n=4, seed=SEED, time_period=DATASET_YEAR)

    assert "probe" not in used.branches
    if with_baseline:
        assert used.baseline is used.branches["baseline"]
        assert used.baseline.tax_benefit_system is baseline_system
        assert used.baseline.persons.count == used.persons.count

    for year in (DATASET_YEAR, "2023"):
        np.testing.assert_array_equal(
            used.calculate("via_branch", year), fresh.calculate("via_branch", year)
        )
    branch = used.get_branch("probe")
    assert branch is not stale_branch
    assert branch.persons.count == used.persons.count
    assert branch.household.count == used.household.count
