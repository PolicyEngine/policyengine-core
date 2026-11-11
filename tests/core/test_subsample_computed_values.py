"""Regression tests: ``subsample`` rebuilds from inputs, never calculated values.

``subsample`` exports surviving supplied inputs from their recorded storage
tiers, including overrides of formula-backed variables, and rebuilds both
the reform and baseline arms. Prior calculations must not become inputs:
each rebuilt arm must continue to use its own formulas and defaults.

The properties these examples come from are in
``test_subsample_computed_values_property.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyengine_core.country_template import Microsimulation
from policyengine_core.data import Dataset
from policyengine_core.model_api import Reform, Variable
from policyengine_core.periods import ETERNITY

PERIOD = "2022-01"
LATER_PERIOD = "2022-02"
SEED = "subsample-computed-values"
N = 3
REFORM_RATE = 0.42
REFORM_DEFAULT_SALARY = 1000.0
# A subsample of N households draws from 5. These outputs do not depend on
# weights, so each retained person's values should stay the same.
SALARIES = [1000.0, 2000.0, 0.0, 3500.0, 4000.0, 500.0, 6000.0, 0.0, 800.0, 2500.0]


class income_tax(Variable):
    def formula(person, period, parameters):
        return person("salary", period) * 0.5


class halve_salaries_in_tax(Reform):
    """A structural reform: income tax becomes half of salary."""

    def apply(self):
        self.update_variable(income_tax)


class salary(Variable):
    default_value = REFORM_DEFAULT_SALARY


class change_salary_default(Reform):
    def apply(self):
        self.update_variable(salary)


RATE_REFORM = {"taxes.income_tax_rate": {"2022-01-01": REFORM_RATE}}


def _dataframe(**columns) -> pd.DataFrame:
    """Five households of two people each, with salaries for ``PERIOD``."""
    df = pd.DataFrame(
        {
            "person_id__2022": list(range(1, 11)),
            "household_id__2022": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "person_household_id__2022": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "household_weight__2022": [100.0] * 2
            + [200.0] * 2
            + [300.0] * 2
            + [400.0] * 2
            + [500.0] * 2,
            f"salary__{PERIOD}": SALARIES,
        }
    )
    for name, values in columns.items():
        df[name] = values
    return df


def _simulation(reform=None, df=None) -> Microsimulation:
    return Microsimulation(
        dataset=Dataset.from_dataframe(_dataframe() if df is None else df, "2022"),
        reform=reform,
    )


def _subsample(simulation: Microsimulation) -> Microsimulation:
    return simulation.subsample(n=N, seed=SEED, time_period="2022")


def _values(simulation, variable, period=PERIOD) -> np.ndarray:
    return np.asarray(simulation.calculate(variable, period))


@pytest.mark.parametrize(
    "reform",
    [
        pytest.param(RATE_REFORM, id="parametric"),
        pytest.param(halve_salaries_in_tax, id="structural"),
    ],
)
@pytest.mark.parametrize("variable", ["income_tax", "disposable_income"])
def test_a_value_calculated_under_a_reform_is_not_the_baselines_after_subsample(
    reform, variable
):
    """The baseline arm equals an unreformed simulation of the same households."""
    simulation = _simulation(reform)
    reform_before = _values(simulation, variable)
    baseline_before = _values(simulation.baseline, variable)
    assert not np.allclose(reform_before, baseline_before)

    _subsample(simulation)
    unreformed = _subsample(_simulation())
    uncalculated = _subsample(_simulation(reform))

    np.testing.assert_array_equal(
        _values(simulation, "person_id", ETERNITY),
        _values(unreformed, "person_id", ETERNITY),
    )
    np.testing.assert_allclose(
        _values(simulation.baseline, variable), _values(unreformed, variable)
    )
    np.testing.assert_allclose(
        _values(simulation, variable), _values(uncalculated, variable)
    )


def test_a_default_the_reform_changes_is_not_the_baselines_after_subsample():
    """A default read under the reform is calculated, not an input."""
    simulation = _simulation(change_salary_default)
    assert (_values(simulation, "salary", LATER_PERIOD) == REFORM_DEFAULT_SALARY).all()

    _subsample(simulation)

    assert (_values(simulation.baseline, "salary", LATER_PERIOD) == 0).all()
    assert (_values(simulation, "salary", LATER_PERIOD) == REFORM_DEFAULT_SALARY).all()


def test_a_value_calculated_where_an_input_was_deleted_is_not_kept():
    """Deleting an override restores the formula without creating a new input.

    A subsequent calculation under the reform must not become an override
    of the baseline formula when subsampling rebuilds both arms.
    """
    simulation = _simulation(RATE_REFORM)
    simulation.set_input("income_tax", PERIOD, np.zeros(10))
    simulation.get_holder("income_tax").delete_arrays(PERIOD)
    reform_tax = _values(simulation, "income_tax")
    assert reform_tax.sum() > 0

    _subsample(simulation)
    unreformed = _subsample(_simulation())

    np.testing.assert_allclose(
        _values(simulation.baseline, "income_tax"), _values(unreformed, "income_tax")
    )


def test_a_dataset_value_of_a_formula_backed_variable_is_kept():
    """A dataset can supply a variable that has a formula, overriding it.

    Subsampling keeps such a value for the sampled households: the rebuilt
    population reads what the original read, in both arms.
    """
    supplied = np.arange(10, dtype=float) * 7
    df = _dataframe(**{f"income_tax__{PERIOD}": supplied})
    simulation = _simulation(RATE_REFORM, df=df)
    by_person = dict(zip(_values(simulation, "person_id", ETERNITY), supplied))

    _subsample(simulation)

    ids = _values(simulation, "person_id", ETERNITY)
    expected = np.array([by_person[person_id] for person_id in ids])
    np.testing.assert_array_equal(_values(simulation, "income_tax"), expected)
    np.testing.assert_array_equal(_values(simulation.baseline, "income_tax"), expected)
