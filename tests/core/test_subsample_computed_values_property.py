"""Properties of ``subsample`` on a reform simulation.

For any population of one to four households of one to three people, any
salaries and household weights, with or without a dataset value for a
formula-backed variable (income tax), under any of four reforms or none,
after any calculations on either arm, and for any sample size and seed:

* The recorded-input export is a function of the inputs: calculating
  anything on either arm leaves it unchanged.
* History independence (differential): subsampling after the calculations
  rebuilds the same inputs, and both arms then return the same values, as
  subsampling without them.
* Baseline fidelity (differential): the baseline arm returns what a
  simulation without the reform, subsampled the same way, returns.
* Restriction: every sampled person reads, in each arm, what they read in a
  simulation that was not subsampled.

Examples, including the ones that failed before, are in
``test_subsample_computed_values.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.country_template import Microsimulation  # noqa: E402
from policyengine_core.data import Dataset  # noqa: E402
from policyengine_core.periods import ETERNITY  # noqa: E402
from tests.core.test_reform_baseline_system import add_doubled_salary  # noqa: E402
from tests.core.test_subsample_computed_values import (  # noqa: E402
    RATE_REFORM,
    change_salary_default,
    halve_salaries_in_tax,
)

PERIOD = "2022-01"
REFORMS = {
    "none": None,
    "rate": RATE_REFORM,
    "structural": halve_salaries_in_tax,
    "salary default": change_salary_default,
    "added variable": add_doubled_salary,
}
# Values each arm returns, by variable and period. None depends on weights.
OUTPUTS = [
    ("salary", PERIOD),
    ("salary", "2022-02"),
    ("income_tax", PERIOD),
    ("income_tax", "2022-02"),
    ("social_security_contribution", PERIOD),
    ("disposable_income", PERIOD),
    ("total_taxes", PERIOD),
    ("household_income", PERIOD),
    ("doubled_salary", PERIOD),
]
# What may be calculated before subsampling: the outputs and weights,
# including a household weight for a year the dataset has none for.
CALCULATIONS = OUTPUTS + [
    ("person_weight", "2022"),
    ("household_weight", "2022"),
    ("household_weight", "2023"),
]


@st.composite
def populations(draw):
    sizes = draw(st.lists(st.integers(1, 3), min_size=1, max_size=4))
    people = sum(sizes)
    amounts = st.integers(min_value=0, max_value=10_000).map(float)
    salaries = draw(st.lists(amounts, min_size=people, max_size=people))
    weights = draw(
        st.lists(
            st.integers(min_value=1, max_value=1_000).map(float),
            min_size=len(sizes),
            max_size=len(sizes),
        )
    )
    income_tax = draw(st.none() | st.lists(amounts, min_size=people, max_size=people))
    return sizes, salaries, weights, income_tax


def _dataset(population) -> Dataset:
    sizes, salaries, weights, income_tax = population
    # Ids that are not row positions, so a mix-up between the two shows.
    household_ids = [10 * (index + 1) for index, size in enumerate(sizes)]
    membership = np.repeat(household_ids, sizes)
    df = pd.DataFrame(
        {
            "person_id__2022": 100 + np.arange(len(salaries)),
            "household_id__2022": membership,
            "person_household_id__2022": membership,
            "household_weight__2022": np.repeat(weights, sizes),
            f"salary__{PERIOD}": salaries,
        }
    )
    if income_tax is not None:
        df[f"income_tax__{PERIOD}"] = income_tax
    return Dataset.from_dataframe(df, "2022")


def _simulation(reform, population) -> Microsimulation:
    return Microsimulation(dataset=_dataset(population), reform=REFORMS[reform])


def _arms(simulation):
    """The simulation, and its baseline arm if it has a reform."""
    baseline = getattr(simulation, "baseline", None)
    return [simulation] if baseline is None else [simulation, baseline]


def _by_person(simulation, variable, period):
    """What each person reads, keyed by person id."""
    ids = np.asarray(simulation.calculate("person_id", ETERNITY))
    values = np.asarray(simulation.calculate(variable, period, map_to="person"))
    return dict(zip(ids.tolist(), values.tolist()))


def _recorded_inputs(simulation):
    """The input export ``subsample`` uses, including formula-backed inputs."""
    return simulation._to_person_dataframe(simulation._get_set_input_periods)


def _check(reform, population, calculations, n, seed):
    def subsample(simulation):
        return simulation.subsample(n=n, seed=seed, time_period="2022")

    reference = _simulation(reform, population)
    calculated = _simulation(reform, population)
    exports = [_recorded_inputs(arm) for arm in _arms(calculated)]
    for (variable, period), on_baseline in calculations:
        arm = _arms(calculated)[-1] if on_baseline else calculated
        if variable in arm.tax_benefit_system.variables:
            arm.calculate(variable, period)
    for arm, export in zip(_arms(calculated), exports):
        pd.testing.assert_frame_equal(_recorded_inputs(arm), export)

    subsample(calculated)
    uncalculated = subsample(_simulation(reform, population))
    unreformed = subsample(_simulation("none", population))

    assert len(_arms(calculated)) == len(_arms(uncalculated)) == len(_arms(reference))
    for arm, uncalculated_arm in zip(_arms(calculated), _arms(uncalculated)):
        pd.testing.assert_frame_equal(
            _recorded_inputs(arm),
            _recorded_inputs(uncalculated_arm),
        )
    for variable, period in OUTPUTS:
        for arm, uncalculated_arm, reference_arm in zip(
            _arms(calculated), _arms(uncalculated), _arms(reference)
        ):
            if variable not in arm.tax_benefit_system.variables:
                continue
            label = f"{variable}@{period} in {arm.branch_name}"
            values = np.asarray(arm.calculate(variable, period))
            np.testing.assert_array_equal(
                values,
                np.asarray(uncalculated_arm.calculate(variable, period)),
                err_msg=label,
            )
            before = _by_person(reference_arm, variable, period)
            after = _by_person(arm, variable, period)
            assert after == {key: before[key] for key in after}, label
        if variable in unreformed.tax_benefit_system.variables:
            np.testing.assert_array_equal(
                np.asarray(_arms(calculated)[-1].calculate(variable, period)),
                np.asarray(unreformed.calculate(variable, period)),
                err_msg=f"{variable}@{period} in the baseline arm",
            )


TWO_HOUSEHOLDS = ([1, 2], [1000.0, 2000.0, 3000.0], [1.0, 3.0], None)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    reform=st.sampled_from(list(REFORMS)),
    population=populations(),
    calculations=st.lists(
        st.tuples(st.sampled_from(CALCULATIONS), st.booleans()), max_size=6
    ),
    n=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**16),
)
# Income tax calculated under the reform became the baseline's.
@example(
    reform="rate",
    population=TWO_HOUSEHOLDS,
    calculations=[(("income_tax", PERIOD), False)],
    n=2,
    seed=0,
)
# So did a formula the reform replaces, and what it feeds.
@example(
    reform="structural",
    population=TWO_HOUSEHOLDS,
    calculations=[(("disposable_income", PERIOD), False)],
    n=2,
    seed=0,
)
# A default the reform changes, read where no input is.
@example(
    reform="salary default",
    population=TWO_HOUSEHOLDS,
    calculations=[(("salary", "2022-02"), False)],
    n=2,
    seed=0,
)
# A household weight read for a year the dataset has none for: its default
# came back as an input for that year.
@example(
    reform="none",
    population=TWO_HOUSEHOLDS,
    calculations=[(("household_weight", "2023"), False)],
    n=2,
    seed=0,
)
# A dataset value for a formula-backed variable stays an input of both arms.
@example(
    reform="rate",
    population=([2, 1], [1000.0, 2000.0, 3000.0], [1.0, 3.0], [5.0, 6.0, 7.0]),
    calculations=[(("income_tax", "2022-02"), True)],
    n=3,
    seed=1,
)
# Combined calculations on both arms cannot affect either rebuild.
@example(
    reform="structural",
    population=TWO_HOUSEHOLDS,
    calculations=[
        (("income_tax", PERIOD), False),
        (("disposable_income", PERIOD), True),
        (("household_weight", "2023"), False),
        (("salary", "2022-02"), True),
    ],
    n=2,
    seed=0,
)
def test_subsample_after_calculating_is_subsample_before(
    reform, population, calculations, n, seed
):
    _check(reform, population, calculations, n, seed)
