"""A copied simulation is a simulation in its own right.

For random situations, warm caches and branches, a ``copy.deepcopy`` or
pickle round trip of a simulation calculates what a freshly built simulation
does, and writing to the copy leaves the original unchanged.
``test_simulation_copy_pickle.py`` pins the same behaviour with examples.
"""

from __future__ import annotations

import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.tools import assert_near
from tests.fixtures.simulation_copy import COPIERS, FEB, JAN, build_simulation

YEAR = "2025"
VARIABLES = (
    "income_tax",
    "social_security_contribution",
    "basic_income",
    "disposable_income",
    "household_income",
    "total_taxes",
    "total_benefits",
)


@st.composite
def situations(draw):
    salaries = draw(st.lists(st.integers(0, 20_000), min_size=1, max_size=4))
    count = len(salaries)
    order = draw(st.permutations(range(count)))
    cuts = sorted(draw(st.sets(st.integers(1, count - 1)))) if count > 1 else []
    bounds = [0, *cuts, count]
    households = [list(order[a:b]) for a, b in zip(bounds, bounds[1:])]
    occupancy = draw(
        st.lists(
            st.sampled_from([None, "owner", "tenant", "free_lodger", "homeless"]),
            min_size=len(households),
            max_size=len(households),
        )
    )
    warmed = draw(st.lists(st.sampled_from(VARIABLES), unique=True))
    branch_salary = draw(st.none() | st.integers(0, 20_000))
    return salaries, households, occupancy, warmed, branch_salary


@hypothesis.settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(situation=situations(), copier_name=st.sampled_from(sorted(COPIERS)))
def test_copy_calculates_like_a_fresh_simulation_and_stays_independent(
    situation, copier_name
):
    """For any situation and warm cache, a copy matches a freshly built
    simulation on every variable, and writing to the copy leaves the original
    unchanged."""
    salaries, households, occupancy, warmed, branch_salary = situation
    tax_benefit_system = CountryTaxBenefitSystem()
    original = build_simulation(tax_benefit_system, salaries, households, occupancy)
    for variable in warmed:
        original.calculate(variable, JAN)
    if branch_salary is not None:
        original.get_branch("child").set_input(
            "salary", FEB, [branch_salary] * len(salaries)
        )

    copied = COPIERS[copier_name](original)
    fresh = build_simulation(tax_benefit_system, salaries, households, occupancy)

    for variable in VARIABLES:
        assert_near(copied.calculate(variable, JAN), fresh.calculate(variable, JAN))
    assert_near(
        copied.calculate("housing_tax", YEAR), fresh.calculate("housing_tax", YEAR)
    )
    assert list(copied.household("housing_occupancy_status", JAN).decode_to_str()) == [
        status or "tenant" for status in occupancy
    ]
    if branch_salary is not None:
        fresh_child = fresh.get_branch("child")
        fresh_child.set_input("salary", FEB, [branch_salary] * len(salaries))
        assert_near(
            copied.branches["child"].calculate("income_tax", FEB),
            fresh_child.calculate("income_tax", FEB),
        )

    copied.set_input("salary", FEB, [1] * len(salaries))
    copied.calculate("disposable_income", FEB)
    assert original.persons.get_holder("salary").get_array(FEB) is None
    assert original.persons.get_holder("disposable_income").get_array(FEB) is None
    for variable in warmed:
        assert_near(original.calculate(variable, JAN), fresh.calculate(variable, JAN))
