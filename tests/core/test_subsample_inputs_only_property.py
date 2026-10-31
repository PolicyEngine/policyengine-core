"""Property: what was calculated before ``subsample`` leaves no trace.

For any calculations made before subsampling (formula variables, weights and
inputs, at the dataset year and other years, with auto-carry-over on or off),
the subsampled simulation stores exactly the values a simulation subsampled
straight after loading does, and every later result agrees.
"""

import pytest

from tests.fixtures.subsample_inputs import (
    DATASET_YEAR,
    FORMULA_VARIABLES,
    build,
    stored,
)

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

VARIABLES = FORMULA_VARIABLES + ["base", "overridden", "household_weight"]
YEARS = ["2021", DATASET_YEAR, "2023", "2024"]
REQUESTS = st.tuples(st.sampled_from(VARIABLES), st.sampled_from(YEARS))


def subsampled(carry_over, prior, n, seed):
    simulation = build(carry_over)
    for variable, period in prior:
        simulation.calculate(variable, period)
    simulation.subsample(n=n, seed=seed, time_period=DATASET_YEAR)
    return simulation


@hypothesis.settings(max_examples=150, deadline=None)
@hypothesis.example(
    carry_over=True,
    prior=[("via_branch", DATASET_YEAR)],
    later=[("via_branch", DATASET_YEAR), ("via_branch", "2023")],
    n=4,
    seed="a",
)
@hypothesis.given(
    carry_over=st.booleans(),
    prior=st.lists(REQUESTS, min_size=1, max_size=6),
    later=st.lists(REQUESTS, min_size=1, max_size=4),
    n=st.integers(1, 8),
    seed=st.sampled_from(["a", "b", "c"]),
)
def test_calculations_before_subsample_leave_no_trace(
    carry_over, prior, later, n, seed
):
    fresh = subsampled(carry_over, [], n, seed)
    used = subsampled(carry_over, prior, n, seed)
    assert stored(used) == stored(fresh)
    for variable, period in later:
        assert (
            used.calculate(variable, period).tolist()
            == fresh.calculate(variable, period).tolist()
        )
    assert stored(used) == stored(fresh)
