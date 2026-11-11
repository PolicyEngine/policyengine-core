"""Differential sampling coverage across requested periods, seeds and sizes."""

import pytest

from policyengine_core.country_template import Microsimulation, Simulation
from tests.core.test_subsample_sampling_period import check_sampling_period
from tests.fixtures.subsample_inputs import DATASET_YEAR

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies


@hypothesis.settings(max_examples=35, deadline=None)
@hypothesis.given(
    simulation_class=st.sampled_from([Simulation, Microsimulation]),
    year=st.sampled_from([DATASET_YEAR, "2023", "2024", "2022-07", "2023-07"]),
    n=st.integers(1, 8),
    seed=st.sampled_from(["a", "b", "c"]),
    quantize_weights=st.booleans(),
)
def test_requested_period_sampling_matches_direct_weights(
    simulation_class, year, n, seed, quantize_weights
):
    check_sampling_period(simulation_class, year, n, seed, quantize_weights)
