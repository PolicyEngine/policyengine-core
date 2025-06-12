import os
import pytest


@pytest.mark.smoke
@pytest.mark.skipif(
    os.getenv("RUN_SMOKE_TESTS") != "1",
    reason="Skip smoke tests unless explicitly enabled",
)
def test_policyengine_us_microsimulation_runs():
    from policyengine_us import Microsimulation

    baseline = Microsimulation()
    baseline.subsample(100)

    baseline.calculate("household_net_income")
