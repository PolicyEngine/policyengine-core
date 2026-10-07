"""The YAML runner must release per-case state during one process run."""

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from tests.fixtures.yaml_runner_memory import run_with_probe, write_cases


@pytest.mark.parametrize("reform_every", [0, 3])
def test_python_allocations_plateau_over_hundreds_of_cases(
    tmp_path,
    reform_every,
):
    """Later cases stay within a small allocation bound of the first 80."""

    path = write_cases(tmp_path / "cases.yaml", 240, reform_every=reform_every)
    probe = run_with_probe(
        CountryTaxBenefitSystem(),
        path,
        {"policy_system_cache_size": 2},
    )

    assert len(probe.outcomes) == 240
    assert probe.live_simulations == 0
    # The baseline entry retains one clone. Each of the two derived reform
    # entries retains both the reform and its immediate baseline clone.
    assert probe.live_systems <= 5
    # One sample every ten cases; index 7 is immediately after case 80.
    growth = max(probe.traced[8:]) - probe.traced[7]
    assert growth < 2_000_000, f"traced memory grew {growth / 1e6:.1f} MB"
