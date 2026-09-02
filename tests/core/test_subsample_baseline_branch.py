"""Regression tests for rebuilding the baseline branch in ``subsample``."""

from __future__ import annotations

import pytest

from policyengine_core.country_template import Microsimulation

REFORM_RATE = 0.42
BASELINE_RATE = 0.15
INSTANT = "2022-01-01"


def test_subsample_restores_baseline_system_and_branch_wiring():
    """After subsample, the baseline branch must keep the baseline policy."""
    simulation = Microsimulation(
        reform={"taxes.income_tax_rate": {INSTANT: REFORM_RATE}}
    )
    baseline_system_before = simulation.branches["baseline"].tax_benefit_system

    simulation.subsample(n=3, seed="fix-522", time_period="2022")

    baseline = simulation.branches["baseline"]
    assert "tax_benefit_system" not in simulation.branches
    assert baseline.tax_benefit_system is baseline_system_before
    assert (
        baseline.tax_benefit_system.parameters.taxes.income_tax_rate(INSTANT)
        == BASELINE_RATE
    )
    assert (
        simulation.tax_benefit_system.parameters.taxes.income_tax_rate(INSTANT)
        == REFORM_RATE
    )
    assert baseline.branch_name == "baseline"
    assert baseline.parent_branch is simulation
    assert simulation.baseline is baseline
    assert baseline.persons.count == simulation.persons.count

    reform_tax = simulation.calculate("income_tax").sum()
    baseline_tax = baseline.calculate("income_tax").sum()
    assert reform_tax > 0
    # The template's income tax is flat-rate, so the branch must reproduce
    # the baseline policy proportionally on the same subsample.
    assert baseline_tax == pytest.approx(
        reform_tax * BASELINE_RATE / REFORM_RATE, rel=1e-6
    )


def test_subsample_without_reform_creates_no_baseline_branch():
    simulation = Microsimulation()

    simulation.subsample(n=3, seed="fix-522", time_period="2022")

    assert "baseline" not in simulation.branches
    assert "tax_benefit_system" not in simulation.branches
