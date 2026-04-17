"""Regression test: ``Simulation.apply_reform`` must invalidate caches (H3).

Before this fix, ``sim.apply_reform(r)`` left the in-memory holder storage,
the ``_fast_cache`` and the on-disk storage populated with the pre-reform
values. Subsequent calls to ``calculate`` returned stale data.
"""

from __future__ import annotations

import numpy as np

from policyengine_core.model_api import Reform
from policyengine_core.country_template import situation_examples
from policyengine_core.simulations import SimulationBuilder


def test_apply_reform_invalidates_holder_cache(tax_benefit_system):
    sim = SimulationBuilder().build_from_entities(
        tax_benefit_system, situation_examples.single
    )
    period = "2017-01"

    before_reform = sim.calculate("basic_income", period=period)
    assert before_reform[0] > 0

    class NeutraliseBasicIncome(Reform):
        def apply(self):
            self.neutralize_variable("basic_income")

    sim.apply_reform(NeutraliseBasicIncome)

    after_reform = sim.calculate("basic_income", period=period)
    # After neutralisation the variable must return 0 regardless of cached
    # pre-reform values.
    assert after_reform[0] == 0, (
        f"apply_reform left stale cache values for basic_income; got "
        f"{after_reform[0]} instead of 0."
    )
