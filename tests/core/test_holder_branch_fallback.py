"""Regression tests for Holder.get_array branch fallback behaviour (C1, H2).

Previously ``Holder.get_array`` would silently return the value stored under
*any* branch when the requested branch had no value for the period — simply
picking the first branch from dict-insertion order. That meant a call for the
baseline branch could get the reform's value (or vice versa) and reform
deltas came out silently wrong.

The corrected behaviour: fall back to the ``default`` branch only; if that has
nothing either, return ``None`` so the caller (``_calculate``) runs the
formula on the requested branch.

H2 is the related bug where ``simulation.py`` called ``holder.get_array``
without passing ``branch_name`` in the auto-carry-over path, short-circuiting
to ``"default"``.
"""

from __future__ import annotations

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import situation_examples
from policyengine_core.simulations import SimulationBuilder


def _build_single(tax_benefit_system):
    return SimulationBuilder().build_from_entities(
        tax_benefit_system, situation_examples.single
    )


def test_get_array_does_not_leak_values_across_branches(tax_benefit_system):
    """A value set on one branch must NOT be returned for another branch."""
    sim = _build_single(tax_benefit_system)
    holder = sim.person.get_holder("salary")
    period = periods.period("2017-01")

    # Put a value on the "reform" branch only.
    holder._memory_storage.put(np.asarray([12345.0]), period, "reform")

    # Asking for the "baseline" branch must NOT return the reform value.
    # The previous implementation returned the first available branch
    # (reform, here) because it iterated insertion order.
    result = holder.get_array(period, "baseline")
    assert result is None, (
        "get_array must not fall back to an arbitrary branch; that caused "
        "cross-branch contamination (bug C1)."
    )


def test_get_array_falls_back_to_default_branch(tax_benefit_system):
    """If the ``default`` branch has a value, it should be returned."""
    sim = _build_single(tax_benefit_system)
    holder = sim.person.get_holder("salary")
    period = periods.period("2017-01")

    holder._memory_storage.put(np.asarray([42.0]), period, "default")
    # Asking for a non-existent branch should fall back to default's value.
    result = holder.get_array(period, "baseline")
    assert result is not None
    assert result[0] == 42.0
