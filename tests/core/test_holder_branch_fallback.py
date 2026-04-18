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


def test_get_array_falls_back_through_parent_branch_chain(tax_benefit_system):
    """Nested branches must inherit values from their parent branch.

    ``policyengine-us`` uses a two-level branch pattern:

    1. ``tax_liability_if_itemizing`` creates an ``itemizing`` branch from
       the default simulation and calls ``branch.set_input("tax_unit_itemizes", True)``.
    2. Calculating ``income_tax`` on that branch reaches
       ``ctc_limiting_tax_liability``, which creates a ``no_salt`` sub-branch
       from the ``itemizing`` branch and calls
       ``no_salt.calculate("income_tax_before_credits")``.

    The ``no_salt`` branch must see ``tax_unit_itemizes=True`` inherited
    from its parent ``itemizing`` branch — otherwise ``tax_unit_itemizes``
    re-runs its formula on ``no_salt``, which calls
    ``tax_liability_if_itemizing`` again, creating a circular definition
    / infinite recursion. Surfaced in ``PolicyEngine/policyengine-us#8058``.
    """
    sim = _build_single(tax_benefit_system)
    itemizing_branch = sim.get_branch("itemizing")
    no_salt_branch = itemizing_branch.get_branch("no_salt")

    holder = no_salt_branch.person.get_holder("salary")
    period = periods.period("2017-01")

    # Simulate ``itemizing_branch.set_input("salary", ...)``: the storage
    # key lives under the ``itemizing`` branch name. The cloned ``no_salt``
    # holder starts with the same storage dict because ``Population.clone``
    # deep-copies ``_arrays`` from the source.
    holder._memory_storage.put(np.asarray([7_777.0]), period, "itemizing")

    # Asking the ``no_salt`` branch for this value must walk up the
    # ``parent_branch`` chain and return the itemizing branch's value.
    result = holder.get_array(period, "no_salt")
    assert result is not None, (
        "get_array on a nested branch must fall back through parent_branch "
        "to the ancestor that actually has the value"
    )
    assert result[0] == 7_777.0


def test_group_population_clone_sets_holder_simulation_to_clone(tax_benefit_system):
    """``GroupPopulation.clone`` must point holders at the cloned simulation.

    Previously ``GroupPopulation.clone`` called ``holder.clone(self)``
    (the *source* population), so every cloned holder's
    ``.simulation`` reference pointed back at the source simulation. That
    broke branch-aware lookups: the holder thought it belonged to the
    parent branch even when the clone was a nested branch, so
    ``parent_branch`` walks started from the wrong simulation and missed
    the ancestor's inputs.
    """
    sim = _build_single(tax_benefit_system)
    branch = sim.get_branch("nested")

    # Find a group-entity variable (household-level).
    household = branch.household
    holder = household.get_holder("housing_tax")

    assert holder.simulation is branch, (
        "GroupPopulation.clone must pass the CLONED population to "
        "holder.clone so holder.simulation points at the new branch, "
        "not the source simulation"
    )
    assert holder.simulation.branch_name == "nested"
