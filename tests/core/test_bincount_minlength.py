"""Regression tests for GroupPopulation.sum / .nb_persons with empty last entity (H4).

When the highest-indexed entity has zero members, ``numpy.bincount`` without
``minlength`` returns an array shorter than the number of entities, which
silently misaligns downstream broadcasting. Both the role-less ``sum`` branch
and ``nb_persons`` without a role were missing ``minlength``.
"""

from __future__ import annotations

from typing import Iterable

import numpy

from policyengine_core.simulations import SimulationBuilder


def test_sum_when_last_entity_has_no_members(tax_benefit_system):
    """A household with no persons must still appear in ``sum`` output."""
    persons_ids: Iterable = [1, 2, 3]
    # Three households exist, but only the first two have persons.
    households_ids: Iterable = ["a", "b", "c"]
    persons_households: Iterable = ["a", "a", "b"]

    builder = SimulationBuilder()
    builder.create_entities(tax_benefit_system)
    builder.declare_person_entity("person", persons_ids)
    household = builder.declare_entity("household", households_ids)
    builder.join_with_persons(household, persons_households, ["first_parent"] * 3)

    # ``sum`` (no role): the last household (index 2) has no members, so
    # ``numpy.bincount(...)`` without ``minlength`` would return a length-2
    # array and break broadcasting against the 3-element entity array.
    result = household.sum(numpy.asarray([1.0, 1.0, 1.0]))
    assert len(result) == household.count == 3
    assert result.tolist() == [2.0, 1.0, 0.0]


def test_nb_persons_when_last_entity_has_no_members(tax_benefit_system):
    persons_ids: Iterable = [1, 2, 3]
    households_ids: Iterable = ["a", "b", "c"]
    persons_households: Iterable = ["a", "a", "b"]

    builder = SimulationBuilder()
    builder.create_entities(tax_benefit_system)
    builder.declare_person_entity("person", persons_ids)
    household = builder.declare_entity("household", households_ids)
    builder.join_with_persons(household, persons_households, ["first_parent"] * 3)

    # ``nb_persons`` without a role goes through the no-role branch of
    # ``bincount`` on line 263.
    counts = household.nb_persons()
    assert len(counts) == household.count == 3
    assert counts.tolist() == [2, 1, 0]
