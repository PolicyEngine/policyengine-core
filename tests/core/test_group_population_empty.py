"""Group reductions and ranks when a simulation has no persons.

``GroupPopulation.reduce`` (behind ``all``, ``max`` and ``min``) returns its
neutral element for a group with no members, and ``sum`` and ``nb_persons``
return 0. With no persons at all, ``reduce``, ``members_position`` and
``Population.get_rank`` took ``numpy.max`` of an empty array and raised
"zero-size array to reduction operation maximum which has no identity".
"""

from __future__ import annotations

import numpy as np

from policyengine_core.entities import build_entity
from policyengine_core.populations import GroupPopulation, Population

PERSON = build_entity("person", "persons", "Person", is_person=True)
HOUSEHOLD = build_entity(
    "household",
    "households",
    "Household",
    roles=[{"key": "member", "plural": "members"}],
)


def populations(members_entity_id, count):
    persons = Population(PERSON)
    persons.count = len(members_entity_id)
    households = GroupPopulation(HOUSEHOLD, persons)
    households.count = count
    households.members_entity_id = np.asarray(members_entity_id, dtype=int)
    return persons, households


def test_reductions_over_no_persons_give_neutral_elements():
    _, households = populations([], 1)
    no_values = np.array([], dtype=float)

    assert households.all(no_values > 0).tolist() == [True]
    assert households.any(no_values > 0).tolist() == [False]
    assert households.max(no_values).tolist() == [-np.inf]
    assert households.min(no_values).tolist() == [np.inf]
    assert households.sum(no_values).tolist() == [0]
    assert households.nb_persons().tolist() == [0]
    assert households.members_position.tolist() == []


def test_rank_over_no_persons_is_empty():
    persons, households = populations([], 2)

    assert persons.get_rank(households, np.array([], dtype=float)).tolist() == []
