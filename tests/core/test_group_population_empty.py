"""Group reductions and ranks when a simulation has no persons.

``GroupPopulation.reduce`` (behind ``all``, ``max`` and ``min``) returns its
neutral element for a group with no members, and ``sum`` and ``nb_persons``
return 0. With no persons at all, ``reduce``, ``members_position`` and
``Population.get_rank`` took ``numpy.max`` of an empty array and raised
"zero-size array to reduction operation maximum which has no identity".
Memberships given as an empty list, or built for no one, were also float,
which NumPy cannot index with.
"""

from __future__ import annotations

import numpy as np

from policyengine_core.entities import build_entity
from policyengine_core.populations import GroupPopulation, Population
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.taxbenefitsystems import TaxBenefitSystem

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


def test_no_members_given_as_a_plain_empty_list():
    # ``[]`` and ``numpy.array([])`` are float; NumPy indexes only with
    # integers, so ranks and projections raised IndexError.
    persons, households = populations([], 1)
    households.members_entity_id = []

    assert households.members_position.dtype.kind == "i"
    assert persons.get_rank(households, np.array([], dtype=float)).tolist() == []
    assert households.project(np.array([5.0])).tolist() == []
    assert households.max(np.array([], dtype=float)).tolist() == [-np.inf]


def test_default_simulation_of_no_one():
    simulation = SimulationBuilder().build_default_simulation(
        TaxBenefitSystem([PERSON, HOUSEHOLD]), count=0
    )
    persons = simulation.persons
    households = simulation.populations["household"]
    no_values = np.array([], dtype=float)

    assert households.ids.dtype.kind == "i"
    assert persons.get_rank(households, no_values).tolist() == []
    assert households.value_nth_person(0, no_values).tolist() == []
    assert households.project(no_values).tolist() == []
    assert persons.household.max(no_values).tolist() == []
    assert households.all(no_values > 0).tolist() == []
