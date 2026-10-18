"""Flat-file households with explicit roles, for testing what ``subsample`` keeps.

Shared by ``tests/core/test_subsample_explicit_roles.py`` and its Hypothesis
property module, which pytest must be able to skip on its own when
Hypothesis is not installed.

Households are written as tables of role keys, and the expected role of each
person is read from that table, never from the loader: a key that names a
flattened household role (``first_parent``, ``second_parent``, ``child``) is
that role, and any other key (``parent``, ``lodger``) is no role.
"""

from collections import Counter

import numpy as np
import pandas as pd

from policyengine_core.country_template import CountryTaxBenefitSystem, Simulation
from policyengine_core.country_template.entities import Person
from policyengine_core.data import Dataset
from policyengine_core.periods import ETERNITY
from policyengine_core.variables import Variable

DATASET_YEAR = "2024"
ROLE = "person_household_role"
FLATTENED_KEYS = ("first_parent", "second_parent", "child")
# How these helpers report a member the loader gave no role.
NO_ROLE = ""
UNKNOWN_COLUMN_WARNING = "do not match any variable"


class integer_role(Variable):
    value_type = int
    entity = Person
    definition_period = ETERNITY
    label = "Integer person household role"

    def formula(person, period):
        return person.filled_array(2)


integer_role.__name__ = ROLE


def role_system(role_variable: str = "absent") -> CountryTaxBenefitSystem:
    """The template system with its household role variable ``absent``,
    as shipped (``string``), or replaced by an ``integer`` formula."""
    system = CountryTaxBenefitSystem()
    if role_variable == "absent":
        del system.variables[ROLE]
    elif role_variable == "integer":
        system.replace_variable(integer_role)
    else:
        assert role_variable == "string"
    return system


def expected_role(key: str) -> str:
    return key if key in FLATTENED_KEYS else NO_ROLE


def dataframe(households, role_column=ROLE, order=None, weights=None):
    """One row per person: person ``i`` has ID ``100 + i``, household ``h``
    has membership label ``10 * (h + 1)`` and weight ``weights[h]``."""
    keys = [key for household in households for key in household]
    labels = [10 * (h + 1) for h, household in enumerate(households) for _ in household]
    weights = weights or [float(h + 1) for h in range(len(households))]
    columns = {
        f"person_id__{DATASET_YEAR}": [100 + i for i in range(len(keys))],
        f"person_household_id__{DATASET_YEAR}": labels,
        f"household_weight__{DATASET_YEAR}": [
            float(weights[label // 10 - 1]) for label in labels
        ],
    }
    if role_column == ROLE:
        columns[f"{ROLE}__{DATASET_YEAR}"] = keys
    elif role_column is not None:
        columns[role_column] = keys
    data = pd.DataFrame(columns)
    if order is not None:
        data = data.iloc[list(order)].reset_index(drop=True)
    return data


def source_roles(households) -> dict:
    """``{person ID: expected role key}`` from the household table."""
    keys = [key for household in households for key in household]
    return {100 + i: expected_role(key) for i, key in enumerate(keys)}


def source_households(households) -> dict:
    """``{frozenset of person IDs: Counter of expected role keys}``."""
    result, person = {}, 100
    for household in households:
        ids = frozenset(range(person, person + len(household)))
        result[ids] = Counter(expected_role(key) for key in household)
        person += len(household)
    return result


def load_dataframe(
    data, role_variable="absent", default_role="first_parent", system=None
):
    simulation_class = type(
        "RoleSimulation", (Simulation,), {"default_role": default_role}
    )
    return simulation_class(
        tax_benefit_system=system or role_system(role_variable),
        dataset=Dataset.from_dataframe(data, DATASET_YEAR),
    )


def load(
    households,
    role_variable="absent",
    role_column=ROLE,
    default_role="first_parent",
    order=None,
    weights=None,
    system=None,
):
    return load_dataframe(
        dataframe(households, role_column, order, weights),
        role_variable,
        default_role,
        system,
    )


def role_keys(simulation) -> np.ndarray:
    household = simulation.populations["household"]
    roles = np.broadcast_to(
        np.asarray(household.members_role, dtype=object), simulation.persons.count
    )
    return np.array([getattr(role, "key", NO_ROLE) for role in roles], dtype=object)


def roles_by_person(simulation) -> dict:
    return dict(zip(simulation.persons.ids.tolist(), role_keys(simulation).tolist()))


def households_by_members(simulation) -> dict:
    """``{frozenset of person IDs: Counter of role keys}`` per household."""
    household = simulation.populations["household"]
    keys = role_keys(simulation)
    result = {}
    for index in range(household.count):
        members = household.members_entity_id == index
        ids = frozenset(simulation.persons.ids[members].tolist())
        result[ids] = Counter(keys[members].tolist())
    return result


def role_counts(simulation) -> dict:
    """``{role key: nb_persons(role) per household}``, from the projector."""
    household = simulation.populations["household"]
    return {
        role.key: household.nb_persons(role).tolist()
        for role in household.entity.flattened_roles
    }


def assert_roles_kept(simulation, households):
    """Every kept household is a whole source household with its roles."""
    source = source_households(households)
    kept = households_by_members(simulation)
    assert kept, "subsample kept no household"
    for members, roles in kept.items():
        assert members in source, f"{sorted(members)} is not a source household"
        assert roles == source[members]
    expected = source_roles(households)
    assert roles_by_person(simulation) == {
        person: expected[person] for person in simulation.persons.ids.tolist()
    }
    # The projector counts the same roles the members hold.
    household_order = [
        frozenset(
            simulation.persons.ids[
                simulation.household.members_entity_id == index
            ].tolist()
        )
        for index in range(simulation.household.count)
    ]
    for key, counts in role_counts(simulation).items():
        assert counts == [kept[members][key] for members in household_order]
