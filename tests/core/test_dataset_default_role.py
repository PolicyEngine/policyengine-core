"""A dataset without role columns gives every person the default role.

``Simulation.build_from_dataset`` falls back to ``Simulation.default_role``
when a dataset has no ``person_<group>_role`` or ``role`` column. It used to
size that fallback by the number of groups, so ``members_role`` had one entry
per group instead of one per person, and role queries (``nb_persons``,
``sum``, ``project``, ``has_role`` with a role) raised or broadcast a
single group's entry over everyone. A default that named no role of the
entity (``"member"`` in a household of parents and children) also left
everyone holding no role at all.

``test_dataset_default_role_property.py`` checks the same invariants for
random datasets, group entities and default roles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyengine_core.country_template import entities
from policyengine_core.data import Dataset
from policyengine_core.entities import build_entity
from policyengine_core.periods import ETERNITY
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.simulations.simulation import _default_role_key

Household = entities.Household


def in_memory_dataset(data, data_format=Dataset.ARRAYS, time_period="2024"):
    """A dataset whose ``load`` returns ``data``, like ``Dataset.from_dataframe``."""
    return type(
        "InMemoryDataset",
        (Dataset,),
        {
            "name": "in_memory",
            "label": "In-memory dataset",
            "data_format": data_format,
            "file_path": "in_memory",
            "time_period": time_period,
            "load": lambda self: data,
        },
    )()


def simulate(tax_benefit_system, dataset, default_role=Simulation.default_role):
    simulation_class = type(
        "DefaultRoleSimulation", (Simulation,), {"default_role": default_role}
    )
    return simulation_class(tax_benefit_system=tax_benefit_system, dataset=dataset)


def test_one_household_of_three_gives_three_roles(tax_benefit_system):
    # The audit witness: three persons in one household, no role column.
    simulation = simulate(
        tax_benefit_system,
        in_memory_dataset(
            {
                "person_id": np.array([1, 2, 3]),
                "household_id": np.array([10]),
                "person_household_id": np.array([10, 10, 10]),
            }
        ),
    )
    household = simulation.populations["household"]

    assert len(household.members_role) == 3
    # "member" names no household role, so persons take the first one.
    assert list(household.members_role) == [Household.FIRST_PARENT] * 3
    assert household.nb_persons(role=Household.PARENT).tolist() == [3]
    assert household.nb_persons(role=Household.CHILD).tolist() == [0]
    assert household.sum(np.array([1.0, 2.0, 4.0]), role=Household.PARENT).tolist() == [
        7.0
    ]
    assert simulation.persons.has_role(Household.FIRST_PARENT).tolist() == [True] * 3


def test_saved_dataset_gives_one_role_per_person(tax_benefit_system, tmp_path):
    # The same failure through an HDF5 file, with two households of
    # different sizes so no shape broadcasts by accident.
    class TwoHouseholds(Dataset):
        name = "two_households"
        label = "Two households"
        file_path = tmp_path / "two_households.h5"
        data_format = Dataset.TIME_PERIOD_ARRAYS

    TwoHouseholds().save_dataset(
        {
            "person_id": {ETERNITY: np.array([1, 2, 3, 4, 5])},
            "household_id": {ETERNITY: np.array([10, 20])},
            "person_household_id": {ETERNITY: np.array([20, 10, 20, 20, 10])},
        }
    )
    simulation = simulate(tax_benefit_system, TwoHouseholds(), default_role="child")
    household = simulation.populations["household"]

    assert list(household.members_role) == [Household.CHILD] * 5
    assert household.nb_persons(role=Household.CHILD).tolist() == [2, 3]
    assert household.nb_persons(role=Household.PARENT).tolist() == [0, 0]
    assert household.project(np.array([7.0, 9.0]), role=Household.CHILD).tolist() == [
        9.0,
        7.0,
        9.0,
        9.0,
        7.0,
    ]
    assert household.max(np.arange(5.0), role=Household.CHILD).tolist() == [4.0, 3.0]


def test_flat_file_gives_one_role_per_person(tax_benefit_system):
    dataframe = pd.DataFrame(
        {
            "person_id": [10, 11, 12, 13],
            "person_household_id": [700, 500, 700, 700],
        }
    )
    simulation = simulate(tax_benefit_system, Dataset.from_dataframe(dataframe, "2024"))
    household = simulation.populations["household"]

    assert len(household.members_role) == 4
    assert household.nb_persons(role=Household.PARENT).tolist() == [1, 3]


@pytest.mark.parametrize(
    "default_role, expected",
    [
        ("child", "child"),
        ("second_parent", "second_parent"),
        # A role with subroles is held through its first subrole.
        ("parent", "first_parent"),
        # A key that names no household role falls back to the first role.
        ("member", "first_parent"),
    ],
)
def test_default_role_key_names_a_role_of_the_entity(default_role, expected):
    assert _default_role_key(Household, default_role) == expected


def test_default_role_key_resolves_against_each_entity():
    # One ``default_role`` serves every group entity; a role with subroles is
    # matched even when it is not the entity's first role.
    family = build_entity(
        key="family",
        plural="families",
        label="Family",
        roles=[
            {"key": "child", "plural": "children"},
            {"key": "parent", "plural": "parents", "subroles": ["mother", "father"]},
        ],
    )
    unit = build_entity(
        key="unit",
        plural="units",
        label="Unit",
        roles=[{"key": "member", "plural": "members"}],
    )

    assert _default_role_key(family, "parent") == "mother"
    assert _default_role_key(family, "member") == "child"
    assert _default_role_key(unit, "parent") == "member"


def test_no_default_role_still_requires_a_role_column(tax_benefit_system):
    with pytest.raises(ValueError, match="person_household_role"):
        simulate(
            tax_benefit_system,
            in_memory_dataset(
                {
                    "person_id": np.array([1, 2]),
                    "household_id": np.array([10]),
                    "person_household_id": np.array([10, 10]),
                }
            ),
            default_role=None,
        )


def test_role_column_overrides_the_default(tax_benefit_system):
    simulation = simulate(
        tax_benefit_system,
        in_memory_dataset(
            {
                "person_id": np.array([1, 2, 3]),
                "household_id": np.array([10]),
                "person_household_id": np.array([10, 10, 10]),
                # Indices into the flattened roles: first_parent, child, child.
                "person_household_role": np.array([0, 2, 2]),
            }
        ),
    )
    household = simulation.populations["household"]

    assert household.nb_persons(role=Household.PARENT).tolist() == [1]
    assert household.nb_persons(role=Household.CHILD).tolist() == [2]


def test_join_rejects_one_role_per_group(tax_benefit_system):
    builder = SimulationBuilder()
    builder.create_entities(tax_benefit_system)
    builder.declare_person_entity("person", [1, 2, 3])
    household = builder.declare_entity("household", [10])

    with pytest.raises(ValueError, match="Got 1 household role\\(s\\) for 3 person"):
        builder.join_with_persons(household, [10, 10, 10], ["first_parent"])
