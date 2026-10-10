"""A dataset without role columns gives every person the default role.

``Simulation.build_from_dataset`` falls back to ``Simulation.default_role``
when a dataset has no ``person_<group>_role`` or ``role`` column. It used to
size that fallback by the number of groups, so ``members_role`` had one entry
per group instead of one per person, and role queries (``nb_persons``,
``sum``, ``project``, ``has_role`` with a role) raised or broadcast a
single group's entry over everyone. Literal default keys keep their existing
meaning: unmatched and compound keys leave members holding no recognized role.
Recognized defaults must fit the role's per-group capacity.

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
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.tools import simulation_dumper

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


@pytest.mark.parametrize("default_role", ["member", "parent"])
@pytest.mark.parametrize(
    "query", ["value_from_person", "unique_role_projector", "partner_projection"]
)
def test_unmatched_default_preserves_unique_role_queries(
    tax_benefit_system, default_role, query
):
    # Review witness: both people belong to 10; household 20 is empty.
    simulation = simulate(
        tax_benefit_system,
        in_memory_dataset(
            {
                "person_id": np.array([1, 2]),
                "household_id": np.array([10, 20]),
                "person_household_id": np.array([10, 10]),
            }
        ),
        default_role=default_role,
    )
    household = simulation.populations["household"]
    values = np.array([100, 200])
    if query == "value_from_person":
        result = household.value_from_person(values, Household.FIRST_PARENT)
    elif query == "unique_role_projector":
        result = household.first_parent.transform(values)
    else:
        result = simulation.persons.value_from_partner(
            values, simulation.persons.household, Household.PARENT
        )
    # Neither literal key is a flattened role: no parent contributes a value.
    assert result.tolist() == [0, 0]


@pytest.mark.parametrize("default_role", ["first_parent", "second_parent"])
def test_unique_default_requires_explicit_roles_when_capacity_exceeded(
    tax_benefit_system, default_role
):
    with pytest.raises(
        ValueError, match="default role.*at most 1.*person_household_role"
    ):
        simulate(
            tax_benefit_system,
            in_memory_dataset(
                {
                    "person_id": np.array([1, 2]),
                    "household_id": np.array([10, 20]),
                    "person_household_id": np.array([10, 10]),
                }
            ),
            default_role=default_role,
        )


def test_unique_default_with_one_person_per_group_supports_projection(
    tax_benefit_system,
):
    simulation = simulate(
        tax_benefit_system,
        in_memory_dataset(
            {
                "person_id": np.array([1, 2]),
                "household_id": np.array([10, 20, 30]),
                "person_household_id": np.array([20, 10]),
            }
        ),
        default_role="first_parent",
    )
    household = simulation.household
    values = np.array([100, 200])
    # ID 10 has value 200; ID 20 has value 100; ID 30 has no parent (0).
    assert household.value_from_person(values, Household.FIRST_PARENT).tolist() == [
        200,
        100,
        0,
    ]
    assert household.first_parent.transform(values).tolist() == [200, 100, 0]
    assert simulation.persons.value_from_partner(
        values, simulation.persons.household, Household.PARENT
    ).tolist() == [0, 0]


@pytest.mark.parametrize("capacity, persons", [(0, 0), (0, 1), (2, 2), (2, 3)])
def test_default_respects_noncompound_role_capacity(capacity, persons):
    unit = build_entity(
        key="unit",
        plural="units",
        label="Unit",
        roles=[{"key": "member", "max": capacity}],
    )
    dataset = in_memory_dataset(
        {
            "person_id": np.arange(persons),
            "unit_id": np.array([10, 20]),
            "person_unit_id": np.full(persons, 10, dtype=int),
        }
    )
    system = TaxBenefitSystem([entities.Person, unit])
    if persons > capacity:
        with pytest.raises(ValueError, match=f"default role.*at most {capacity}"):
            simulate(system, dataset)
    else:
        simulation = simulate(system, dataset)
        assert list(simulation.unit.members_role) == [unit.MEMBER] * persons


@pytest.mark.parametrize(
    "default_role, encoded", [("child", "child"), ("member", "unknown")]
)
def test_dataset_default_roles_survive_dump_restore(
    tax_benefit_system, tmp_path, default_role, encoded
):
    simulation = simulate(
        tax_benefit_system,
        in_memory_dataset(
            {
                "person_id": np.array([1, 2, 3]),
                "household_id": np.array([10, 20]),
                "person_household_id": np.array([10, 10, 20]),
            }
        ),
        default_role=default_role,
    )
    directory = tmp_path / "dump"
    simulation_dumper.dump_simulation(simulation, str(directory))
    stored_roles = np.load(
        directory / "__entities__" / "household" / "members_role.npy"
    )
    assert stored_roles.tolist() == [encoded] * 3
    restored = simulation_dumper.restore_simulation(str(directory), tax_benefit_system)
    np.testing.assert_array_equal(
        restored.household.members_role, simulation.household.members_role
    )
    np.testing.assert_array_equal(
        restored.household.nb_persons(role=Household.CHILD),
        simulation.household.nb_persons(role=Household.CHILD),
    )


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
    # As on master, "member" names no flattened household role.
    assert list(household.members_role) == [0] * 3
    assert household.nb_persons(role=Household.PARENT).tolist() == [0]
    assert household.nb_persons(role=Household.CHILD).tolist() == [0]
    assert household.sum(np.array([1.0, 2.0, 4.0]), role=Household.PARENT).tolist() == [
        # No recognized parents: the filtered sum is 0 + 0 + 0 = 0.
        0.0
    ]
    assert simulation.persons.has_role(Household.FIRST_PARENT).tolist() == [False] * 3


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
    # ID 10 has persons 1 and 4 (2); ID 20 has persons 0, 2 and 3 (3).
    assert household.nb_persons(role=Household.CHILD).tolist() == [2, 3]
    assert household.nb_persons(role=Household.PARENT).tolist() == [0, 0]
    assert household.project(np.array([7.0, 9.0]), role=Household.CHILD).tolist() == [
        9.0,
        7.0,
        9.0,
        9.0,
        7.0,
    ]
    # max(1, 4) = 4 for ID 10; max(0, 2, 3) = 3 for ID 20.
    assert household.max(np.arange(5.0), role=Household.CHILD).tolist() == [4.0, 3.0]


def test_flat_file_gives_one_role_per_person(tax_benefit_system):
    dataframe = pd.DataFrame(
        {
            "person_id": [10, 11, 12, 13],
            "person_household_id": [700, 500, 700, 700],
        }
    )
    simulation = simulate(
        tax_benefit_system,
        Dataset.from_dataframe(dataframe, "2024"),
        default_role="child",
    )
    household = simulation.populations["household"]

    assert len(household.members_role) == 4
    assert household.nb_persons(role=Household.CHILD).tolist() == [1, 3]


@pytest.mark.parametrize(
    "default_role, expected_role",
    [
        ("child", Household.CHILD),
        ("second_parent", Household.SECOND_PARENT),
        ("parent", 0),
        ("member", 0),
    ],
)
def test_default_role_keeps_literal_join_semantics(
    tax_benefit_system, default_role, expected_role
):
    simulation = simulate(
        tax_benefit_system,
        in_memory_dataset(
            {
                "person_id": np.array([1, 2]),
                "household_id": np.array([10, 20]),
                "person_household_id": np.array([20, 10]),
            }
        ),
        default_role=default_role,
    )
    assert list(simulation.household.members_role) == [expected_role] * 2


def test_default_role_keeps_literal_semantics_for_each_entity():
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

    simulation = simulate(
        TaxBenefitSystem([entities.Person, family, unit]),
        in_memory_dataset(
            {
                "person_id": np.array([1, 2]),
                "family_id": np.array([10]),
                "person_family_id": np.array([10, 10]),
                "unit_id": np.array([20]),
                "person_unit_id": np.array([20, 20]),
            }
        ),
        default_role="member",
    )
    assert list(simulation.family.members_role) == [0, 0]
    assert list(simulation.unit.members_role) == [unit.MEMBER] * 2


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


@pytest.mark.parametrize("role_column", ["person_household_role", "role"])
def test_role_column_overrides_the_default(tax_benefit_system, role_column):
    simulation = simulate(
        tax_benefit_system,
        in_memory_dataset(
            {
                "person_id": np.array([1, 2, 3]),
                "household_id": np.array([10]),
                "person_household_id": np.array([10, 10, 10]),
                # Indices into the flattened roles: first_parent, child, child.
                role_column: np.array([0, 2, 2]),
            }
        ),
        # Three people exceed this default's capacity; explicit roles override it.
        default_role="first_parent",
    )
    household = simulation.populations["household"]

    assert household.nb_persons(role=Household.PARENT).tolist() == [1]
    assert household.nb_persons(role=Household.CHILD).tolist() == [2]


@pytest.mark.parametrize("roles", [["first_parent"], [["first_parent"]] * 3, "child"])
def test_join_rejects_invalid_role_shapes(tax_benefit_system, roles):
    builder = SimulationBuilder()
    builder.create_entities(tax_benefit_system)
    builder.declare_person_entity("person", [1, 2, 3])
    household = builder.declare_entity("household", [10])

    with pytest.raises(ValueError, match="household role\\(s\\) for 3 person"):
        builder.join_with_persons(household, [10, 10, 10], roles)
