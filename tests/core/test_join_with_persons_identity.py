"""Joining persons to groups keeps each person in the group their ID names.

``SimulationBuilder.join_with_persons`` turns each person's group ID into
``members_entity_id``, the position of that group among the declared IDs.
Empty groups may sit anywhere in the declared order (``test_bincount_minlength``
covers one at the end), so the join must look each ID up among all declared
IDs, not rank it among the occupied ones. Ranking moved a person in group
``"b"`` into group ``"a"`` whenever ``"a"`` had no members.

``test_join_with_persons_identity_property.py`` checks the same identity for
random IDs, declared orders, empty groups and assignments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyengine_core.data import Dataset
from policyengine_core.entities import build_entity
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.simulations.simulation_builder import group_positions
from policyengine_core.taxbenefitsystems import TaxBenefitSystem


def _join(tax_benefit_system, group_ids, persons_group_ids):
    builder = SimulationBuilder()
    builder.create_entities(tax_benefit_system)
    builder.declare_person_entity("person", list(range(len(persons_group_ids))))
    household = builder.declare_entity("household", group_ids)
    builder.join_with_persons(
        household, persons_group_ids, ["first_parent"] * len(persons_group_ids)
    )
    return household


def test_person_joins_named_group_when_earlier_group_is_empty(tax_benefit_system):
    # The audit witness: group "a" is empty and sorts before group "b".
    household = _join(tax_benefit_system, ["a", "b"], ["b"])

    assert household.members_entity_id.tolist() == [1]
    assert household.sum(np.array([100.0])).tolist() == [0.0, 100.0]
    assert household.project(np.array([10.0, 20.0])).tolist() == [20.0]


def test_person_joins_named_group_around_an_interior_empty_group(
    tax_benefit_system,
):
    household = _join(tax_benefit_system, ["c", "a", "b"], ["c", "a", "a"])

    assert household.members_entity_id.tolist() == [0, 1, 1]
    assert household.nb_persons().tolist() == [1, 2, 0]


def test_join_rejects_an_undeclared_group(tax_benefit_system):
    with pytest.raises(ValueError, match="not declared: \\['z'\\]"):
        _join(tax_benefit_system, ["a", "b"], ["a", "z"])


def test_join_rejects_repeated_group_ids(tax_benefit_system):
    with pytest.raises(ValueError, match="must be unique"):
        _join(tax_benefit_system, ["a", "b", "a"], ["a", "b"])


def test_group_positions_of_no_persons_is_empty():
    assert group_positions(["a", "b"], []).tolist() == []
    assert group_positions([], []).tolist() == []


def test_dataset_person_joins_named_household_when_another_is_empty(
    tax_benefit_system, tmp_path
):
    # Household 20 is declared but empty; the person in household 30 must
    # see household 30's rent, not household 20's.
    class InteriorEmptyHousehold(Dataset):
        name = "interior_empty_household"
        label = "Interior empty household"
        file_path = tmp_path / "interior_empty_household.h5"
        data_format = Dataset.ARRAYS
        time_period = "2024"

    InteriorEmptyHousehold().save_dataset(
        {
            "person_id": np.array([1, 2]),
            "household_id": np.array([10, 20, 30]),
            "person_household_id": np.array([10, 30]),
            # HDF5 stores no unicode strings, so give roles by index.
            "person_household_role": np.array([0, 0]),
        }
    )
    simulation = Simulation(
        tax_benefit_system=tax_benefit_system,
        dataset=InteriorEmptyHousehold(),
    )
    simulation.set_input("rent", "2024-01", np.array([100.0, 200.0, 300.0]))

    household = simulation.populations["household"]
    assert household.members_entity_id.tolist() == [0, 2]
    assert simulation.calculate("rent", "2024-01", map_to="person").tolist() == [
        100.0,
        300.0,
    ]


def test_flat_file_numbers_households_by_their_members(tax_benefit_system):
    # A flat file has one row per person; households 500 and 700 become
    # households 0 and 1, in order of their IDs.
    dataframe = pd.DataFrame(
        {
            "person_id__2024": [10, 11, 12],
            "person_household_id__2024": [700, 500, 700],
            "salary__2024-01": [1.0, 2.0, 4.0],
        }
    )
    simulation = Simulation(
        tax_benefit_system=tax_benefit_system,
        dataset=Dataset.from_dataframe(dataframe, "2024"),
    )

    household = simulation.populations["household"]
    assert household.ids.tolist() == [0, 1]
    assert household.members_entity_id.tolist() == [1, 0, 1]
    assert simulation.calculate("salary", "2024-01", map_to="household").tolist() == [
        2.0,
        5.0,
    ]


def test_flat_file_with_household_id_column_counts_its_households(
    tax_benefit_system,
):
    # A ``household_id`` column without a period suffix used to leave the
    # previous entity's IDs declared: three persons made three households.
    dataframe = pd.DataFrame(
        {
            "person_id": [10, 11, 12],
            "household_id": [500, 500, 700],
            "person_household_id": [500, 500, 700],
            "salary__2024-01": [1.0, 2.0, 4.0],
        }
    )
    simulation = Simulation(
        tax_benefit_system=tax_benefit_system,
        dataset=Dataset.from_dataframe(dataframe, "2024"),
    )

    household = simulation.populations["household"]
    assert household.count == 2
    assert household.members_entity_id.tolist() == [0, 0, 1]


@pytest.mark.parametrize(
    "group_ids, persons_group_ids, expected",
    [
        # Converting both to float64 rounds 2**53 + 3 up to 2**53 + 4, so the
        # person matched the first household.
        (
            np.array([2**53 + 3, 2**53 + 4], dtype=np.int64),
            np.array([2**53 + 4], dtype=np.float64),
            [1],
        ),
        # int64 and uint64 meet as float64, where 2**53 + 1 becomes 2**53.
        (
            np.array([2**53, 2**53 + 1], dtype=np.int64),
            np.array([2**53, 2**53 + 1], dtype=np.uint64),
            [0, 1],
        ),
        (np.array([5, 7], dtype=np.int32), np.array([7, 5, 7]), [1, 0, 1]),
    ],
)
def test_ids_of_different_dtypes_match_by_value(group_ids, persons_group_ids, expected):
    assert group_positions(group_ids, persons_group_ids).tolist() == expected


def test_flat_file_with_only_a_household_id_column_reads_it_as_membership(
    tax_benefit_system,
):
    # With no ``person_household_id`` column, each row's ``household_id`` is
    # its person's household.
    dataframe = pd.DataFrame({"person_id": [0, 1, 2], "household_id": [7, 7, 9]})
    simulation = Simulation(
        tax_benefit_system=tax_benefit_system,
        dataset=Dataset.from_dataframe(dataframe, "2024"),
    )

    household = simulation.populations["household"]
    assert household.count == 2
    assert household.members_entity_id.tolist() == [0, 0, 1]


def test_flat_file_default_roles_cover_every_person():
    # Two persons in one household need two default roles: counting one
    # per household gave role-filtered sums a one-entry filter.
    person = build_entity("person", "persons", "", is_person=True)
    household = build_entity(
        "household", "households", "", roles=[{"key": "member", "plural": "members"}]
    )
    dataframe = pd.DataFrame(
        {
            "person_id": [10, 11],
            "household_id": [500, 500],
            "person_household_id": [500, 500],
        }
    )
    simulation = Simulation(
        tax_benefit_system=TaxBenefitSystem([person, household]),
        dataset=Dataset.from_dataframe(dataframe, "2024"),
    )

    population = simulation.populations["household"]
    assert len(population.members_role) == 2
    assert population.sum(
        np.array([1.0, 2.0]), role=population.entity.MEMBER
    ).tolist() == [3.0]


@pytest.mark.parametrize(
    "id_columns",
    [
        {
            "person_id__2024": [0, 1, 2],
            "household_id__2024": [9, 7, 8],
            "person_household_id__2024": [9, 7, 8],
        },
        {"person_id": [0, 1, 2], "household_id": [9, 7, 8]},
    ],
)
def test_flat_file_group_values_follow_membership_not_row_order(
    tax_benefit_system, id_columns
):
    # One person per household, IDs not in row order. A household column
    # as long as the household count was taken as already per household,
    # in row order, while households are numbered in ID order: each person
    # read another person's rent.
    dataframe = pd.DataFrame({**id_columns, "rent__2024-01": [900.0, 700.0, 800.0]})
    simulation = Simulation(
        tax_benefit_system=tax_benefit_system,
        dataset=Dataset.from_dataframe(dataframe, "2024"),
    )

    assert simulation.calculate("rent", "2024-01", map_to="person").tolist() == [
        900.0,
        700.0,
        800.0,
    ]
    assert simulation.calculate("household_id", "2024", map_to="person").tolist() == [
        9,
        7,
        8,
    ]


def test_flat_file_rejects_missing_memberships(tax_benefit_system):
    # np.unique counts every NaN as one value, which made one household of
    # everyone with no membership.
    dataframe = pd.DataFrame(
        {"person_id": [0, 1, 2], "person_household_id": [1.0, np.nan, np.nan]}
    )
    with pytest.raises(ValueError, match="2 person\\(s\\) have no"):
        Simulation(
            tax_benefit_system=tax_benefit_system,
            dataset=Dataset.from_dataframe(dataframe, "2024"),
        )


def test_dataset_default_roles_cover_every_person(tmp_path):
    # Every dataset format with no role column got one default role per
    # group: three persons in two households had two roles, and a
    # role-filtered sum raised IndexError.
    person = build_entity("person", "persons", "", is_person=True)
    household = build_entity(
        "household", "households", "", roles=[{"key": "member", "plural": "members"}]
    )

    class NoRoleColumn(Dataset):
        name = "no_role_column"
        label = "No role column"
        file_path = tmp_path / "no_role_column.h5"
        data_format = Dataset.ARRAYS
        time_period = "2024"

    NoRoleColumn().save_dataset(
        {
            "person_id": np.array([1, 2, 3]),
            "household_id": np.array([10, 20]),
            "person_household_id": np.array([10, 20, 20]),
        }
    )
    simulation = Simulation(
        tax_benefit_system=TaxBenefitSystem([person, household]),
        dataset=NoRoleColumn(),
    )

    population = simulation.populations["household"]
    assert len(population.members_role) == 3
    assert population.sum(
        np.array([1.0, 2.0, 2.0]), role=population.entity.MEMBER
    ).tolist() == [1.0, 4.0]
