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
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.simulations.simulation_builder import group_positions


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
