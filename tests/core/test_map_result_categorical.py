"""Strings and enums map to a group as the value its members share."""

import numpy as np
import pandas as pd
import pytest

from policyengine_core.data import Dataset
from policyengine_core.entities import build_entity
from policyengine_core.enums import Enum, EnumArray
from policyengine_core.periods import YEAR
from policyengine_core.simulations import Simulation
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.variables import Variable

Person = build_entity(key="person", plural="persons", label="Person", is_person=True)
Household = build_entity(
    key="household",
    plural="households",
    label="Household",
    roles=[{"key": "member", "plural": "members", "label": "Members"}],
)
Family = build_entity(
    key="family",
    plural="families",
    label="Family",
    roles=[{"key": "member", "plural": "members", "label": "Members"}],
)


class Region(Enum):
    NORTH = "North"
    SOUTH = "South"


def variable(name, entity, value_type, **attributes):
    return type(
        name,
        (Variable,),
        {
            "value_type": value_type,
            "entity": entity,
            "definition_period": YEAR,
            "label": name,
            **attributes,
        },
    )


def build(households, families):
    system = TaxBenefitSystem([Person, Household, Family])
    system.add_variable(variable("amount", Household, float))
    system.add_variable(variable("place", Household, str))
    system.add_variable(variable("name", Person, str))
    system.add_variable(variable("person_amount", Person, float))
    system.add_variable(
        variable(
            "region",
            Household,
            Enum,
            possible_values=Region,
            default_value=Region.NORTH,
        )
    )
    return Simulation(
        tax_benefit_system=system,
        dataset=Dataset.from_dataframe(
            pd.DataFrame({"household_id": households, "family_id": families}), 2024
        ),
    )


def two_families_in_one_household():
    simulation = build([1, 1, 1, 2], [1, 1, 2, 3])
    simulation.set_input("place", 2024, np.array(["A", "B"], dtype=object))
    simulation.set_input("name", 2024, np.array(["x", "x", "y", "z"], dtype=object))
    simulation.set_input(
        "person_amount", 2024, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )
    simulation.set_input("amount", 2024, np.array([30.0, 40.0], dtype=np.float32))
    simulation.set_input(
        "region", 2024, np.array([Region.SOUTH, Region.NORTH], dtype=object)
    )
    return simulation


def test_a_group_string_maps_across_groups():
    simulation = two_families_in_one_household()
    result = simulation.calculate("place", 2024, map_to="family")
    assert list(result) == ["A", "A", "B"]


def test_a_group_string_maps_to_a_group_its_members_share():
    simulation = build([1, 1, 2, 2], [1, 2, 3, 3])
    simulation.set_input("name", 2024, np.array(["x", "y", "z", "z"], dtype=object))
    result = simulation.calculate("name", 2024, map_to="family")
    assert list(result) == ["x", "y", "z"]


def test_an_enum_maps_across_groups_and_stays_an_enum():
    simulation = two_families_in_one_household()
    result = simulation.calculate("region", 2024, map_to="family")
    assert isinstance(result, EnumArray)
    assert list(result.decode_to_str()) == ["SOUTH", "SOUTH", "NORTH"]


def test_a_decoded_enum_maps_across_groups():
    simulation = two_families_in_one_household()
    result = simulation.calculate("region", 2024, map_to="family", decode_enums=True)
    assert list(result) == ["SOUTH", "SOUTH", "NORTH"]


def test_a_person_string_that_differs_within_a_group_says_so():
    simulation = two_families_in_one_household()
    with pytest.raises(ValueError, match="1 household group"):
        simulation.calculate("name", 2024, map_to="household")


def test_a_group_string_that_differs_within_a_target_group_says_so():
    simulation = build([1, 2, 2], [1, 1, 2])
    simulation.set_input("place", 2024, np.array(["A", "B"], dtype=object))
    with pytest.raises(ValueError, match="different values"):
        simulation.calculate("place", 2024, map_to="family")


def test_a_group_string_still_maps_to_people_and_itself():
    simulation = two_families_in_one_household()
    assert list(simulation.calculate("place", 2024, map_to="person")) == [
        "A",
        "A",
        "A",
        "B",
    ]
    assert list(simulation.calculate("place", 2024, map_to="household")) == ["A", "B"]


def test_amounts_still_sum_and_share():
    simulation = two_families_in_one_household()
    assert simulation.calculate("amount", 2024, map_to="family").tolist() == [
        20.0,
        10.0,
        40.0,
    ]
    assert simulation.calculate("person_amount", 2024, map_to="household").tolist() == [
        6.0,
        4.0,
    ]


def test_an_explicit_how_keeps_its_meaning():
    simulation = two_families_in_one_household()
    values = np.array(["x", "x", "y", "z"], dtype=object)
    assert list(
        simulation.map_result(
            values, "person", "household", how="value_from_first_person"
        )
    ) == ["x", "z"]
