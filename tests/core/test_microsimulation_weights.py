import numpy as np
import pandas as pd

from policyengine_core.country_template import Microsimulation
from policyengine_core.data import Dataset
from policyengine_core.entities import build_entity
from policyengine_core.model_api import ETERNITY, MONTH, YEAR, Variable
from policyengine_core.simulations import Microsimulation as CoreMicrosimulation
from policyengine_core.taxbenefitsystems import TaxBenefitSystem


def _weighted_dataset(include_person_weight: bool = True) -> Dataset:
    data = {
        "person_id__2022": [0, 1, 2],
        "person_household_id__2022": [0, 0, 1],
        "person_household_role__2022": ["parent", "child", "parent"],
        "household_weight__2022": [10.0, 10.0, 20.0],
        "salary__2022-01": [100.0, 200.0, 300.0],
    }
    if include_person_weight:
        data["person_weight__2022"] = [1.0, 2.0, 3.0]
    return Dataset.from_dataframe(pd.DataFrame(data), "2022")


def test__given_stale_person_weight__then_get_weights_uses_household_weight():
    # Given
    simulation = Microsimulation(dataset=_weighted_dataset())

    # When
    weights = simulation.get_weights("salary", "2022-01")

    # Then
    np.testing.assert_array_equal(weights, np.array([10.0, 10.0, 20.0]))


def test__given_no_person_weight_variable__then_get_weights_uses_household_weight(
    isolated_tax_benefit_system,
):
    # Given
    del isolated_tax_benefit_system.variables["person_weight"]
    simulation = Microsimulation(
        tax_benefit_system=isolated_tax_benefit_system,
        dataset=_weighted_dataset(include_person_weight=False),
    )

    # When
    weights = simulation.get_weights("salary", "2022-01")

    # Then
    np.testing.assert_array_equal(weights, np.array([10.0, 10.0, 20.0]))


def test__given_dataframe_mapped_to_household__then_weights_are_household_weights():
    # Given
    simulation = Microsimulation(dataset=_weighted_dataset())

    # When
    dataframe = simulation.calculate_dataframe(
        ["salary"],
        "2022-01",
        map_to="household",
    )

    # Then
    np.testing.assert_array_equal(dataframe.weights, np.array([10.0, 20.0]))


Family = build_entity(
    key="family",
    plural="families",
    label="Family",
    roles=[{"key": "member", "plural": "members", "label": "Members"}],
)

Household = build_entity(
    key="household",
    plural="households",
    label="Household",
    roles=[{"key": "member", "plural": "members", "label": "Members"}],
)

Person = build_entity(
    key="person",
    plural="persons",
    label="Person",
    is_person=True,
)


class FamilyWeightTaxBenefitSystem(TaxBenefitSystem):
    entities = [Family, Household, Person]


class person_id(Variable):
    value_type = int
    entity = Person
    definition_period = ETERNITY
    label = "Person ID"


class person_family_id(Variable):
    value_type = int
    entity = Person
    definition_period = ETERNITY
    label = "Person family ID"


class person_family_role(Variable):
    value_type = str
    entity = Person
    definition_period = ETERNITY
    label = "Person family role"


class person_household_id(Variable):
    value_type = int
    entity = Person
    definition_period = ETERNITY
    label = "Person household ID"


class person_household_role(Variable):
    value_type = str
    entity = Person
    definition_period = ETERNITY
    label = "Person household role"


class household_weight(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household weight"


class family_income_for_weight_test(Variable):
    value_type = float
    entity = Family
    definition_period = MONTH
    label = "Family income for weight tests"


def _family_weight_tax_benefit_system():
    tax_benefit_system = FamilyWeightTaxBenefitSystem()
    for variable in (
        person_id,
        person_family_id,
        person_family_role,
        person_household_id,
        person_household_role,
        household_weight,
        family_income_for_weight_test,
    ):
        tax_benefit_system.add_variable(variable)
    return tax_benefit_system


def _family_weight_dataset() -> Dataset:
    return Dataset.from_dataframe(
        pd.DataFrame(
            {
                "person_id__2022": [0, 1, 2],
                "person_family_id__2022": [0, 0, 1],
                "person_family_role__2022": ["member", "member", "member"],
                "person_household_id__2022": [0, 0, 0],
                "person_household_role__2022": ["member", "member", "member"],
                "household_weight__2022": [50.0, 50.0, 50.0],
                "family_income_for_weight_test__2022-01": [100.0, 100.0, 200.0],
            }
        ),
        "2022",
    )


def test__given_non_household_group_entity__then_weights_are_not_summed_by_members():
    # Given
    simulation = CoreMicrosimulation(
        tax_benefit_system=_family_weight_tax_benefit_system(),
        dataset=_family_weight_dataset(),
    )

    # When
    weights = simulation.get_weights(
        "family_income_for_weight_test",
        "2022-01",
    )
    dataframe = simulation.calculate_dataframe(
        ["family_income_for_weight_test"],
        "2022-01",
        map_to="family",
    )

    # Then
    np.testing.assert_array_equal(weights, np.array([50.0, 50.0]))
    np.testing.assert_array_equal(dataframe.weights, np.array([50.0, 50.0]))
