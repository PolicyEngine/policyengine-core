import warnings

import pytest
from datetime import datetime
from policyengine_core.errors import (
    VariableNotFoundError,
    VariableNameConflictError,
)
from policyengine_core.model_api import *
from policyengine_core.country_template.entities import Person
from policyengine_core.variables import Variable
from policyengine_core.reforms import StructuralReform
from policyengine_core.country_template import CountryTaxBenefitSystem


@pytest.fixture(scope="module")
def prefilled_tax_benefit_system():

    class test_variable_to_add(Variable):
        value_type = str
        default_value = "Returning default value [pre-loaded]"
        entity = Person
        label = "Dworkin"
        definition_period = YEAR

        def formula():
            return "Returning value from formula [pre-loaded]"

    tax_benefit_system = CountryTaxBenefitSystem()
    tax_benefit_system.add_variable(test_variable_to_add)
    return tax_benefit_system


class test_variable_to_add(Variable):
    value_type = str
    default_value = "Returning default value"
    entity = Person
    label = "Maxwell"
    definition_period = YEAR

    def formula():
        return "Returning value from formula"


class TestGivenEmptyTaxBenefitSystem:
    # Given an empty tax-benefit system...
    def test_structural_reform_init(self, isolated_tax_benefit_system):

        # When a new structural reform is created with default settings...
        test_reform = StructuralReform(isolated_tax_benefit_system)

        # Then the reform is created successfully for the current year
        assert test_reform.tax_benefit_system == isolated_tax_benefit_system
        assert test_reform.start_instant == str(datetime.now().year) + "-01-01"
        assert test_reform.end_instant == None

    def test_structural_reform_init_with_dates(
        self, isolated_tax_benefit_system
    ):

        # When a new structural reform is created with specific dates...
        reform = StructuralReform(
            isolated_tax_benefit_system, "2020-01-01", "2021-01-01"
        )

        # Then the reform is created successfully for the specified dates
        assert reform.tax_benefit_system == isolated_tax_benefit_system
        assert reform.start_instant == "2020-01-01"
        assert reform.end_instant == "2021-01-01"

    def test_structural_reform_init_with_invalid_date_type(
        self, isolated_tax_benefit_system
    ):

        # When a new structural reform is created with incorrectly typed dates...

        # Then the reform raises a TypeError

        with pytest.raises(TypeError):
            StructuralReform(isolated_tax_benefit_system, "2020-01-01", 15)

    def test_structural_reform_init_with_invalid_date_format(
        self, isolated_tax_benefit_system
    ):

        # When a new structural reform is created with incorrectly formatted dates...

        # Then the reform raises a ValueError

        with pytest.raises(ValueError):
            StructuralReform(
                isolated_tax_benefit_system, "2020-01-01", "2020-13-01"
            )

    def test_add_variable_no_end_dates(
        self,
        isolated_tax_benefit_system,
    ):
        # Given an empty tax-benefit system with an endless structural reform...
        test_reform = StructuralReform(
            isolated_tax_benefit_system,
            "2025-01-01",
        )

        # When a new variable is added...
        test_reform.add_variable(test_variable_to_add)

        # Then add_variable(test_var) adds new variable with proper formulas
        assert (
            "test_variable_to_add"
            in test_reform.tax_benefit_system.variables.keys()
        )

        added_test_variable = test_reform.tax_benefit_system.get_variable(
            "test_variable_to_add"
        )
        assert added_test_variable.value_type == str
        assert added_test_variable.label == "Maxwell"
        assert (
            added_test_variable.get_formula("2025-01-01")()
            == "Returning value from formula"
        )
        assert (
            added_test_variable.get_formula("2021-01-01")()
            == "Returning default value"
        )

    def test_update_variable_no_end_dates(
        self,
        isolated_tax_benefit_system,
    ):
        # Given an empty tax-benefit system with an endless structural reform...
        test_reform = StructuralReform(
            isolated_tax_benefit_system,
            "2025-01-01",
        )

        # When update_variable is called on a variable that does not exist...
        test_reform.update_variable(test_variable_to_add)

        # Then update_variable(test_var) adds new variable with proper formulas
        assert (
            "test_variable_to_add"
            in test_reform.tax_benefit_system.variables.keys()
        )

        added_test_variable = test_reform.tax_benefit_system.get_variable(
            "test_variable_to_add"
        )
        assert added_test_variable.value_type == str
        assert added_test_variable.label == "Maxwell"
        assert (
            added_test_variable.get_formula("2025-01-01")()
            == "Returning value from formula"
        )
        assert (
            added_test_variable.get_formula("2021-01-01")()
            == "Returning default value"
        )

    def test_neutralize_variable_no_end_dates(
        self,
        isolated_tax_benefit_system,
    ):
        # Given an empty tax-benefit system with an endless structural reform...
        test_reform = StructuralReform(
            isolated_tax_benefit_system,
            "2025-01-01",
        )

        # When neutralize_variable is called on a variable that does not exist...

        # Then neutralize_variable(test_var) raises error
        with pytest.raises(VariableNotFoundError):
            test_reform.neutralize_variable("test_variable_to_add")


class TestGivenPreFilledTaxBenefitSystem:

    # Given a tax-benefit system with variable test_variable_to_add...

    def test_add_variable_no_end_dates(
        self,
        prefilled_tax_benefit_system,
    ):
        # Given a tax-benefit system with an endless structural reform...
        test_reform = StructuralReform(
            prefilled_tax_benefit_system,
            "2025-01-01",
        )

        # When a new variable is added...

        # Then add_variable(test_var) raises error
        with pytest.raises(VariableNameConflictError):
            test_reform.add_variable(test_variable_to_add)

    def test_update_variable_no_end_dates(
        self,
        prefilled_tax_benefit_system,
    ):
        # Given a tax-benefit system with an endless structural reform...
        test_reform = StructuralReform(
            prefilled_tax_benefit_system,
            "2025-01-01",
        )

        # When update_variable is called on a variable that already exists...
        test_reform.update_variable(test_variable_to_add)

        # Then update_variable(test_var) updates variable with proper formulas
        added_test_variable = test_reform.tax_benefit_system.get_variable(
            "test_variable_to_add"
        )
        assert added_test_variable.value_type == str
        assert added_test_variable.label == "Maxwell"
        assert (
            added_test_variable.get_formula("2025-01-01")()
            == "Returning value from formula"
        )
        assert (
            added_test_variable.get_formula("2021-01-01")()
            == "Returning default value"
        )

    def test_neutralize_variable_no_end_dates(
        self,
        prefilled_tax_benefit_system,
    ):
        # Given a tax-benefit system with an endless structural reform...
        test_reform = StructuralReform(
            prefilled_tax_benefit_system,
            "2025-01-01",
        )

        # When neutralize_variable is called on a variable that already exists...
        test_reform.neutralize_variable("test_variable_to_add")

        # Then neutralize_variable(test_var) neutralizes variable
        added_test_variable = test_reform.tax_benefit_system.get_variable(
            "test_variable_to_add"
        )
        assert added_test_variable.value_type == str
        assert added_test_variable.label == "Maxwell"
        assert (
            added_test_variable.get_formula("2025-01-01")()
            == "Returning default value"
        )
        assert (
            added_test_variable.get_formula("2021-01-01")()
            == "Returning default value"
        )


# Given a TBS with a complex structural reform...
# The reform successfully adds a variable, updates a variable, then neutralizes a variable
