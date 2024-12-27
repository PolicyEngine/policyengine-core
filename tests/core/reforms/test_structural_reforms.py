import warnings

import pytest
from datetime import datetime
from policyengine_core.errors import VariableNotFoundError
from policyengine_core.model_api import *
from policyengine_core.country_template.entities import Person
from policyengine_core.variables import Variable
from policyengine_core.reforms import StructuralReform


class test_variable_to_add(Variable):
    value_type = str
    default_value = "Returning default value"
    entity = Person
    label = "Maxwell"
    definition_period = YEAR

    def formula():
        return "Returning value from formula"


class test_existing_variable(Variable):
    value_type = str
    default_value = "Returning default value, existing variable"
    entity = Person
    label = "Dworkin"
    definition_period = YEAR

    def formula():
        return "Returning value from formula, existing variable"


def test_structural_reform_init(tax_benefit_system):
    # Given an empty tax-benefit system...

    # When a new structural reform is created with default settings...
    test_reform = StructuralReform(tax_benefit_system)

    # Then the reform is created successfully for the current year
    assert test_reform.tax_benefit_system == tax_benefit_system
    assert test_reform.start_instant == str(datetime.now().year) + "-01-01"
    assert test_reform.end_instant == None


def test_structural_reform_init_with_dates(tax_benefit_system):
    # Given an empty tax-benefit system...

    # When a new structural reform is created with specific dates...
    reform = StructuralReform(tax_benefit_system, "2020-01-01", "2021-01-01")

    # Then the reform is created successfully for the specified dates
    assert reform.tax_benefit_system == tax_benefit_system
    assert reform.start_instant == "2020-01-01"
    assert reform.end_instant == "2021-01-01"


def test_empty_tbs_endless_structural_reform_add_variable(tax_benefit_system):
    # Given an empty tax-benefit system with an endless structural reform...
    test_reform = StructuralReform(
        tax_benefit_system,
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
    # TODO - Figure out how to test formula additions


def test_empty_tbs_endless_structural_reform_update_variable(
    tax_benefit_system,
):
    # Given an empty tax-benefit system with an endless structural reform...
    test_reform = StructuralReform(
        tax_benefit_system,
        "2025-01-01",
    )

    # When update_variable is called on a variable that does not exist...
    test_reform.update_variable(test_variable_to_add)

    # Then update_variable(test_var) adds new variable with proper formulas
    assert test_variable_to_add in test_reform.tax_benefit_system.variables

    added_test_variable = test_reform.tax_benefit_system.get_variable(
        "test_variable_to_add"
    )
    assert (
        added_test_variable.get_formula("2025-01-01")()
        == "Returning value from formula"
    )
    assert (
        added_test_variable.get_formula("2021-01-01")()
        == "Returning default value"
    )


def test_empty_tbs_endless_structural_reform_neutralize_variable(
    tax_benefit_system,
):
    # Given an empty tax-benefit system with an endless structural reform...
    test_reform = StructuralReform(
        tax_benefit_system,
        "2025-01-01",
    )

    # When neutralize_variable is called on a variable that does not exist...

    # Then neutralize_variable(test_var) raises error
    with pytest.raises(VariableNotFoundError):
        test_reform.neutralize_variable("test_variable_to_add")


# Given a TBS with a variable test_var...
# add_variable(test_var) raises error

# update_variable(test_var) updates variable

# neutralize_variable(test_var) neutralizes variable


# Given a TBS with a complex structural reform...
# The reform successfully adds a variable, updates a variable, then neutralizes a variable
