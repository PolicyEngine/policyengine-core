import policyengine_core.country_template as country_template
from policyengine_core.country_template.entities import Person
from policyengine_core.variables import Variable
from policyengine_core.periods import YEAR
from test_variables import get_message

tax_benefit_system = country_template.CountryTaxBenefitSystem()


class variable__bound(Variable):
    value_type = int
    entity = Person
    definition_period = YEAR
    label = "Variable with bound."
    min_value = 0
    max_value = 100


tax_benefit_system.add_variable(variable__bound)


def test_variable__bound():
    variable = tax_benefit_system.variables["variable__bound"]
    assert variable.min_value == 0
    assert variable.max_value == 100


class variable__no_bound(Variable):
    value_type = int
    entity = Person
    definition_period = YEAR
    label = "Variable with no bound."


tax_benefit_system.add_variable(variable__no_bound)


def test_variable__no_bound():
    variable = tax_benefit_system.variables["variable__no_bound"]
    assert variable.min_value is None
    assert variable.max_value is None


class variable__small_max(Variable):
    value_type = int
    entity = Person
    definition_period = YEAR
    label = "Variable with max value smaller than min value."
    min_value = 100
    max_value = 99


def test_variable__small_max():
    try:
        tax_benefit_system.add_variable(variable__small_max)
    except ValueError as e:
        message = get_message(e)
        assert message.startswith("min_value cannot be greater than max_value")
    assert not tax_benefit_system.variables.get("variable__small_max")


class variable__wrong_type(Variable):
    value_type = int
    entity = Person
    definition_period = YEAR
    label = "Variable with wrong max value data type."
    max_value = "string"


def test_variable__wrong_type():
    try:
        tax_benefit_system.add_variable(variable__wrong_type)
    except ValueError as e:
        message = get_message(e)
        assert message.startswith(
            "Invalid value 'string' for attribute 'max_value' in variable 'variable__wrong_type'. "
            "Must be of type '(<class 'float'>, <class 'int'>)'."
        )
    assert not tax_benefit_system.variables.get("variable__wrong_type")
