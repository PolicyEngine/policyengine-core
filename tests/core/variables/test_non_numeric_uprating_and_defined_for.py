"""``uprating`` and ``defined_for`` need values that support arithmetic.

Uprating multiplies a variable's earlier value by an index ratio, and
``defined_for`` keeps values where another variable is greater than zero.
Enum arrays allow only ``==`` and ``!=``, and ``str`` and date arrays cannot
be multiplied by a float or compared with a number. So an Enum, ``str`` or
date variable with ``uprating``, or a variable ``defined_for`` one, used to
register without complaint and then raise ``TypeError: Forbidden operation``
(or a numpy error) in the middle of a calculation.

Both are now rejected when the variables are registered, with a message that
names them.
"""

from __future__ import annotations

import datetime
import textwrap

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template import entities as template_entities
from policyengine_core.country_template.entities import Person
from policyengine_core.enums import Enum
from policyengine_core.model_api import YEAR, Reform, Variable
from policyengine_core.parameters import ParameterNode
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.taxbenefitsystems import TaxBenefitSystem


class State(Enum):
    absent = "Absent"
    present = "Present"


class Flag(Enum):
    # The value names a variable: ``defined_for = Flag.in_scope``.
    in_scope = "in_scope"


NON_NUMERIC = {
    "Enum": dict(value_type=Enum, possible_values=State, default_value=State.absent),
    "str": dict(value_type=str),
    "date": dict(value_type=datetime.date),
}
NUMERIC = {
    "bool": dict(value_type=bool),
    "int": dict(value_type=int),
    "float": dict(value_type=float),
}


def _variable(name, **attributes):
    return type(
        name,
        (Variable,),
        dict(entity=Person, definition_period=YEAR, label=name, **attributes),
    )


def _system() -> CountryTaxBenefitSystem:
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = True
    system.parameters.add_child(
        "probe",
        ParameterNode(
            "probe",
            data={"index": {"values": {"2010-01-01": 100, "2015-01-01": 200}}},
        ),
    )
    return system


def _simulation(system, **inputs):
    return SimulationBuilder().build_from_entities(system, {"persons": {"p": inputs}})


# --- uprating ---------------------------------------------------------------


@pytest.mark.parametrize("type_name", NON_NUMERIC)
def test_uprating_on_a_non_numeric_variable_is_rejected(type_name):
    system = _system()
    variable = _variable("uprated", uprating="probe.index", **NON_NUMERIC[type_name])

    with pytest.raises(ValueError) as error:
        system.add_variable(variable)

    message = str(error.value)
    assert 'Variable "uprated" has uprating "probe.index"' in message
    assert f"value_type is {type_name}" in message
    assert "uprated" not in system.variables


@pytest.mark.parametrize("type_name", NUMERIC)
def test_uprating_on_a_numeric_variable_is_accepted(type_name):
    system = _system()
    system.add_variable(
        _variable("uprated", uprating="probe.index", **NUMERIC[type_name])
    )
    simulation = _simulation(system, uprated={"2012": 1})

    # The index doubles between 2012 and 2015.
    expected = {"bool": True, "int": 2, "float": 2.0}[type_name]
    assert simulation.calculate("uprated", "2015").tolist() == [expected]


@pytest.mark.parametrize("type_name", NON_NUMERIC)
def test_assigning_uprating_to_a_non_numeric_variable_is_rejected(type_name):
    # Country packages assign ``variable.uprating`` after loading (default
    # uprating for dollar inputs), so the setter checks as well.
    system = _system()
    variable = system.add_variable(_variable("plain", **NON_NUMERIC[type_name]))

    with pytest.raises(ValueError, match='Variable "plain" has uprating'):
        variable.uprating = "probe.index"

    assert variable.uprating is None


def test_a_reform_cannot_turn_an_uprated_variable_into_an_enum():
    # The reform's class declares no uprating, but inherits the baseline's.
    system = _system()
    system.add_variable(_variable("uprated", value_type=float, uprating="probe.index"))

    with pytest.raises(ValueError, match="value_type is Enum"):
        system.update_variable(_variable("uprated", **NON_NUMERIC["Enum"]))


def test_an_enum_input_without_uprating_carries_over():
    # What the error message advises: with auto-carry-over, an input without
    # ``uprating`` keeps its value in later periods.
    system = _system()
    system.add_variable(_variable("state", **NON_NUMERIC["Enum"]))
    simulation = _simulation(system, state={"2012": "present"})

    assert simulation.calculate("state", "2015").decode_to_str().tolist() == ["present"]


# --- defined_for ------------------------------------------------------------


@pytest.mark.parametrize("target_first", [True, False])
@pytest.mark.parametrize("type_name", NON_NUMERIC)
def test_defined_for_a_non_numeric_variable_is_rejected(type_name, target_first):
    system = _system()
    target = _variable("condition", **NON_NUMERIC[type_name])
    dependent = _variable("amount", value_type=float, defined_for="condition")

    with pytest.raises(ValueError) as error:
        # Whichever is added second completes the pair.
        system.add_variables(
            *((target, dependent) if target_first else (dependent, target))
        )

    message = str(error.value)
    assert 'Variable "amount" is defined_for "condition"' in message
    assert f"value_type is {type_name}" in message


@pytest.mark.parametrize(
    "type_name, value, expected",
    [("bool", True, 5.0), ("int", 3, 5.0), ("float", 0.5, 5.0), ("int", 0, 0.0)],
)
def test_defined_for_a_numeric_variable_keeps_nonzero_entities(
    type_name, value, expected
):
    system = _system()
    system.add_variables(
        _variable("condition", **NUMERIC[type_name]),
        _variable("amount", value_type=float, defined_for="condition"),
    )
    simulation = _simulation(system, condition={"2015": value}, amount={"2012": 5.0})

    # No 2015 input: the 2012 one carries over where the condition is nonzero.
    assert simulation.calculate("amount", "2015").tolist() == [expected]


def test_defined_for_an_enum_member_names_the_variable_called_after_its_value():
    system = _system()
    system.add_variables(
        _variable("in_scope", value_type=bool),
        _variable("amount", value_type=float, defined_for=Flag.in_scope),
    )
    assert system.variables["amount"].defined_for == "in_scope"
    simulation = _simulation(system, in_scope={"2015": True}, amount={"2012": 5.0})

    assert simulation.calculate("amount", "2015").tolist() == [5.0]


def test_defined_for_a_variable_that_is_not_registered_is_left_alone():
    # As before: it only fails if the variable is still missing when
    # calculated.
    system = _system()
    system.add_variable(
        _variable("amount", value_type=float, defined_for="not_defined")
    )

    assert system.variables["amount"].defined_for == "not_defined"


def test_a_reform_cannot_turn_a_defined_for_variable_into_an_enum():
    system = _system()
    system.add_variables(
        _variable("condition", value_type=bool),
        _variable("amount", value_type=float, defined_for="condition"),
    )

    class make_condition_an_enum(Reform):
        def apply(self):
            self.update_variable(_variable("condition", **NON_NUMERIC["Enum"]))

    with pytest.raises(ValueError, match='"amount" is defined_for "condition"'):
        make_condition_an_enum(system)


VARIABLE_FILE = textwrap.dedent(
    """
    from policyengine_core.country_template.entities import Person
    from policyengine_core.enums import Enum
    from policyengine_core.model_api import YEAR, Variable


    class State(Enum):
        absent = "Absent"
        present = "Present"
    """
)
CONDITION = textwrap.dedent(
    """

    class condition(Variable):
        value_type = {value_type}
        {extra}
        entity = Person
        definition_period = YEAR
        label = "Condition"
    """
)
AMOUNT = textwrap.dedent(
    """

    class amount(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        defined_for = "condition"
        label = "Amount"
    """
)


def _directory_system(directory):
    class DirectorySystem(TaxBenefitSystem):
        entities = template_entities.entities
        variables_dir = str(directory)

    return DirectorySystem


@pytest.mark.parametrize("condition_loaded_first", [True, False])
def test_a_variables_directory_is_checked_once_it_is_loaded(
    tmp_path, condition_loaded_first
):
    # Files in a directory load before its subdirectories, so each layout
    # fixes which of the two variables is registered first.
    enum_condition = CONDITION.format(
        value_type="Enum",
        extra="possible_values = State\n    default_value = State.absent",
    )
    first, second = (
        (enum_condition, AMOUNT) if condition_loaded_first else (AMOUNT, enum_condition)
    )
    (tmp_path / "first.py").write_text(VARIABLE_FILE + first)
    (tmp_path / "later").mkdir()
    (tmp_path / "later" / "second.py").write_text(VARIABLE_FILE + second)

    with pytest.raises(ValueError, match='"amount" is defined_for "condition"'):
        _directory_system(tmp_path)()


def test_a_variables_directory_with_a_bool_condition_loads(tmp_path):
    (tmp_path / "first.py").write_text(VARIABLE_FILE + AMOUNT)
    (tmp_path / "later").mkdir()
    (tmp_path / "later" / "second.py").write_text(
        VARIABLE_FILE + CONDITION.format(value_type="bool", extra="")
    )

    system = _directory_system(tmp_path)()

    assert system.variables["amount"].defined_for == "condition"


def test_defined_for_changed_after_registration_fails_with_the_same_message():
    # Registration cannot see an attribute assigned afterwards; ``calculate``
    # then raises the same error instead of a TypeError from ``> 0``.
    system = _system()
    system.add_variables(
        _variable("state", **NON_NUMERIC["Enum"]),
        _variable("amount", value_type=float),
    )
    system.variables["amount"].defined_for = "state"
    simulation = _simulation(system, state={"2015": "present"}, amount={"2012": 5.0})

    with pytest.raises(ValueError, match='"amount" is defined_for "state"'):
        simulation.calculate("amount", "2015")
