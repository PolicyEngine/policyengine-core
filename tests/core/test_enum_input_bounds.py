"""Public enum-input boundaries reject bad positions before dtype narrowing."""

import numpy as np
import pytest

from policyengine_core.entities import build_entity
from policyengine_core.enums import Enum
from policyengine_core.errors import SituationParsingError
from policyengine_core.periods import YEAR
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.variables import Variable


class PositionChoice(Enum):
    FIRST = "First"
    SECOND = "Second"
    LAST = "Last"


Person = build_entity("person", "persons", "Person", is_person=True)


class reported_position(Variable):
    value_type = Enum
    possible_values = PositionChoice
    default_value = PositionChoice.FIRST
    entity = Person
    definition_period = YEAR
    label = "Reported synthetic position"


INVALID_SCALARS = [
    pytest.param(-1, id="negative"),
    pytest.param(3, id="upper-bound"),
    pytest.param(np.int64(65536), id="wide-signed"),
    pytest.param(np.uint64(65536), id="wide-unsigned"),
    pytest.param(np.uint64(np.iinfo(np.uint64).max), id="unsigned-maximum"),
]


@pytest.fixture
def position_system():
    system = TaxBenefitSystem([Person])
    system.add_variable(reported_position)
    return system


@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_variable_checks_integer_bounds_before_narrowing(position_system, value):
    variable = position_system.get_variable("reported_position")
    with pytest.raises(ValueError, match="PositionChoice"):
        variable.check_set_value(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param("FIRST", 0, id="name"),
        pytest.param(0, 0, id="first-position"),
        pytest.param(np.int64(2), 2, id="last-position"),
    ],
)
def test_variable_preserves_valid_names_and_positions(position_system, value, expected):
    assert (
        position_system.get_variable("reported_position").check_set_value(value)
        == expected
    )


@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_set_input_rejects_bad_positions_without_replacing_valid_input(
    position_system, value
):
    simulation = SimulationBuilder().build_from_dict(
        position_system, {"persons": {"p": {}}}
    )
    simulation.set_input("reported_position", 2026, np.array([1], dtype=np.int64))
    with pytest.raises(ValueError, match="PositionChoice"):
        simulation.set_input("reported_position", 2026, np.array([value]))
    np.testing.assert_array_equal(
        np.asarray(simulation.calculate("reported_position", 2026)), [1]
    )


@pytest.mark.parametrize("dtype", [np.int64, np.uint64], ids=["signed", "unsigned"])
@pytest.mark.parametrize("value", [0, 1, 2], ids=["first", "middle", "last"])
def test_set_input_accepts_valid_zero_based_positions(position_system, dtype, value):
    simulation = SimulationBuilder().build_from_dict(
        position_system, {"persons": {"p": {}}}
    )
    simulation.set_input("reported_position", 2026, np.array([value], dtype=dtype))
    np.testing.assert_array_equal(
        np.asarray(simulation.calculate("reported_position", 2026)), [value]
    )


@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_situation_rejects_bad_positions_before_builder_narrowing(
    position_system, value
):
    situation = {"persons": {"p": {"reported_position": {"2026": value}}}}
    with pytest.raises(SituationParsingError) as error:
        SimulationBuilder().build_from_dict(position_system, situation)
    assert "reported_position" in str(error.value)
    assert "PositionChoice" in str(error.value)


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param("FIRST", 0, id="name"),
        pytest.param(0, 0, id="first-position"),
        pytest.param(2, 2, id="last-position"),
        pytest.param(np.int64(2), 2, id="numpy-position"),
    ],
)
def test_situation_keeps_valid_names_and_positions(position_system, value, expected):
    simulation = SimulationBuilder().build_from_dict(
        position_system,
        {"persons": {"p": {"reported_position": {"2026": value}}}},
    )
    np.testing.assert_array_equal(
        np.asarray(simulation.calculate("reported_position", 2026)), [expected]
    )
