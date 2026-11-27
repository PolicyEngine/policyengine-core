"""What a traced simulation records about the parameters it reads.

Every test here runs a calculation and looks at the reads recorded on the
variable's node of the trace, so none depends on how reads reach the tracer.
"""

import enum
import json

import numpy as np
import pytest

from policyengine_core.country_template.entities import Person
from policyengine_core.parameters import Parameter, ParameterNode
from policyengine_core.parameters.operations.get_parameter import get_parameter
from policyengine_core.periods import ETERNITY, MONTH, YEAR
from policyengine_core.simulations import Simulation
from policyengine_core.tracers import SimpleTracer
from policyengine_core.variables import Variable
from tests.fixtures.tracing import assert_untraced

JANUARY = "2017-01"
INSTANT = "2017-01-01"
SOCIAL_SECURITY_SCALE = "taxes.social_security_contribution"
SOCIAL_SECURITY_DESCRIPTION = {
    "type": "MarginalRateTaxScale",
    "thresholds": [0.0, 6000.0, 12400.0],
    "rates": [0.02, 0.06, 0.12],
}


def trace_node(simulation, variable_name):
    """The first node calculating ``variable_name``, anywhere in the trace."""
    pending_nodes = list(simulation.tracer.trees)
    while pending_nodes:
        node = pending_nodes.pop(0)
        if node.name == variable_name:
            return node
        pending_nodes.extend(node.children)
    raise AssertionError(f"{variable_name} is not in the trace")


def recorded_reads(simulation, variable_name):
    """The reads on the variable's node, as (name, period, branch, value)."""
    return [
        (read.name, read.period, read.branch_name, read.value)
        for read in trace_node(simulation, variable_name).parameters
    ]


def recorded_names(simulation, variable_name):
    return [read[0] for read in recorded_reads(simulation, variable_name)]


def define_variable(system, name, formula, definition_period=MONTH, **attributes):
    """Add a float person variable to ``system``."""
    class_attributes = {
        "value_type": float,
        "entity": Person,
        "definition_period": definition_period,
        "label": name,
        **attributes,
    }
    if formula is not None:
        class_attributes["formula"] = formula
    system.add_variable(type(name, (Variable,), class_attributes))


@pytest.fixture
def system(isolated_tax_benefit_system):
    return isolated_tax_benefit_system


@pytest.fixture
def traced_simulation(system, make_simulation):
    def build(variables=None, period=JANUARY):
        simulation = make_simulation(system, variables or {"salary": 2000}, period)
        simulation.trace = True
        return simulation

    return build


# ----- Reads at an instant ----- #


def test_yearly_formula_read_is_recorded_when_the_instant_was_read_before(
    system, traced_simulation
):
    def formula(person, period, parameters):
        return person.filled_array(parameters(period).taxes.income_tax_rate)

    define_variable(system, "yearly_rate", formula, definition_period=YEAR)
    rate = system.parameters(INSTANT).taxes.income_tax_rate
    simulation = traced_simulation()

    simulation.calculate("yearly_rate", "2017")

    assert recorded_reads(simulation, "yearly_rate") == [
        ("taxes.income_tax_rate", INSTANT, "default", rate)
    ]


def test_scale_read_is_recorded_with_its_schedule(system, traced_simulation):
    simulation = traced_simulation()

    simulation.calculate("social_security_contribution", JANUARY)

    assert recorded_reads(simulation, "social_security_contribution") == [
        (SOCIAL_SECURITY_SCALE, INSTANT, "default", SOCIAL_SECURITY_DESCRIPTION)
    ]
    json.dumps(simulation.tracer.get_serialized_flat_trace())


def test_scale_read_by_key_is_recorded(system, traced_simulation):
    def formula(person, period, parameters):
        scale = parameters(period).taxes["social_security_contribution"]
        return scale.calc(person("salary", period))

    define_variable(system, "scale_by_key", formula)
    simulation = traced_simulation()

    simulation.calculate("scale_by_key", JANUARY)

    assert recorded_names(simulation, "scale_by_key") == [SOCIAL_SECURITY_SCALE]


def test_reading_private_attributes_of_a_node_records_nothing(
    system, traced_simulation
):
    def formula(person, period, parameters):
        taxes = parameters(period).taxes
        return person.filled_array(len(taxes._name) + len(taxes._instant_str))

    define_variable(system, "node_labels", formula)
    simulation = traced_simulation()

    simulation.calculate("node_labels", JANUARY)

    assert recorded_reads(simulation, "node_labels") == []


# ----- Reads of a node's children ----- #


def test_listing_a_nodes_children_records_the_node_with_their_names(
    system, traced_simulation
):
    def formula(person, period, parameters):
        return person.filled_array(len(list(parameters(period).taxes._children)))

    define_variable(system, "tax_count", formula)
    child_names = list(system.parameters(INSTANT).taxes._children)
    simulation = traced_simulation()

    simulation.calculate("tax_count", JANUARY)

    assert recorded_reads(simulation, "tax_count") == [
        ("taxes", INSTANT, "default", child_names)
    ]


@pytest.mark.parametrize(
    "read_children",
    [
        lambda children: len(children),
        lambda children: "income_tax_rate" in children,
        lambda children: "no_such_child" in children,
        lambda children: list(children.keys()),
        lambda children: children.get("no_such_child", 0),
    ],
    ids=["len", "in", "not-in", "keys", "get-missing"],
)
def test_reading_which_children_exist_records_the_node(
    system, traced_simulation, read_children
):
    def formula(person, period, parameters):
        read_children(parameters(period).taxes._children)
        return person.filled_array(0)

    define_variable(system, "children_shape", formula)
    simulation = traced_simulation()

    simulation.calculate("children_shape", JANUARY)

    assert recorded_names(simulation, "children_shape") == ["taxes"]


def test_child_taken_from_the_mapping_is_recorded_at_its_own_path(
    system, traced_simulation
):
    def formula(person, period, parameters):
        return person.filled_array(
            parameters(period).taxes._children["income_tax_rate"]
        )

    define_variable(system, "rate_from_mapping", formula)
    rate = system.parameters(INSTANT).taxes.income_tax_rate
    simulation = traced_simulation()

    simulation.calculate("rate_from_mapping", JANUARY)

    assert recorded_reads(simulation, "rate_from_mapping") == [
        ("taxes.income_tax_rate", INSTANT, "default", rate)
    ]


def test_reads_below_a_node_taken_from_the_mapping_are_recorded(
    system, traced_simulation
):
    def formula(person, period, parameters):
        housing_tax = parameters(period).taxes._children["housing_tax"]
        return person.filled_array(housing_tax.rate)

    define_variable(system, "nested_rate", formula)
    simulation = traced_simulation()

    simulation.calculate("nested_rate", JANUARY)

    assert recorded_names(simulation, "nested_rate") == ["taxes.housing_tax.rate"]


def test_values_of_the_mapping_are_traced(system, traced_simulation):
    def formula(person, period, parameters):
        child_nodes = [
            child
            for child in parameters(period).benefits._children.values()
            if not isinstance(child, (int, float))
        ]
        return person.filled_array(len(child_nodes[0]._children))

    define_variable(system, "first_benefit_size", formula)
    simulation = traced_simulation()

    simulation.calculate("first_benefit_size", JANUARY)

    names = recorded_names(simulation, "first_benefit_size")
    assert names[0] == "benefits"
    assert "benefits.parenting_allowance" in names
    assert not [name for name in names if name.endswith("_children")]


def test_iterating_a_node_records_it_and_yields_child_names(system, traced_simulation):
    def formula(person, period, parameters):
        return person.filled_array(len(sorted(parameters(period).taxes)))

    define_variable(system, "iterated_taxes", formula)
    child_names = list(system.parameters(INSTANT).taxes)
    simulation = traced_simulation()

    assert simulation.calculate("iterated_taxes", JANUARY)[0] == len(child_names)

    assert recorded_reads(simulation, "iterated_taxes") == [
        ("taxes", INSTANT, "default", child_names)
    ]


def test_membership_test_on_a_node_records_it(system, traced_simulation):
    def formula(person, period, parameters):
        return person.filled_array("income_tax_rate" in parameters(period).taxes)

    define_variable(system, "has_income_tax", formula)
    simulation = traced_simulation()

    assert simulation.calculate("has_income_tax", JANUARY)[0] == 1

    assert recorded_names(simulation, "has_income_tax") == ["taxes"]


# ----- Reads through the tree a formula receives, before choosing an instant ----- #


def test_parameter_called_from_the_undated_tree_is_recorded(system, traced_simulation):
    def formula(person, period, parameters):
        return person.filled_array(parameters.taxes.income_tax_rate(period))

    define_variable(system, "undated_rate", formula)
    rate = system.parameters(INSTANT).taxes.income_tax_rate
    simulation = traced_simulation()

    assert simulation.calculate("undated_rate", JANUARY)[0] == pytest.approx(rate)

    assert recorded_reads(simulation, "undated_rate") == [
        ("taxes.income_tax_rate", INSTANT, "default", rate)
    ]


def test_parameter_called_at_another_instant_records_that_instant(
    system, traced_simulation
):
    def formula(person, period, parameters):
        return person.filled_array(parameters.taxes.income_tax_rate("2015-06-01"))

    define_variable(system, "frozen_rate", formula)
    rate = system.parameters("2015-06-01").taxes.income_tax_rate
    simulation = traced_simulation()

    simulation.calculate("frozen_rate", JANUARY)

    assert recorded_reads(simulation, "frozen_rate") == [
        ("taxes.income_tax_rate", "2015-06-01", "default", rate)
    ]


def test_node_called_from_the_undated_tree_records_later_reads(
    system, traced_simulation
):
    def formula(person, period, parameters):
        return person.filled_array(parameters.taxes.housing_tax(period).rate)

    define_variable(system, "undated_node_rate", formula)
    simulation = traced_simulation()

    simulation.calculate("undated_node_rate", JANUARY)

    assert recorded_names(simulation, "undated_node_rate") == ["taxes.housing_tax.rate"]


def test_scale_called_from_the_undated_tree_is_recorded(system, traced_simulation):
    def formula(person, period, parameters):
        scale = parameters.taxes.social_security_contribution(period)
        return scale.calc(person("salary", period))

    define_variable(system, "undated_scale", formula)
    simulation = traced_simulation()

    simulation.calculate("undated_scale", JANUARY)

    assert recorded_reads(simulation, "undated_scale") == [
        (SOCIAL_SECURITY_SCALE, INSTANT, "default", SOCIAL_SECURITY_DESCRIPTION)
    ]


def test_bracket_of_a_scale_on_the_undated_tree_is_recorded(system, traced_simulation):
    def formula(person, period, parameters):
        bracket = parameters.taxes.social_security_contribution[1]
        return person.filled_array(bracket.rate(period))

    define_variable(system, "undated_bracket", formula)
    simulation = traced_simulation()

    assert simulation.calculate("undated_bracket", JANUARY)[0] == pytest.approx(0.06)

    assert recorded_names(simulation, "undated_bracket") == [
        f"{SOCIAL_SECURITY_SCALE}[1].rate"
    ]


def test_undated_reads_leave_the_system_untraced(system, traced_simulation):
    def formula(person, period, parameters):
        return person.filled_array(parameters.taxes.income_tax_rate(period))

    define_variable(system, "undated_rate", formula)

    traced_simulation().calculate("undated_rate", JANUARY)

    assert_untraced(system.parameters)


def test_traced_and_untraced_undated_reads_agree(
    system, traced_simulation, make_simulation
):
    def formula(person, period, parameters):
        return parameters.taxes.social_security_contribution(period).calc(
            person("salary", period)
        ) + parameters.taxes.income_tax_rate(period)

    define_variable(system, "undated_total", formula)

    traced = traced_simulation().calculate("undated_total", JANUARY)
    untraced = make_simulation(system, {"salary": 2000}, JANUARY).calculate(
        "undated_total", JANUARY
    )

    np.testing.assert_array_equal(traced, untraced)


def read_clone(rate):
    return rate.clone()("2017-01-01")


def read_values_list(rate):
    return rate.values_list[0].value


@pytest.mark.parametrize(
    "read_rate", [read_clone, read_values_list], ids=["clone", "values-list"]
)
def test_parameter_used_outside_a_dated_call_is_recorded_for_all_instants(
    system, traced_simulation, read_rate
):
    def formula(person, period, parameters):
        return person.filled_array(read_rate(parameters.taxes.income_tax_rate))

    define_variable(system, "rate_history", formula)
    simulation = traced_simulation()

    simulation.calculate("rate_history", JANUARY)

    assert recorded_reads(simulation, "rate_history") == [
        ("taxes.income_tax_rate", ETERNITY, "default", None)
    ]


def test_node_children_read_before_an_instant_are_recorded_at_the_node(
    system, traced_simulation
):
    def formula(person, period, parameters):
        rate = parameters.taxes.children["income_tax_rate"]
        return person.filled_array(rate(period))

    define_variable(system, "rate_from_children", formula)
    simulation = traced_simulation()

    simulation.calculate("rate_from_children", JANUARY)

    assert recorded_names(simulation, "rate_from_children") == ["taxes"]


def test_scale_brackets_read_before_an_instant_are_recorded_at_the_scale(
    system, traced_simulation
):
    def formula(person, period, parameters):
        bracket = parameters.taxes.social_security_contribution.brackets[1]
        return person.filled_array(bracket.rate(period))

    define_variable(system, "rate_from_brackets", formula)
    simulation = traced_simulation()

    simulation.calculate("rate_from_brackets", JANUARY)

    assert recorded_names(simulation, "rate_from_brackets") == [SOCIAL_SECURITY_SCALE]


def test_child_found_by_path_on_the_undated_tree_is_recorded(system, traced_simulation):
    def formula(person, period, parameters):
        rate = parameters.get_child("taxes.income_tax_rate")
        return person.filled_array(rate(period))

    define_variable(system, "rate_by_path", formula)
    simulation = traced_simulation()

    simulation.calculate("rate_by_path", JANUARY)

    assert recorded_names(simulation, "rate_by_path") == ["taxes.income_tax_rate"]


def test_get_parameter_on_the_undated_tree_is_recorded(system, traced_simulation):
    def formula(person, period, parameters):
        rate = get_parameter(parameters.taxes, "income_tax_rate")
        return person.filled_array(rate(period))

    define_variable(system, "rate_from_helper", formula)
    simulation = traced_simulation()

    simulation.calculate("rate_from_helper", JANUARY)

    assert recorded_names(simulation, "rate_from_helper") == ["taxes"]


def test_labels_of_undated_parameters_are_not_reads(system, traced_simulation):
    def formula(person, period, parameters):
        rate = parameters.taxes.income_tax_rate
        return person.filled_array(len(rate.name) + len(rate.description))

    define_variable(system, "rate_label_size", formula)
    simulation = traced_simulation()

    simulation.calculate("rate_label_size", JANUARY)

    assert recorded_reads(simulation, "rate_label_size") == []


# ----- Reads of a child that is absent ----- #


def read_absent_with_default(taxes):
    return getattr(taxes, "no_such_rate", 0.0)


def read_absent_with_hasattr(taxes):
    return float(hasattr(taxes, "no_such_rate"))


def read_rate_by_key_with_default(node, key):
    try:
        return node[key]
    except KeyError:
        return 0.0


def taxes_node(parameters_at_instant):
    return parameters_at_instant.taxes


def root_node(parameters_at_instant):
    return parameters_at_instant


@pytest.mark.parametrize(
    "select_node, missing_path",
    [(taxes_node, "taxes.no_such_rate"), (root_node, "no_such_rate")],
    ids=["below-a-node", "below-the-root"],
)
@pytest.mark.parametrize(
    "read_absent",
    [read_absent_with_default, read_absent_with_hasattr],
    ids=["getattr-default", "hasattr"],
)
def test_attribute_look_up_that_finds_no_child_records_the_missing_path(
    system, traced_simulation, read_absent, select_node, missing_path
):
    def formula(person, period, parameters):
        return person.filled_array(read_absent(select_node(parameters(period))))

    define_variable(system, "absent_rate", formula)
    simulation = traced_simulation()

    assert simulation.calculate("absent_rate", JANUARY)[0] == 0

    assert recorded_reads(simulation, "absent_rate") == [
        (missing_path, INSTANT, "default", None)
    ]


class ChildThatFailsToBuild:
    """A parameter child that raises whenever a value is asked of it."""

    def __init__(self):
        self.build_attempts = 0

    def _get_at_instant(self, instant):
        self.build_attempts += 1
        raise RuntimeError("this child has no valid value")


@pytest.mark.parametrize(
    "read_absent",
    [read_absent_with_default, read_absent_with_hasattr],
    ids=["getattr-default", "hasattr"],
)
def test_attribute_look_up_that_finds_no_child_builds_no_other_child(
    system, make_simulation, read_absent
):
    """A node at an instant builds each child when it is first read, so a
    child that fails to build stops only the formulas that read it. Looking
    for an absent child must not build the others when traced either."""
    fragile = ParameterNode("fragile", data={"rate": {"values": {"2015-01-01": 1}}})
    failing_child = ChildThatFailsToBuild()
    fragile.children = {**fragile.children, "broken": failing_child}
    system.parameters.add_child("fragile", fragile)

    def formula(person, period, parameters):
        node = parameters(period).fragile
        return person.filled_array(node.rate + read_absent(node))

    define_variable(system, "rate_with_fallback", formula)
    untraced = make_simulation(system, {"salary": 2000}, JANUARY)
    traced = make_simulation(system, {"salary": 2000}, JANUARY)
    traced.trace = True

    assert untraced.calculate("rate_with_fallback", JANUARY)[0] == 1
    assert traced.calculate("rate_with_fallback", JANUARY)[0] == 1

    assert failing_child.build_attempts == 0
    assert recorded_reads(traced, "rate_with_fallback") == [
        ("fragile.rate", INSTANT, "default", 1),
        ("fragile.no_such_rate", INSTANT, "default", None),
    ]


@pytest.mark.parametrize(
    "select_node, missing_path",
    [(taxes_node, "taxes.no_such_rate"), (root_node, "no_such_rate")],
    ids=["below-a-node", "below-the-root"],
)
def test_key_look_up_that_finds_no_child_records_the_missing_path(
    system, traced_simulation, select_node, missing_path
):
    def formula(person, period, parameters):
        node = select_node(parameters(period))
        return person.filled_array(read_rate_by_key_with_default(node, "no_such_rate"))

    define_variable(system, "rate_by_key", formula)
    simulation = traced_simulation()

    assert simulation.calculate("rate_by_key", JANUARY)[0] == 0

    assert recorded_reads(simulation, "rate_by_key") == [
        (missing_path, INSTANT, "default", None)
    ]


def test_key_look_up_before_a_parameter_starts_records_that_parameter(
    system, traced_simulation
):
    """A value given to the parameter at the instant read would change the
    result, so a reform of it must find this formula among its readers."""
    future_rate = Parameter(
        "taxes.future_rate", {"values": {"2030-01-01": {"value": 0.3}}}
    )
    system.parameters.taxes.add_child("future_rate", future_rate)

    def formula(person, period, parameters):
        taxes = parameters(period).taxes
        return person.filled_array(read_rate_by_key_with_default(taxes, "future_rate"))

    define_variable(system, "rate_from_2030", formula)
    simulation = traced_simulation()

    assert simulation.calculate("rate_from_2030", JANUARY)[0] == 0

    assert recorded_reads(simulation, "rate_from_2030") == [
        (future_rate.name, INSTANT, "default", None)
    ]


class Zone(str, enum.Enum):
    NORTH = "north"


def test_key_look_up_by_a_string_enum_member_records_the_member_as_a_path(
    system, traced_simulation
):
    """``str()`` of such a member is ``Zone.NORTH`` from Python 3.12, but the
    look-up uses its string value, so that is the path a value would go to."""

    def formula(person, period, parameters):
        taxes = parameters(period).taxes
        return person.filled_array(read_rate_by_key_with_default(taxes, Zone.NORTH))

    define_variable(system, "rate_by_zone", formula)
    simulation = traced_simulation()

    assert simulation.calculate("rate_by_zone", JANUARY)[0] == 0

    assert recorded_reads(simulation, "rate_by_zone") == [
        ("taxes.north", INSTANT, "default", None)
    ]


def look_up_by_attribute_with_default(node, name):
    return getattr(node, name, 0.0)


def look_up_with_hasattr(node, name):
    return float(hasattr(node, name))


@pytest.mark.parametrize(
    "look_up",
    [
        look_up_by_attribute_with_default,
        look_up_with_hasattr,
        read_rate_by_key_with_default,
    ],
    ids=["getattr-default", "hasattr", "key"],
)
def test_look_up_by_a_dotted_name_records_nothing(system, traced_simulation, look_up):
    """A look-up does not descend into a dotted name, and no child's name has
    a dot, so the look-up finds nothing whatever the parameters hold. Recorded
    at the dotted path, it would also replace the value read there in the flat
    trace."""

    def formula(person, period, parameters):
        taxes = parameters(period).taxes
        dotted_result = look_up(taxes, "housing_tax.rate")
        return person.filled_array(taxes.housing_tax.rate + dotted_result)

    define_variable(system, "rate_and_dotted_look_up", formula)
    rate = system.parameters(INSTANT).taxes.housing_tax.rate
    simulation = traced_simulation()

    assert simulation.calculate("rate_and_dotted_look_up", JANUARY)[0] == rate

    assert recorded_reads(simulation, "rate_and_dotted_look_up") == [
        ("taxes.housing_tax.rate", INSTANT, "default", rate)
    ]
    flat_trace = simulation.tracer.get_serialized_flat_trace()
    assert flat_trace[f"rate_and_dotted_look_up<{JANUARY}, (default)>"][
        "parameters"
    ] == {f"taxes.housing_tax.rate<{INSTANT}, (default)>": rate}


ABOLITION_TREES = {
    "switch-off": {"abolitions": {"income_tax": {"2015-01-01": {"value": False}}}},
    "no-switch": {"abolitions": {"other_variable": {"2015-01-01": {"value": False}}}},
    "no-gov-node": None,
}


@pytest.mark.parametrize("gov_data", ABOLITION_TREES.values(), ids=ABOLITION_TREES)
def test_abolition_check_on_a_tracing_root_records_nothing(
    system, traced_simulation, gov_data
):
    """A country package can switch tracing on in its own parameter root, so
    core's abolition check, a look-up per variable, would reach the trace.
    Core reads the untraced tree instead: the check is not the formula's."""
    if gov_data is not None:
        system.parameters.add_child("gov", ParameterNode("gov", data=gov_data))
    simulation = traced_simulation()
    system.parameters.trace = True
    system.parameters.tracer = simulation.tracer
    system.parameters.branch_name = "default"

    simulation.calculate("income_tax", JANUARY)

    assert recorded_names(simulation, "income_tax") == ["taxes.income_tax_rate"]


def test_absent_child_on_the_undated_tree_records_the_node(system, traced_simulation):
    def formula(person, period, parameters):
        return person.filled_array(float(hasattr(parameters.taxes, "no_such_rate")))

    define_variable(system, "undated_absent", formula)
    simulation = traced_simulation()

    assert simulation.calculate("undated_absent", JANUARY)[0] == 0

    assert recorded_reads(simulation, "undated_absent") == [
        ("taxes", ETERNITY, "default", None)
    ]


# ----- Reads core makes for a variable ----- #


def add_list_parameter(system, name, value):
    system.parameters.taxes.add_child(
        name,
        Parameter(f"taxes.{name}", {"values": {"2015-01-01": {"value": value}}}),
    )


def test_parameter_in_an_adds_list_is_recorded(system, traced_simulation):
    define_variable(
        system,
        "salary_plus_basic_income",
        None,
        adds=["salary", "benefits.basic_income"],
    )
    basic_income = system.parameters(INSTANT).benefits.basic_income
    simulation = traced_simulation()

    assert simulation.calculate("salary_plus_basic_income", JANUARY)[
        0
    ] == pytest.approx(2000 + basic_income)

    assert recorded_reads(simulation, "salary_plus_basic_income") == [
        ("benefits.basic_income", INSTANT, "default", basic_income)
    ]


def test_parameter_in_a_subtracts_list_is_recorded(system, traced_simulation):
    define_variable(
        system, "minus_basic_income", None, subtracts=["benefits.basic_income"]
    )
    basic_income = system.parameters(INSTANT).benefits.basic_income
    simulation = traced_simulation()

    simulation.calculate("minus_basic_income", JANUARY)

    assert recorded_reads(simulation, "minus_basic_income") == [
        ("benefits.basic_income", INSTANT, "default", basic_income)
    ]


def test_parameter_holding_the_adds_list_is_recorded(system, traced_simulation):
    add_list_parameter(system, "income_components", ["salary"])
    define_variable(system, "income_total", None, adds="taxes.income_components")
    simulation = traced_simulation()

    assert simulation.calculate("income_total", JANUARY)[0] == 2000

    assert recorded_reads(simulation, "income_total") == [
        ("taxes.income_components", INSTANT, "default", ["salary"])
    ]


def test_parameter_holding_the_subtracts_list_is_recorded(system, traced_simulation):
    add_list_parameter(system, "income_deductions", ["salary"])
    define_variable(system, "minus_income", None, subtracts="taxes.income_deductions")
    simulation = traced_simulation()

    assert simulation.calculate("minus_income", JANUARY)[0] == -2000

    assert recorded_reads(simulation, "minus_income") == [
        ("taxes.income_deductions", INSTANT, "default", ["salary"])
    ]


def test_uprating_records_the_index_at_both_instants(system, traced_simulation):
    define_variable(system, "uprated_amount", None, uprating="taxes.income_tax_rate")
    rate = system.parameters.taxes.income_tax_rate
    simulation = traced_simulation()
    simulation.set_input("uprated_amount", "2015-01", [100])

    uprated = simulation.calculate("uprated_amount", JANUARY)

    assert uprated[0] == pytest.approx(100 * rate(INSTANT) / rate("2015-01-01"))
    assert recorded_reads(simulation, "uprated_amount") == [
        ("taxes.income_tax_rate", "2015-01-01", "default", rate("2015-01-01")),
        ("taxes.income_tax_rate", INSTANT, "default", rate(INSTANT)),
    ]


def test_uprating_records_the_value_held_before_the_index_starts(
    system, traced_simulation
):
    system.parameters.taxes.add_child(
        "late_index",
        Parameter(
            "taxes.late_index",
            {"values": {"2016-01-01": {"value": 2.0}, "2017-01-01": {"value": 3.0}}},
        ),
    )
    define_variable(system, "late_uprated", None, uprating="taxes.late_index")
    simulation = traced_simulation()
    simulation.set_input("late_uprated", "2015-01", [100])

    assert simulation.calculate("late_uprated", JANUARY)[0] == pytest.approx(150)

    assert recorded_reads(simulation, "late_uprated") == [
        ("taxes.late_index", "2015-01-01", "default", 2.0),
        ("taxes.late_index", INSTANT, "default", 3.0),
    ]


def test_untraced_simulation_records_no_core_reads(system, make_simulation):
    define_variable(system, "added_income", None, adds=["benefits.basic_income"])
    simulation = make_simulation(system, {"salary": 2000}, JANUARY)

    simulation.calculate("added_income", JANUARY)

    assert not hasattr(simulation.tracer, "trees")


def adds_naming_a_parameter(system):
    define_variable(
        system, "core_read_case", None, adds=["salary", "benefits.basic_income"]
    )
    return 2000 + system.parameters(INSTANT).benefits.basic_income, {}


def adds_held_in_a_parameter(system):
    add_list_parameter(system, "income_components", ["salary"])
    define_variable(system, "core_read_case", None, adds="taxes.income_components")
    return 2000, {}


def subtracts_naming_a_parameter(system):
    define_variable(system, "core_read_case", None, subtracts=["benefits.basic_income"])
    return -system.parameters(INSTANT).benefits.basic_income, {}


def subtracts_held_in_a_parameter(system):
    add_list_parameter(system, "income_deductions", ["salary"])
    define_variable(system, "core_read_case", None, subtracts="taxes.income_deductions")
    return -2000, {}


def uprating_an_earlier_input(system):
    define_variable(system, "core_read_case", None, uprating="taxes.income_tax_rate")
    rate = system.parameters.taxes.income_tax_rate
    return 100 * rate(INSTANT) / rate("2015-01-01"), {"2015-01": [100]}


@pytest.mark.parametrize(
    "define_case",
    [
        adds_naming_a_parameter,
        adds_held_in_a_parameter,
        subtracts_naming_a_parameter,
        subtracts_held_in_a_parameter,
        uprating_an_earlier_input,
    ],
    ids=[
        "adds-naming-a-parameter",
        "adds-held-in-a-parameter",
        "subtracts-naming-a-parameter",
        "subtracts-held-in-a-parameter",
        "uprating",
    ],
)
def test_untraced_simulation_does_not_call_the_read_recorder(
    system, make_simulation, monkeypatch, define_case
):
    """A country package's tests can build a stand-in simulation that copies
    only some of core's methods, as policyengine-us's Social Security state
    tests do, so an untraced calculation must not call the method that
    records reads."""
    expected, earlier_inputs = define_case(system)
    simulation = make_simulation(system, {"salary": 2000}, JANUARY)
    for input_period, input_values in earlier_inputs.items():
        simulation.set_input("core_read_case", input_period, input_values)
    monkeypatch.delattr(Simulation, "_record_parameter_read")

    assert simulation.calculate("core_read_case", JANUARY)[0] == pytest.approx(expected)


# ----- Branches ----- #


def test_nested_branch_keeps_each_formulas_reads_under_its_own_branch(
    system, traced_simulation
):
    def formula(person, period, parameters):
        rate_before = parameters(period).taxes.income_tax_rate
        person.simulation.get_branch("inner").calculate("income_tax", period)
        rate_after = parameters(period).taxes.income_tax_rate
        return person.filled_array(rate_before + rate_after)

    define_variable(system, "reads_around_a_branch", formula)
    simulation = traced_simulation()

    simulation.calculate("reads_around_a_branch", JANUARY)

    outer_node = trace_node(simulation, "reads_around_a_branch")
    assert [(read.name, read.branch_name) for read in outer_node.parameters] == [
        ("taxes.income_tax_rate", "default"),
        ("taxes.income_tax_rate", "default"),
    ]
    inner_node = next(
        child for child in outer_node.children if child.name == "income_tax"
    )
    assert [(read.name, read.branch_name) for read in inner_node.parameters] == [
        ("taxes.income_tax_rate", "inner")
    ]


def test_core_reads_in_a_branch_carry_the_branch_name(system, traced_simulation):
    define_variable(system, "added_income", None, adds=["benefits.basic_income"])
    simulation = traced_simulation()

    simulation.get_branch("inner").calculate("added_income", JANUARY)

    assert [read[2] for read in recorded_reads(simulation, "added_income")] == ["inner"]


# ----- Which calculations ran a formula ----- #


def test_trace_marks_a_calculation_that_ran_its_formula(system, traced_simulation):
    simulation = traced_simulation()

    simulation.calculate("income_tax", JANUARY)

    assert trace_node(simulation, "income_tax").formula_ran is True
    assert trace_node(simulation, "salary").formula_ran is False


def test_trace_marks_adds_as_having_run(system, traced_simulation):
    define_variable(system, "added_salary", None, adds=["salary"])
    simulation = traced_simulation()

    simulation.calculate("added_salary", JANUARY)

    assert trace_node(simulation, "added_salary").formula_ran is True


def test_trace_marks_a_formula_skipped_by_defined_for(system, traced_simulation):
    def formula(person, period, parameters):
        return person.filled_array(parameters(period).taxes.income_tax_rate)

    define_variable(system, "never_eligible", None, default_value=0)
    define_variable(system, "masked_rate", formula, defined_for="never_eligible")
    simulation = traced_simulation()

    simulation.calculate("masked_rate", JANUARY)

    assert trace_node(simulation, "masked_rate").formula_ran is False
    assert recorded_reads(simulation, "masked_rate") == []


def test_simple_tracer_accepts_a_formula_run(system, make_simulation):
    simulation = make_simulation(system, {"salary": 2000}, JANUARY)
    simulation.trace = True
    simulation.tracer = SimpleTracer()

    assert simulation.calculate("income_tax", JANUARY)[0] > 0
