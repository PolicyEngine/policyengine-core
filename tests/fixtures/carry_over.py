"""Variables, simulations and a reference rule for the auto-carry-over tests.

Shared by ``tests/core/test_carry_over_order.py`` (regressions) and
``tests/core/test_carry_over_order_property.py`` (properties).

The reference rule, for a variable with no formula result for a period: an
input stored for the period itself is read back as stored; otherwise the
period takes the input stored for the latest-starting period that starts no
later than it (on a tie, the one that ends last, then the larger unit),
preferring inputs at the
variable's own definition-period unit, masked by the variable's
``defined_for`` at the period; with no such input, the default.
"""

from __future__ import annotations

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.parameters import ParameterNode
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable

COUNT = 2
FORMULA_END_YEAR = 2013


class carried(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly flow input with no formula"


class carried_count(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly integer input with no formula"


class eligible(Variable):
    value_type = bool
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly boolean input that carried_if_eligible is defined for"


class carried_if_eligible(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    defined_for = "eligible"
    label = "Yearly flow input with no formula, defined for eligible people"


class formula_until_2013(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    end = f"{FORMULA_END_YEAR}-12-31"
    label = "Yearly variable whose formula ends after 2013"

    def formula(person, period, parameters):
        return person("carried", period) + 5


class carried_monthly(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly flow input with no formula"


class year_input_without_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    set_input = None
    label = "Yearly input with no set_input helper, stored at any period"


class month_input_without_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    set_input = None
    label = "Monthly input with no set_input helper, stored at any period"


class day_input_without_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.DAY
    set_input = None
    label = "Daily input with no set_input helper, stored at any period"


class uprated_input_without_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    set_input = None
    uprating = "carry_over_test.index"
    label = "Yearly uprated input with no set_input helper"


def _calculate_while_setting(holder, period, array):
    """A ``set_input`` helper that calculates another variable first."""
    holder.simulation.calculate("formula_until_2013", "2012")
    holder._set(period.this_year, array)


class input_with_calculating_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    set_input = _calculate_while_setting
    label = "Yearly input whose set_input helper calculates"


VARIABLES = (
    carried,
    carried_count,
    eligible,
    carried_if_eligible,
    formula_until_2013,
    carried_monthly,
    year_input_without_helper,
    month_input_without_helper,
    day_input_without_helper,
    uprated_input_without_helper,
    input_with_calculating_helper,
)


def build_system() -> CountryTaxBenefitSystem:
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = True
    system.parameters.add_child(
        "carry_over_test",
        ParameterNode(
            "carry_over_test", data={"index": {"values": {"2010-01-01": 1.0}}}
        ),
    )
    system.add_variables(*VARIABLES)
    return system


def simulation(system, inputs):
    """A simulation of ``COUNT`` people given ``{variable: {period: values}}``."""
    built = SimulationBuilder().build_default_simulation(system, count=COUNT)
    for variable, values in inputs.items():
        for period, array in values.items():
            built.set_input(variable, period, np.array(array))
    return built


def alone(system, inputs, variable, period):
    """What a fresh simulation given ``inputs`` returns for one request."""
    return simulation(system, inputs).calculate(variable, period)


def request(built, kind, variable, period):
    if kind == "add":
        built.calculate_add(variable, period)
    elif kind == "divide":
        built.calculate_divide(variable, period)
    else:
        built.calculate(variable, period)


def _stored_inputs(system, inputs, variable):
    """The inputs ``set_input`` stores: it ignores those after the ``end``."""
    end = system.get_variable(variable).end
    return {
        periods.period(stored): values
        for stored, values in inputs.get(variable, {}).items()
        if end is None or periods.period(stored).start.date <= end
    }


def reference(system, inputs, variable, period):
    """The carry-over rule, computed from the inputs alone.

    ``period`` is in the variable's definition period.
    """
    period = periods.period(period)
    default = system.get_variable(variable).default_array(COUNT)
    stored_inputs = _stored_inputs(system, inputs, variable)
    if period in stored_inputs:
        # An input for the period itself is read back as stored: neither a
        # formula nor ``defined_for`` applies to it.
        return np.array(stored_inputs[period], dtype=default.dtype)
    if variable == "formula_until_2013" and period.start.year <= FORMULA_END_YEAR:
        return (reference(system, inputs, "carried", period) + 5).astype(default.dtype)
    earlier = [stored for stored in stored_inputs if stored.start <= period.start]
    own_unit = [
        stored
        for stored in earlier
        if stored.unit == system.get_variable(variable).definition_period
    ]
    if earlier:
        latest = max(
            own_unit or earlier,
            key=lambda stored: (
                stored.start,
                stored.stop,
                periods.unit_weight(stored.unit),
                str(stored),
            ),
        )
        value = np.array(stored_inputs[latest]).astype(default.dtype)
    else:
        value = default
    if variable == "carried_if_eligible":
        value = np.where(reference(system, inputs, "eligible", period), value, default)
    return value
