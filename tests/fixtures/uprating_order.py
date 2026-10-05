"""Variables, systems and a reference rule for the uprating order tests.

Shared by ``tests/core/test_uprating_order.py`` (regressions) and
``tests/core/test_uprating_order_property.py`` (properties). The simulation
helpers are the auto-carry-over tests' (``tests/fixtures/carry_over.py``).

The reference rule, for a variable with ``uprating`` and no formula result
for a period: an input stored for the period itself is read back as stored;
otherwise the latest input stored in the variable's own unit for a period
that starts before it, times the ratio of the uprating index at the two
period starts, cast to the variable's type. With no such input, the period
gets what auto-carry-over gives it (the input stored for the latest-starting
period that starts no later than it, preferring the variable's own unit) or,
without auto-carry-over or such an input, the default. Either way the value
is masked by the variable's ``defined_for`` at the period.

"Latest" breaks ties the same way on both paths: of inputs that start on the
same day (a yearly variable stores ``year:2012:2`` as given, beside
``2012``), the one that ends last, then the larger unit, then the period's
string form. An input in the variable's own unit that starts on the period's
own first day but is longer than it (``year:2012:2`` for ``2012``) is not an
uprating source: the period gets the uprated earlier input if there is one,
and only otherwise what auto-carry-over gives it (see
PolicyEngine/policyengine-core#583).
"""

from __future__ import annotations

import sys

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.parameters import ParameterNode, get_parameter
from policyengine_core.simulations.simulation import _uprating_index_value
from policyengine_core.variables import QuantityType, Variable
from tests.fixtures.carry_over import COUNT, alone, request, simulation

__all__ = [
    "COUNT",
    "FLAT",
    "INDEX",
    "MONTHLY_INDEX",
    "UPRATING",
    "alone",
    "assert_bitwise_equal",
    "build_system",
    "can_store_on_disk",
    "index",
    "reference",
    "request",
    "simulation",
]

# 3.7% a year: ratios that are not exact in float32, so rounding at an
# intermediate year shows.
INDEX = {f"{year}-01-01": 100 * 1.037 ** (year - 2010) for year in range(2010, 2021)}
UPRATING = "uprating_order.index"
# 0.3% a month, so that which month of a year a value is uprated from shows.
MONTHLY_INDEX = {
    f"{year}-{month:02d}-01": 100 * 1.003 ** ((year - 2010) * 12 + month - 1)
    for year in range(2010, 2021)
    for month in range(1, 13)
}
MONTHLY_UPRATING = "uprating_order.monthly_index"
# An index that never moves: uprating by it multiplies by exactly 1.
FLAT = "uprating_order.flat"

# ``OnDiskStorage`` names a value's file after its period, and Windows does
# not allow ":" in a file name. So on Windows a period several units long
# (``year:2012:2``) cannot be stored on disk: ``numpy.save`` raises
# ``OSError``. The tests keep those periods in memory there (see
# PolicyEngine/policyengine-core#526).
DISK_STORES_ANY_PERIOD = sys.platform != "win32"


def can_store_on_disk(period) -> bool:
    """Whether ``OnDiskStorage`` can store a value for ``period`` here."""
    return DISK_STORES_ANY_PERIOD or ":" not in str(periods.period(period))


class uprated(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = UPRATING
    label = "Uprated yearly input"


class uprated_count(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = UPRATING
    label = "Uprated yearly integer input"


class eligible(Variable):
    value_type = bool
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly boolean input that uprated_if_eligible is defined for"


class uprated_if_eligible(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = UPRATING
    defined_for = "eligible"
    label = "Uprated yearly input, defined for eligible people"


class uprated_with_default(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = UPRATING
    default_value = 10
    label = "Uprated yearly input with a non-zero default"


class uprated_any_unit(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = UPRATING
    set_input = None
    label = "Uprated yearly input with no set_input helper, stored at any period"


class uprated_monthly(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    uprating = UPRATING
    label = "Uprated monthly input (a flow: set_input_divide_by_period)"


class uprated_monthly_stock(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    quantity_type = QuantityType.STOCK
    uprating = MONTHLY_UPRATING
    label = "Uprated monthly input (a stock: set_input_dispatch_by_period)"


class uprated_flat(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = FLAT
    set_input = None
    label = "Yearly input uprated by an index that never moves, stored at any period"


class uprated_daily_flat(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.DAY
    uprating = FLAT
    set_input = None
    label = "Daily input uprated by an index that never moves, stored at any period"


class carried_any_unit(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    set_input = None
    label = "uprated_flat without uprating: a yearly input stored at any period"


VARIABLES = (
    uprated,
    uprated_count,
    eligible,
    uprated_if_eligible,
    uprated_with_default,
    uprated_any_unit,
    uprated_monthly,
    uprated_monthly_stock,
    uprated_flat,
    uprated_daily_flat,
    carried_any_unit,
)


def build_system(auto_carry_over: bool = True) -> CountryTaxBenefitSystem:
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = auto_carry_over
    system.parameters.add_child(
        "uprating_order",
        ParameterNode(
            "uprating_order",
            data={
                "index": {"values": INDEX},
                "monthly_index": {"values": MONTHLY_INDEX},
                "flat": {"values": {"2010-01-01": 100}},
            },
        ),
    )
    system.add_variables(*VARIABLES)
    return system


def index(year: int) -> float:
    return INDEX[f"{year}-01-01"]


def assert_bitwise_equal(actual, expected, message=""):
    """Same dtype, shape and bytes: ``-0.0`` and ``0.0`` differ."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype, (
        f"{message}: {actual.dtype} != {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{message}: {actual.shape} != {expected.shape}"
    )
    assert actual.tobytes() == expected.tobytes(), f"{message}: {actual} != {expected}"


def _stored_inputs(system, inputs, variable):
    """The inputs as ``set_input`` stores them, cast to the variable's type.

    The reference rule covers inputs in the variable's own unit, and inputs
    in any unit to a variable with no ``set_input`` helper (stored as given).
    A helper would split or copy an input in another unit across the
    variable's own periods first, which the rule does not model.
    """
    definition = system.get_variable(variable)
    stored = {}
    for period, values in inputs.get(variable, {}).items():
        period = periods.period(period)
        if period.unit != definition.definition_period and definition.set_input:
            raise ValueError(
                f"{variable}: an input for {period} goes through set_input, "
                "which the reference rule does not model"
            )
        stored[period] = np.asarray(values, dtype=definition.dtype)
    return stored


def _latest(stored_period):
    """Of inputs in one unit class, the latest: the one that starts last; on
    a tie, the one that ends last, then the larger unit, then the period's
    string form. Distinct periods never tie, so the choice does not depend on
    the order the inputs are listed in."""
    try:
        # A period ending after 9999-12-31 has no ``stop``: it ends last.
        ends = (0, stored_period.stop)
    except OverflowError:
        ends = (1,)
    return (
        stored_period.start,
        ends,
        periods.unit_weight(stored_period.unit),
        str(stored_period),
    )


def _uprating_factor(system, definition, earlier, period):
    parameter = get_parameter(system.parameters, definition.uprating)
    value_then = _uprating_index_value(parameter, earlier.start)
    value_now = _uprating_index_value(parameter, period.start)
    if value_then is None or value_now is None or value_then == 0:
        return 1
    return value_now / value_then


def reference(system, inputs, variable, period):
    """The uprating and carry-over rules, computed from the inputs alone.

    ``period`` is in the variable's definition period.
    """
    definition = system.get_variable(variable)
    period = periods.period(period)
    default = definition.default_array(COUNT)
    stored = _stored_inputs(system, inputs, variable)
    if period in stored:
        # An input for the period itself is read back as stored:
        # ``defined_for`` does not apply to it.
        return stored[period]
    earlier = [
        stored_period
        for stored_period in stored
        if stored_period.unit == definition.definition_period
        and stored_period.start < period.start
    ]
    carried = [
        stored_period for stored_period in stored if stored_period.start <= period.start
    ]
    if definition.uprating is not None and earlier:
        latest = max(earlier, key=_latest)
        value = stored[latest] * _uprating_factor(system, definition, latest, period)
    elif system.auto_carry_over_input_variables and carried:
        own_unit = [
            stored_period
            for stored_period in carried
            if stored_period.unit == definition.definition_period
        ]
        latest = max(own_unit or carried, key=_latest)
        value = stored[latest]
    else:
        value = default
    if definition.defined_for is not None:
        mask = reference(system, inputs, definition.defined_for, period)
        value = np.where(mask, value, default)
    return np.asarray(value).astype(definition.dtype)
