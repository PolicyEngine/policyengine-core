"""Calls made inside a calculation get plain arrays in a Microsimulation.

``Microsimulation.calculate``, ``calculate_add`` and ``calculate_divide``
return a weighted ``MicroSeries`` with enums decoded, for a caller that wants
weighted results. Core also calls these methods for itself while it
calculates: for each term of ``adds`` and ``subtracts``, for a
``defined_for`` mask, for each month of a monthly variable read over a year,
and so on. Each such call built weights from ``household_weight`` and decoded
enums, and core then threw the weights away before storing the value.
Decoding broke enum reads with ADD and DIVIDE (core issue #658).

Now a call that arrives while another of these three methods is running, in
the same thread or a context copied from it, is a call from inside a
calculation. Left at their defaults, its ``use_weights`` and ``decode_enums``
mean a plain array with enums left as codes, as a ``Simulation`` returns. A
call from outside a calculation gets weighted values, as before. An explicit
argument wins in both cases.
"""

import sys
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from microdf import MicroDataFrame, MicroSeries

from policyengine_core.enums import EnumArray
from policyengine_core.model_api import ADD, DIVIDE, MONTH
from policyengine_core.periods import period as make_period
from policyengine_core.simulations import Microsimulation, Simulation
from tests.fixtures.microsimulation_internal_calls import (
    RECEIVED,
    NoChange,
    ProbeInterrupt,
    microsimulation,
    simulation,
)
from tests.fixtures.microsimulation_option_reads import DATASET_YEAR, READERS

PERSON_WEIGHTS = [10.0, 10.0, 20.0]
GRANT = [120.0, 240.0, 0.0]
GRANT_A_MONTH = [10.0, 20.0, 0.0]
PAY = [78.0, 156.0, 390.0]
TENURE_CODES = [0, 1, 0]
TENURE_NAMES = ["OWNER", "RENTER", "OWNER"]


@pytest.fixture(autouse=True)
def clear_received():
    RECEIVED.clear()
    yield
    RECEIVED.clear()


def assert_array(value, expected):
    """``value`` is a plain NumPy array (not an ``EnumArray``) of ``expected``."""
    assert type(value) is np.ndarray, type(value).__name__
    np.testing.assert_array_equal(value, expected)


def assert_codes(value, expected=TENURE_CODES):
    assert type(value) is EnumArray, type(value).__name__
    np.testing.assert_array_equal(np.asarray(value), expected)


def assert_weighted(value, expected, weights=PERSON_WEIGHTS):
    assert isinstance(value, MicroSeries), type(value).__name__
    assert value.tolist() == list(expected)
    np.testing.assert_array_equal(value.weights, weights)


@contextmanager
def weight_reads():
    """Record the variable of each ``Microsimulation.get_weights`` call."""
    variables = []
    get_weights = Microsimulation.get_weights

    def record(self, variable_name, *arguments, **keywords):
        variables.append(variable_name)
        return get_weights(self, variable_name, *arguments, **keywords)

    with mock.patch.object(Microsimulation, "get_weights", record):
        yield variables


@contextmanager
def household_weight_calculations():
    """Record the period of each ``_calculate`` of ``household_weight``."""
    periods = []
    calculate = Simulation._calculate

    def record(self, variable_name, period=None):
        if variable_name == "household_weight":
            periods.append(str(period))
        return calculate(self, variable_name, period)

    with mock.patch.object(Simulation, "_calculate", record):
        yield periods


# Calls from inside a calculation.

READS_OF_ITS_OWN_SIMULATION = {
    "population read": (assert_array, GRANT),
    "population enum read": (assert_codes, TENURE_CODES),
    "calculate": (assert_array, GRANT),
    "calculate enum": (assert_codes, TENURE_CODES),
    "calculate again": (assert_array, PAY),
    "calculate_add": (assert_array, PAY),
    "calculate_divide": (assert_array, GRANT),
    "calculate_divide month": (assert_array, GRANT_A_MONTH),
}


@pytest.mark.parametrize("trace", [False, True], ids=["trace off", "trace on"])
@pytest.mark.parametrize("label", list(READS_OF_ITS_OWN_SIMULATION))
def test_formula_call_on_its_own_simulation_gets_a_plain_array(label, trace):
    microsimulation(trace=trace).calculate("probe_reads_own_simulation", DATASET_YEAR)

    check, expected = READS_OF_ITS_OWN_SIMULATION[label]
    check(RECEIVED[label], expected)
    # The fast cache holds the first read of ``pay_added``, so with tracing
    # off the second read comes from it; with tracing on, Simulation.calculate
    # skips the fast cache.
    assert RECEIVED["fast cache holds the first read"]


@pytest.mark.parametrize(
    "label,expected",
    [
        ("calculate use_weights=True", GRANT),
        ("calculate_add use_weights=True", PAY),
        ("calculate_divide use_weights=True", GRANT),
    ],
)
def test_explicit_use_weights_inside_a_formula_gets_weights(label, expected):
    microsimulation().calculate("probe_reads_own_simulation", DATASET_YEAR)

    assert_weighted(RECEIVED[label], expected)


def test_explicit_decode_enums_inside_a_formula_gets_names():
    microsimulation().calculate("probe_reads_own_simulation", DATASET_YEAR)

    names = RECEIVED["calculate decode_enums=True"]
    assert type(names) is np.ndarray
    assert names.tolist() == TENURE_NAMES


@pytest.mark.parametrize("trace", [False, True], ids=["trace off", "trace on"])
def test_formula_call_on_a_branch_gets_a_plain_array(trace):
    microsimulation(trace=trace).calculate("probe_reads_a_branch", DATASET_YEAR)

    assert_array(RECEIVED["branch calculate"], GRANT)
    assert_codes(RECEIVED["branch calculate enum"])


def test_reform_formula_call_on_its_baseline_gets_a_plain_array():
    microsimulation(reform=NoChange).calculate("probe_reads_its_baseline", DATASET_YEAR)

    assert_array(RECEIVED["baseline calculate"], GRANT)


def test_microsimulation_built_inside_a_microsimulation_formula_gets_arrays():
    microsimulation().calculate("probe_builds_a_microsimulation", DATASET_YEAR)

    assert_array(RECEIVED["new microsimulation calculate"], GRANT)


def test_microsimulation_built_inside_a_simulation_formula_gets_weights():
    # Only a running Microsimulation call makes a call internal. A plain
    # Simulation's calculation does not, so this one is weighted, as before.
    simulation().calculate("probe_builds_a_microsimulation", DATASET_YEAR)

    assert_weighted(RECEIVED["new microsimulation calculate"], GRANT)


def test_worker_thread_follows_its_context():
    # A thread started inside a formula runs in an empty context, unless
    # Python is set to give new threads a copy of the starting thread's
    # context (``-X thread_inherit_context=1``, the default on free-threaded
    # builds). In an empty context the thread's call is a call from outside a
    # calculation. A context copied while the formula runs carries the
    # formula's calculation with it.
    microsimulation().calculate("probe_reads_from_worker_threads", DATASET_YEAR)

    if getattr(sys.flags, "thread_inherit_context", 0):
        assert_array(RECEIVED["thread calculate"], GRANT)
    else:
        assert_weighted(RECEIVED["thread calculate"], GRANT)
    assert_array(RECEIVED["thread calculate, copied context"], GRANT)


def test_context_copied_in_a_formula_gets_weights_after_the_calculation_ends():
    simulation_under_test = microsimulation()
    simulation_under_test.calculate("probe_keeps_its_context", DATASET_YEAR)

    result = RECEIVED["context"].run(
        simulation_under_test.calculate, "yearly_grant", DATASET_YEAR
    )

    assert_weighted(result, GRANT)


def test_context_copied_in_a_nested_call_stays_inside_while_an_outer_call_runs():
    microsimulation().calculate("probe_reads_in_a_nested_context", DATASET_YEAR)

    assert_array(RECEIVED["nested context calculate"], GRANT)


def test_call_in_a_copied_context_stays_inside_after_the_first_call_ends():
    # The worker's own calculate call is still running when it reads, so its
    # reads are inside a calculation, though the call whose context it copied
    # has ended.
    try:
        microsimulation().calculate("probe_starts_a_worker", DATASET_YEAR)
    finally:
        if "release worker" in RECEIVED:
            RECEIVED["release worker"].set()
            RECEIVED["worker"].join(timeout=30)

    assert not RECEIVED["worker"].is_alive()
    assert_array(RECEIVED["worker calculate"], GRANT)
    assert_codes(RECEIVED["worker calculate enum"])


# Calls from outside a calculation: weighted, as before.


def test_top_level_calculate_is_weighted_and_decoded():
    simulation_under_test = microsimulation()

    assert_weighted(simulation_under_test.calculate("yearly_grant"), GRANT)
    assert_weighted(simulation_under_test.calculate("yearly_tenure"), TENURE_NAMES)
    assert_weighted(simulation_under_test.calculate("pay_added"), PAY)
    # The second read comes from Simulation.calculate's fast cache.
    assert ("pay_added", make_period(DATASET_YEAR)) in simulation_under_test._fast_cache
    assert_weighted(simulation_under_test.calculate("pay_added"), PAY)


@pytest.mark.parametrize(
    "method,variable,period,expected",
    [
        ("calculate", "pay_and_grant", DATASET_YEAR, [139.5, 279.0, 97.5]),
        ("calculate", "pay_if_granted", DATASET_YEAR, [78.0, 156.0, 0.0]),
        ("calculate", "monthly_pay", DATASET_YEAR, PAY),
        ("calculate", "yearly_grant", "2022-06", GRANT_A_MONTH),
        ("calculate", "pay_added", DATASET_YEAR, PAY),
        ("calculate_add", "monthly_pay", DATASET_YEAR, PAY),
        ("calculate_divide", "yearly_grant", DATASET_YEAR, GRANT),
        ("calculate_divide", "yearly_grant", "2022-06", GRANT_A_MONTH),
        ("calculate_output", "pay_and_grant", DATASET_YEAR, [139.5, 279.0, 97.5]),
    ],
    ids=lambda argument: str(argument) if isinstance(argument, str) else None,
)
def test_top_level_call_builds_one_weighted_series(method, variable, period, expected):
    simulation_under_test = microsimulation()

    with weight_reads() as variables:
        result = getattr(simulation_under_test, method)(variable, period)

    assert_weighted(result, expected)
    # Core's own reads (the terms of ``adds`` and ``subtracts``, the
    # ``defined_for`` mask, each month of a monthly variable read over a year,
    # the year of a yearly variable read over a month) build no weights.
    assert variables == [variable]


def test_top_level_calculate_dataframe_is_weighted():
    result = microsimulation().calculate_dataframe(
        ["yearly_grant", "pay_and_grant", "yearly_tenure"]
    )

    assert isinstance(result, MicroDataFrame)
    assert result["yearly_grant"].tolist() == GRANT
    assert result["pay_and_grant"].tolist() == [139.5, 279.0, 97.5]
    assert result["yearly_tenure"].tolist() == TENURE_NAMES
    np.testing.assert_array_equal(result.weights, PERSON_WEIGHTS)


def test_calculate_dataframe_in_a_formula_has_names_and_weights():
    # A frame is for reading by column, inside a formula or not: a filter such
    # as ``frame[frame.yearly_tenure == "OWNER"]`` needs member names.
    microsimulation().calculate("probe_reads_a_dataframe", DATASET_YEAR)

    weighted = RECEIVED["calculate_dataframe"]
    assert isinstance(weighted, MicroDataFrame)
    assert weighted["yearly_grant"].tolist() == GRANT
    assert weighted["yearly_tenure"].tolist() == TENURE_NAMES
    np.testing.assert_array_equal(weighted.weights, PERSON_WEIGHTS)

    unweighted = RECEIVED["calculate_dataframe use_weights=False"]
    assert type(unweighted) is pd.DataFrame
    assert unweighted["yearly_grant"].tolist() == GRANT
    assert unweighted["yearly_tenure"].tolist() == TENURE_NAMES


@pytest.mark.parametrize(
    "use_weights,expected_weight_reads",
    [(True, ["yearly_grant"]), (False, [])],
)
def test_calculate_dataframe_builds_weights_only_for_the_frame(
    use_weights, expected_weight_reads
):
    simulation_under_test = microsimulation()

    with weight_reads() as variables:
        simulation_under_test.calculate_dataframe(
            ["yearly_grant", "pay_and_grant", "yearly_tenure"],
            use_weights=use_weights,
        )

    assert variables == expected_weight_reads


def test_calculate_dataframe_maps_columns_to_the_first_variables_entity():
    result = microsimulation().calculate_dataframe(
        ["yearly_grant", "household_yearly_grant"]
    )

    assert result["yearly_grant"].tolist() == GRANT
    assert result["household_yearly_grant"].tolist() == [12.0, 12.0, 36.0]
    np.testing.assert_array_equal(result.weights, PERSON_WEIGHTS)


def test_calculate_dataframe_rejects_an_unknown_variable():
    with pytest.raises(ValueError, match="no_such_variable does not exist"):
        microsimulation().calculate_dataframe(["yearly_grant", "no_such_variable"])


def test_top_level_explicit_arguments_win():
    simulation_under_test = microsimulation()

    assert_array(
        simulation_under_test.calculate("pay_and_grant", use_weights=False),
        [139.5, 279.0, 97.5],
    )
    assert_weighted(
        simulation_under_test.calculate("yearly_tenure", decode_enums=False),
        TENURE_CODES,
    )
    assert_array(
        simulation_under_test.calculate_add(
            "monthly_pay", DATASET_YEAR, use_weights=False
        ),
        PAY,
    )
    assert_array(
        simulation_under_test.calculate_divide(
            "yearly_grant", DATASET_YEAR, use_weights=False
        ),
        GRANT,
    )


def test_household_weight_is_calculated_once_per_top_level_call():
    simulation_under_test = microsimulation()

    with household_weight_calculations() as periods:
        simulation_under_test.calculate("pay_and_grant")

    assert periods == [DATASET_YEAR]


# After errors, calls from outside a calculation are weighted again.


@pytest.mark.parametrize(
    "method,variable,period,error,options",
    [
        ("calculate", "probe_raises", DATASET_YEAR, RuntimeError, {}),
        ("calculate", "probe_raises", DATASET_YEAR, RuntimeError, {"trace": True}),
        ("calculate", "probe_raises_base_exception", DATASET_YEAR, ProbeInterrupt, {}),
        ("calculate_add", "probe_monthly_raises", DATASET_YEAR, RuntimeError, {}),
        ("calculate_divide", "probe_raises", "2022-06", RuntimeError, {}),
        ("calculate_divide", "probe_raises", DATASET_YEAR, RuntimeError, {}),
    ],
    ids=[
        "formula raises",
        "formula raises, trace on",
        "formula raises a BaseException",
        "ADD term raises",
        "month DIVIDE term raises",
        "year DIVIDE term raises",
    ],
)
def test_call_after_an_error_is_weighted(method, variable, period, error, options):
    simulation_under_test = microsimulation(**options)

    with pytest.raises(error):
        getattr(simulation_under_test, method)(variable, period)

    assert_weighted(simulation_under_test.calculate("yearly_tenure"), TENURE_NAMES)
    assert_weighted(microsimulation().calculate("yearly_grant"), GRANT)


def test_formula_that_catches_a_branch_error_still_gets_an_array():
    simulation_under_test = microsimulation()

    simulation_under_test.calculate("probe_catches_a_branch_error")

    assert_array(RECEIVED["calculate after a caught branch error"], GRANT)
    assert_weighted(simulation_under_test.calculate("yearly_grant"), GRANT)


def test_calculation_that_runs_again_after_an_input_change_gets_arrays():
    # The probe reads a branch, then sets an input there that drops what it
    # read, so core runs its calculation again, in both classes.
    plain = simulation()
    expected = plain.calculate("probe_sets_an_input")
    assert RECEIVED["runs"] == 2
    RECEIVED.clear()
    simulation_under_test = microsimulation()

    result = simulation_under_test.calculate("probe_sets_an_input")

    assert RECEIVED["runs"] == 2
    assert_array(RECEIVED["branch calculate before the input"], expected)
    assert_array(RECEIVED["branch calculate after the input"], expected)
    assert_weighted(result, expected)
    assert_weighted(simulation_under_test.calculate("yearly_grant"), GRANT)


# Values: the same as a Simulation's.

REQUESTS = [
    *[
        (reader.__name__, "2022-06" if reader.definition_period == MONTH else "2022")
        for reader in READERS
    ],
    ("pay_and_grant", DATASET_YEAR),
    ("pay_if_granted", DATASET_YEAR),
    ("monthly_pay", DATASET_YEAR),
    ("yearly_grant", "2022-06"),
    ("owner_by_add", DATASET_YEAR),
    ("owner_by_divide", DATASET_YEAR),
    ("owner_twelfth", "2022-06"),
]


def stored_arrays(simulation_under_test) -> dict:
    """Every array the simulation stores, by variable and period."""
    arrays = {}
    for name in simulation_under_test.tax_benefit_system.variables:
        for period in simulation_under_test.get_known_periods(name):
            arrays[(name, str(period))] = simulation_under_test.get_array(name, period)
    return arrays


def describe(array) -> tuple:
    """The type, shape, contents and writability of an array."""
    array = np.asarray(array)
    # An object array's bytes are the addresses of its items.
    contents = array.tolist() if array.dtype == object else array.tobytes()
    return array.dtype.str, array.shape, contents, array.flags.writeable


def test_microsimulation_stores_and_returns_what_a_simulation_does():
    plain, micro = simulation(), microsimulation()

    for variable, period in REQUESTS:
        plain_result = plain.calculate(variable, period)
        micro_result = micro.calculate(variable, period, use_weights=False)
        assert type(micro_result) is type(plain_result), variable
        assert describe(micro_result) == describe(plain_result), variable
        # Top-level weighted reads store the same values too.
        micro.calculate(variable, period)

    plain_arrays, micro_arrays = stored_arrays(plain), stored_arrays(micro)
    assert sorted(micro_arrays) == sorted(plain_arrays)
    for key, plain_array in plain_arrays.items():
        assert describe(micro_arrays[key]) == describe(plain_array), key


# Enum reads with ADD and DIVIDE (core issue #658).

ENUM_READS = [
    ("yearly_tenure", DATASET_YEAR, ADD),
    ("yearly_tenure", DATASET_YEAR, DIVIDE),
    ("yearly_tenure", "2022-06", DIVIDE),
]


@pytest.mark.parametrize(
    "variable,period,option", ENUM_READS, ids=lambda argument: str(argument)
)
def test_enum_read_with_an_option_matches_a_simulation(variable, period, option):
    plain_read, micro_read = (
        each.populations["person"](variable, period, options=[option])
        for each in (simulation(), microsimulation())
    )

    assert type(micro_read) is type(plain_read) is EnumArray
    assert describe(micro_read) == describe(plain_read)


@pytest.mark.parametrize(
    "variable,period,expected",
    [
        ("owner_by_add", DATASET_YEAR, [True, False, True]),
        ("owner_by_divide", DATASET_YEAR, [True, False, True]),
        # Stored as the variable's float32.
        ("owner_twelfth", "2022-06", [0.0, np.float32(1 / 12), 0.0]),
    ],
)
def test_formula_reading_an_enum_with_an_option_matches_a_simulation(
    variable, period, expected
):
    micro_result = microsimulation().calculate(variable, period, use_weights=False)

    np.testing.assert_array_equal(micro_result, expected)
    assert describe(micro_result) == describe(simulation().calculate(variable, period))


def test_top_level_enum_divide_over_a_year_returns_weighted_codes():
    simulation_under_test = microsimulation()

    with weight_reads() as variables:
        default = simulation_under_test.calculate_divide("yearly_tenure", DATASET_YEAR)
    explicit = simulation_under_test.calculate_divide(
        "yearly_tenure", DATASET_YEAR, decode_enums=False
    )
    decoded = simulation_under_test.calculate_divide(
        "yearly_tenure", DATASET_YEAR, decode_enums=True
    )

    assert_weighted(default, TENURE_CODES)
    assert_weighted(explicit, TENURE_CODES)
    assert_weighted(decoded, TENURE_NAMES)
    assert variables == ["yearly_tenure"]


def test_top_level_enum_add_over_a_year_decodes_when_asked():
    simulation_under_test = microsimulation()

    assert_weighted(
        simulation_under_test.calculate_add("yearly_tenure", DATASET_YEAR),
        TENURE_CODES,
    )
    assert_weighted(
        simulation_under_test.calculate_add(
            "yearly_tenure", DATASET_YEAR, decode_enums=True
        ),
        TENURE_NAMES,
    )


def test_top_level_enum_divide_over_a_month_is_not_decoded():
    # A twelfth of a code names no member. A Simulation ignores
    # ``decode_enums`` here, and so does a Microsimulation.
    result = microsimulation().calculate_divide(
        "yearly_tenure", "2022-06", decode_enums=True, use_weights=False
    )

    assert type(result) is EnumArray
    np.testing.assert_array_equal(np.asarray(result), [0.0, 1 / 12, 0.0])


def test_unweighted_top_level_add_returns_a_plain_array():
    result = microsimulation().calculate_add(
        "monthly_pay", DATASET_YEAR, use_weights=False
    )

    assert_array(result, PAY)
    assert result.sum() == 624.0
