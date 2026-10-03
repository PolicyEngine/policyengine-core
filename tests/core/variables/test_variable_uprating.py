"""Uprating a variable from its latest known earlier period.

A variable with ``uprating`` and no value for the requested period takes its
latest known earlier value and multiplies it by the ratio of the uprating
index at the two period starts. Two bugs lived in that step:

* The index can have no value at the earlier instant (a value supplied for a
  year before the index starts). The ratio then divided by ``None`` and
  raised ``TypeError``. The index is now held flat where it has no value.
* The earlier period was chosen by indexing the unfiltered list of known
  periods with a position in the filtered list, so a later period stored
  first was used instead of the latest earlier one.
"""

import itertools
import math

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.model_api import MONTH, YEAR, Variable
from policyengine_core.parameters import ParameterNode
from policyengine_core.periods import Instant, Period
from policyengine_core.simulations import SimulationBuilder

INDEX_START = 2015
# The index grows 10% a year from INDEX_START. Integer powers keep the
# expected ratios exact.
INDEX = {
    f"{year}-01-01": 100 * 1.1 ** (year - INDEX_START) for year in range(2015, 2021)
}
VALUE = 5_000.0


def build_system(index_values: dict) -> CountryTaxBenefitSystem:
    system = CountryTaxBenefitSystem()
    system.parameters.add_child(
        "test_uprating",
        ParameterNode("test_uprating", data={"index": {"values": index_values}}),
    )

    class uprated_income(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Uprated yearly income"
        uprating = "test_uprating.index"

    class uprated_monthly_income(Variable):
        value_type = float
        entity = Person
        definition_period = MONTH
        label = "Uprated monthly income"
        uprating = "test_uprating.index"

    system.add_variable(uprated_income)
    system.add_variable(uprated_monthly_income)
    return system


@pytest.fixture(scope="module")
def system():
    return build_system(INDEX)


@pytest.fixture(scope="module")
def backdated_system():
    """The same index with its first value explicitly extended back to 2000,
    the way a country package backdates parameters."""
    system = build_system(INDEX)
    index = system.parameters.test_uprating.index
    first = f"{INDEX_START}-01-01"
    days = (Instant((INDEX_START, 1, 1)).date - Instant((2000, 1, 1)).date).days
    index.update(
        period=Period(("day", Instant((2000, 1, 1)), days)),
        value=index(first),
    )
    assert index("2000-01-01") == index(first)
    return system


def simulate(system, inputs: dict, variable: str = "uprated_income"):
    return SimulationBuilder().build_from_entities(
        system, {"persons": {"p": {variable: inputs}}}
    )


def calculate(system, inputs: dict, period, variable: str = "uprated_income"):
    return float(simulate(system, inputs, variable).calculate(variable, period)[0])


def index_at(year: int) -> float:
    """The index held flat before INDEX_START, as the fix reads it."""
    return INDEX[f"{min(max(year, INDEX_START), 2020)}-01-01"]


# Regression tests.


def test_value_known_before_index_starts_is_uprated_from_index_start(system):
    # Before the fix: TypeError: unsupported operand type(s) for /: 'float'
    # and 'NoneType'.
    assert calculate(system, {2013: VALUE}, 2018) == pytest.approx(
        VALUE * INDEX["2018-01-01"] / INDEX["2015-01-01"]
    )


def test_value_carried_over_unchanged_while_index_undefined(system):
    # Before the fix: TypeError dividing None by None.
    assert calculate(system, {2012: VALUE}, 2014) == pytest.approx(VALUE)
    assert calculate(system, {2012: VALUE}, 2015) == pytest.approx(VALUE)


def test_monthly_value_known_before_index_starts(system):
    assert calculate(
        system, {"2014-12": VALUE}, "2016-01", "uprated_monthly_income"
    ) == pytest.approx(VALUE * INDEX["2016-01-01"] / INDEX["2015-01-01"])


def test_uprates_from_latest_earlier_period_when_later_period_stored_first(
    system,
):
    # Before the fix the 2020 value (stored first) was used: 999.
    inputs = {2020: 999.0, 2015: VALUE}
    assert calculate(system, inputs, 2018) == pytest.approx(
        VALUE * INDEX["2018-01-01"] / INDEX["2015-01-01"]
    )


def test_cached_default_before_input_does_not_pull_later_input_back(system):
    # A formula reading an earlier year (a lagged-income lookup, say) caches
    # the default there. Before the fix a later request between that year
    # and the input year then deflated the later input instead of using the
    # latest earlier period, so the result depended on evaluation order.
    direct = calculate(system, {2018: VALUE}, 2017)
    lagged_first = simulate(system, {2018: VALUE})
    lagged_first.calculate("uprated_income", 2016)
    assert direct == 0
    assert float(lagged_first.calculate("uprated_income", 2017)[0]) == direct


def test_index_gap_from_explicit_null_is_held_flat():
    # ``update`` on a period before an index starts leaves an explicit null
    # between the new value and the old first value.
    system = build_system(INDEX)
    index = system.parameters.test_uprating.index
    index.update(period="2010", value=50)
    assert index("2012-01-01") is None
    # 2012 falls in the gap: the index is held at its 2010 value there.
    assert calculate(system, {2012: VALUE}, 2016) == pytest.approx(
        VALUE * INDEX["2016-01-01"] / 50
    )
    assert calculate(system, {2011: VALUE}, 2013) == pytest.approx(VALUE)


def test_index_with_no_values_carries_value_over():
    system = build_system({"2015-01-01": None})
    assert calculate(system, {2013: VALUE}, 2018) == pytest.approx(VALUE)


def test_index_value_helper():
    # Imported here so the rest of the module also runs against code without
    # the helper.
    from policyengine_core.simulations.simulation import _uprating_index_value

    system = build_system(INDEX)
    index = system.parameters.test_uprating.index
    # A null from 2030 with no end date: the index is undefined from then on.
    index.update(start="2030-01-01", value=None)
    assert _uprating_index_value(index, Instant((2010, 6, 1))) == INDEX["2015-01-01"]
    assert _uprating_index_value(index, Instant((2017, 1, 1))) == INDEX["2017-01-01"]
    # After an explicit null the last value before it holds.
    assert index("2030-06-01") is None
    assert _uprating_index_value(index, Instant((2030, 6, 1))) == INDEX["2020-01-01"]
    assert _uprating_index_value(index, Instant((2040, 1, 1))) == INDEX["2020-01-01"]
    empty = build_system({"2015-01-01": None}).parameters.test_uprating.index
    assert _uprating_index_value(empty, Instant((2016, 1, 1))) is None


# Invariants, checked exhaustively over a grid of known and requested years
# (2008-2022), which spans the index start and its last value.

YEARS = range(2008, 2023)
PAIRS = [(known, requested) for known, requested in itertools.combinations(YEARS, 2)]


@pytest.mark.parametrize("known,requested", PAIRS)
def test_uprated_value_equals_value_times_flat_index_ratio(system, known, requested):
    """For every known year before every requested year, the result is the
    known value times the ratio of the index held flat where undefined. Where
    the index is defined at both years this is the ratio uprating always
    used; where it is undefined at both the value carries over unchanged."""
    result = calculate(system, {known: VALUE}, requested)
    expected = VALUE * index_at(requested) / index_at(known)
    assert math.isfinite(result)
    assert result == pytest.approx(expected, rel=1e-6)
    if requested <= INDEX_START:
        assert result == pytest.approx(VALUE, rel=1e-6)


@pytest.mark.parametrize("known,requested", PAIRS)
def test_matches_explicitly_backdated_index(system, backdated_system, known, requested):
    """Differential: holding the index flat before it starts gives the same
    result as a country package explicitly extending the index's first value
    backward and uprating through the ordinary defined-index path."""
    assert calculate(system, {known: VALUE}, requested) == pytest.approx(
        calculate(backdated_system, {known: VALUE}, requested), rel=1e-6
    )


@pytest.mark.parametrize("known", range(2008, 2016))
def test_result_does_not_depend_on_intermediate_years_computed(system, known):
    """Path independence: computing every year in turn first gives exactly
    the value uprated straight from the known year. Each year uprates from
    the input, not from the year before (see test_uprating_order.py); chained
    from the year before, float32 rounding compounded."""
    stepwise = simulate(system, {known: VALUE})
    for year in range(known + 1, 2022):
        stepwise.calculate("uprated_income", year)
    stepwise_result = float(stepwise.calculate("uprated_income", 2022)[0])
    assert stepwise_result == calculate(system, {known: VALUE}, 2022)


@pytest.mark.parametrize(
    "stored_years", list(itertools.permutations([2015, 2017, 2019]))
)
def test_latest_earlier_period_wins_regardless_of_storage_order(system, stored_years):
    """Order independence: for any insertion order of the known periods, each
    requested year uprates from the latest known year before it. All years
    are inside the index's range, so only the choice of base period is
    tested here."""
    inputs = {year: 1_000.0 * (i + 1) for i, year in enumerate(stored_years)}
    for requested in (2016, 2018, 2020):
        latest = max(year for year in inputs if year < requested)
        assert calculate(system, inputs, requested) == pytest.approx(
            inputs[latest] * index_at(requested) / index_at(latest), rel=1e-6
        )


def test_value_known_after_requested_period_is_not_used(system):
    # Uprating only runs forward; a later value alone gives the default.
    assert calculate(system, {2018: VALUE}, 2016) == 0
