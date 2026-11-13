"""Properties of base-mode uprating, and of chained mode beside it.

Base mode (``metadata.uprating.base``) gives each index period ``t`` after a
parameter's last explicit value

    base + round(base * (index[t] / index[base] - 1)).

Over the domain generated here:

1. The increase over the base, ``value - base``, is a whole number of rounding
   intervals, up to the float error in printing the result.
2. Each value is the exact one: the increase computed in rational arithmetic
   on the decimals the base, the index values and the interval print as,
   rounded down, up, or to the nearest multiple with an exact half going up.
   Every recorded decimal is exact, including a computed float's printed
   value. The test computes this itself and imports nothing from the
   implementation.
3. With a positive base, a higher index at the projected period never gives
   a lower value.
4. Without rounding, base mode and chained mode agree to 1e-12 when the
   explicit values after the base follow the index exactly, and to the last
   bit when the base is the last explicit value.
5. Explicit values win: base mode keeps every explicit value and adds values
   only after the last of them, at the index's periods.
6. The values base mode adds depend on the explicit values only through the
   base amount and the last explicit instant.
7. Every chained-mode parameter gets exactly the values, or the error, that
   the implementation before base mode gives it
   (``tests/fixtures/chained_uprating_reference.py``, a frozen copy of the
   module at 7c84d941b), including at rounding boundaries, where a change to
   the arithmetic before rounding shows. Three such boundaries are also
   pinned as literals.
8. ``base.floor: true`` changes nothing where the index is at or above its
   base-period level, to the last bit, and gives exactly the base where the
   index is below it, so a positive base never projects below itself.

The domain: the index and the parameter have values on January 1 of years
from 1995 to 2040. Index levels run from 1 to about 1,000 with yearly changes
from -10% to +15%, or are tenths from 100.0 to 3,000.0 drawn independently
each year, so the ratios include exact decimals such as 140 / 100. Amounts are
cents from $0.01 to $1m, rounding intervals from $0.01 to $1,000.
"""

from __future__ import annotations

import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

import copy  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from decimal import Inexact, localcontext  # noqa: E402
from fractions import Fraction  # noqa: E402
from typing import Optional  # noqa: E402

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.parameters import ParameterNode, uprate_parameters  # noqa: E402
from tests.fixtures.chained_uprating_reference import (  # noqa: E402
    reference_uprate_parameters,
)

PROPERTY_SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

MONEY = st.integers(min_value=1, max_value=100_000_000).map(lambda cents: cents / 100)
INTERVALS = st.sampled_from([0.01, 0.25, 1, 5, 10, 25, 50, 100, 1_000])
ROUNDING_TYPES = st.sampled_from(["nearest", "upwards", "downwards"])


def year_instant(year: int) -> str:
    return f"{year}-01-01"


@st.composite
def index_paths(draw, first_year: int, last_year: int) -> dict:
    """Index values on January 1 of each year from first_year to last_year."""
    years = range(first_year, last_year + 1)
    if draw(st.booleans()):
        level = draw(st.floats(min_value=1.0, max_value=1_000.0))
        values = {}
        for year in years:
            values[year_instant(year)] = level
            level *= draw(st.floats(min_value=0.9, max_value=1.15))
        return values
    return {year_instant(year): draw(st.integers(1_000, 30_000)) / 10 for year in years}


@st.composite
def roundings(draw):
    """A ``rounding`` block in one of the forms chained mode also reads."""
    interval = draw(INTERVALS)
    form = draw(st.sampled_from(["mapping", "bare"]))
    if form == "bare":
        return float(interval)
    return {"type": draw(ROUNDING_TYPES), "interval": interval}


def rounding_rule(rounding) -> tuple[str, float]:
    if isinstance(rounding, dict):
        return rounding["type"], rounding["interval"]
    return "nearest", rounding


@dataclass
class BaseModeCase:
    base_year: int
    base_value: float
    last_explicit_year: int
    values: dict
    index: dict
    rounding: object
    state_value: bool
    floor: bool = False

    def meta(self) -> dict:
        base = {"period": self.base_year}
        if self.state_value:
            base["value"] = self.base_value
        if self.floor:
            base["floor"] = True
        meta = {"parameter": "index", "base": base}
        if self.rounding is not None:
            meta["rounding"] = self.rounding
        return meta

    def uprate(self, values: Optional[dict] = None, index: Optional[dict] = None):
        root = ParameterNode(
            data={
                "amount": {
                    "values": dict(values if values is not None else self.values),
                    "metadata": {"uprating": self.meta()},
                },
                "index": {"values": dict(index if index is not None else self.index)},
            }
        )
        return uprate_parameters(root).amount

    def projected(self, parameter) -> list:
        last = year_instant(self.last_explicit_year)
        return [
            (value.instant_str, value.value)
            for value in reversed(parameter.values_list)
            if value.instant_str > last
        ]


@st.composite
def base_mode_cases(draw, rounded: Optional[bool] = None) -> BaseModeCase:
    first_year = draw(st.integers(1995, 2010))
    last_year = draw(st.integers(first_year + 1, 2040))
    index = draw(index_paths(first_year, last_year))
    base_year = draw(st.integers(first_year, last_year - 1))
    last_explicit_year = draw(st.integers(base_year, last_year - 1))
    base_value = draw(MONEY)
    values = {year_instant(base_year): base_value}
    for year in draw(st.sets(st.integers(base_year - 5, base_year - 1), max_size=3)):
        values[year_instant(year)] = draw(MONEY)
    if last_explicit_year > base_year:
        later_years = draw(
            st.sets(st.integers(base_year + 1, last_explicit_year), max_size=4)
        )
        for year in later_years | {last_explicit_year}:
            values[year_instant(year)] = draw(MONEY)
    if rounded is None:
        rounding = draw(st.one_of(st.none(), roundings()))
    elif rounded:
        rounding = draw(roundings())
    else:
        rounding = None
    return BaseModeCase(
        base_year=base_year,
        base_value=base_value,
        last_explicit_year=last_explicit_year,
        values=values,
        index=index,
        rounding=rounding,
        state_value=draw(st.booleans()),
    )


# A 40% rise on $1,400 is exactly $560, but 559.9999999999998 in floats.
FLOAT_NOISE_CASE = BaseModeCase(
    base_year=2020,
    base_value=1_400.0,
    last_explicit_year=2020,
    values={"2020-01-01": 1_400.0},
    index={"2020-01-01": 100.0, "2021-01-01": 140.0},
    rounding={"type": "downwards", "interval": 10},
    state_value=True,
)
# $5,685,910.75 exactly, 22,743,643 quarter-dollar intervals, but 4e-9
# intervals short in floats: past the fixed tolerance, inside the relative one.
LARGE_MULTIPLE_CASE = BaseModeCase(
    base_year=2020,
    base_value=572_903.25,
    last_explicit_year=2020,
    values={"2020-01-01": 572_903.25},
    index={"2020-01-01": 186.0, "2021-01-01": 2_032.0},
    rounding={"type": "downwards", "interval": 0.25},
    state_value=False,
)
# Rev. Proc. 2026-35, section 3.15: heads of household, 2027.
HEAD_OF_HOUSEHOLD_CASE = BaseModeCase(
    base_year=2025,
    base_value=23_625.0,
    last_explicit_year=2026,
    values={"2025-01-01": 23_625.0, "2026-01-01": 24_150.0},
    index={"2025-01-01": 100.0, "2026-01-01": 102.6, "2027-01-01": 105.53},
    rounding={"type": "downwards", "interval": 50},
    state_value=True,
)


# The review's counterexamples to the first base-mode rounding (2026-11-11).
# Exact ties for nearest, which floats put a hair below the half:
# 375 * (523.8 / 202.5 - 1) is exactly $595, 59.5 intervals of $10.
TIE_CASE = BaseModeCase(
    base_year=2020,
    base_value=375.0,
    last_explicit_year=2020,
    values={"2020-01-01": 375.0},
    index={"2020-01-01": 202.5, "2021-01-01": 523.8},
    rounding={"type": "nearest", "interval": 10},
    state_value=True,
)
# 100 * 5% is $5, half an interval of $10, where Python's round would take
# the even multiple, 0.
EVEN_TIE_CASE = BaseModeCase(
    base_year=2020,
    base_value=100.0,
    last_explicit_year=2020,
    values={"2020-01-01": 100.0},
    index={"2020-01-01": 100.0, "2021-01-01": 105.0},
    rounding={"type": "nearest", "interval": 10},
    state_value=True,
)
# 85 * (111 / 102 - 1) is exactly $7.50.
SMALL_TIE_CASE = BaseModeCase(
    base_year=2020,
    base_value=85.0,
    last_explicit_year=2020,
    values={"2020-01-01": 85.0},
    index={"2020-01-01": 102.0, "2021-01-01": 111.0},
    rounding={"type": "nearest", "interval": 1},
    state_value=False,
)
# An increase 5e-10 intervals below 16, which only float error, not index
# error, can put there: it rounds down to 15 intervals.
NEAR_MISS_CASE = BaseModeCase(
    base_year=2020,
    base_value=24_567.87,
    last_explicit_year=2020,
    values={"2020-01-01": 24_567.87},
    index={"2020-01-01": 315.605, "2021-01-01": 325.882},
    rounding={"type": "downwards", "interval": 50},
    state_value=True,
)
# 0.99 and 1.01 intervals lie on opposite sides of a whole interval.
NEARLY_ONE_INTERVAL_DOWN = BaseModeCase(
    base_year=2020,
    base_value=1_000.0,
    last_explicit_year=2020,
    values={"2020-01-01": 1_000.0},
    index={"2020-01-01": 100.0, "2021-01-01": 104.95},
    rounding={"type": "downwards", "interval": 50},
    state_value=True,
)
NEARLY_ONE_INTERVAL_UP = replace(
    NEARLY_ONE_INTERVAL_DOWN,
    index={"2020-01-01": 100.0, "2021-01-01": 105.05},
    rounding={"type": "upwards", "interval": 50},
)
# A 10% rise on $100 is exactly $10, but 10.000000000000014 in floats.
FLOAT_NOISE_UP_CASE = BaseModeCase(
    base_year=2020,
    base_value=100.0,
    last_explicit_year=2020,
    values={"2020-01-01": 100.0},
    index={"2020-01-01": 100.0, "2021-01-01": 110.0},
    rounding={"type": "upwards", "interval": 10},
    state_value=True,
)


def decimal(number) -> Fraction:
    """The exact value of the decimal a number prints as."""
    if isinstance(number, float):
        return Fraction(repr(number))
    return Fraction(number)


def round_intervals(intervals: Fraction, rounding_type: str) -> int:
    if rounding_type == "downwards":
        return math.floor(intervals)
    if rounding_type == "upwards":
        return math.ceil(intervals)
    return math.floor(intervals + Fraction(1, 2))


def exact_value(case: BaseModeCase, instant_str: str) -> float:
    """The unique rounded value of the recorded decimal inputs."""
    rounding_type, interval = rounding_rule(case.rounding)
    step = decimal(interval)
    base = decimal(case.base_value)
    ratio = decimal(case.index[instant_str]) / decimal(
        case.index[year_instant(case.base_year)]
    )
    if case.floor:
        ratio = max(ratio, Fraction(1))
    exact = base * (ratio - 1) / step
    return float(base + round_intervals(exact, rounding_type) * step)


EXACT_EXAMPLES = [
    FLOAT_NOISE_CASE,
    FLOAT_NOISE_UP_CASE,
    LARGE_MULTIPLE_CASE,
    HEAD_OF_HOUSEHOLD_CASE,
    TIE_CASE,
    EVEN_TIE_CASE,
    SMALL_TIE_CASE,
    NEAR_MISS_CASE,
    NEARLY_ONE_INTERVAL_DOWN,
    NEARLY_ONE_INTERVAL_UP,
]


def with_examples(*extra_args):
    """Add each of EXACT_EXAMPLES as a Hypothesis example, then ``extra_args``."""

    def decorate(test):
        for case in reversed(EXACT_EXAMPLES):
            test = example(case, *extra_args)(test)
        return test

    return decorate


@PROPERTY_SETTINGS
@given(base_mode_cases(rounded=True))
@with_examples()
def test_the_increase_is_a_whole_number_of_intervals(case):
    _, interval = rounding_rule(case.rounding)
    projected = case.projected(case.uprate())
    assert projected
    for _, value in projected:
        intervals = (value - case.base_value) / interval
        # Printing base + k * interval as a float, and the float subtraction
        # and division here, each add error of order epsilon of the amounts.
        float_error = (
            4 * sys.float_info.epsilon * (abs(value) + abs(case.base_value)) / interval
        )
        assert abs(intervals - round(intervals)) <= 1e-12 + float_error


@PROPERTY_SETTINGS
@given(base_mode_cases(rounded=True), st.booleans())
@with_examples(False)
def test_each_value_is_the_exact_rounded_increase(case, floor):
    case = replace(case, floor=floor) if floor else case
    for instant_str, value in case.projected(case.uprate()):
        assert value == exact_value(case, instant_str), instant_str


@pytest.mark.parametrize(
    "case, expected",
    [
        (FLOAT_NOISE_CASE, 1_960.0),
        (FLOAT_NOISE_UP_CASE, 110.0),
        (LARGE_MULTIPLE_CASE, 572_903.25 + 22_743_643 * 0.25),
        (HEAD_OF_HOUSEHOLD_CASE, 24_925.0),
        (TIE_CASE, 975.0),
        (EVEN_TIE_CASE, 110.0),
        (SMALL_TIE_CASE, 93.0),
        (NEAR_MISS_CASE, 25_317.87),
        (NEARLY_ONE_INTERVAL_DOWN, 1_000.0),
        (NEARLY_ONE_INTERVAL_UP, 1_100.0),
    ],
)
def test_review_counterexamples_give_the_exact_value(case, expected):
    (_, value), *_ = reversed(case.projected(case.uprate()))
    assert value == expected


def one_step_case(base, index_at_base, index_at_t, rounding_type, interval):
    return BaseModeCase(
        base_year=2020,
        base_value=base,
        last_explicit_year=2020,
        values={"2020-01-01": base},
        index={"2020-01-01": index_at_base, "2021-01-01": index_at_t},
        rounding={"type": rounding_type, "interval": interval},
        state_value=True,
    )


@pytest.mark.parametrize(
    "case, expected",
    [
        # Exact increases a few billionths of an interval from a multiple
        # must remain on the recorded side of that multiple.
        (one_step_case(1_819_252.81, 370.811, 395.69, "downwards", 5), 1_941_307.81),
        (one_step_case(2_226_662.91, 382.408, 411.597, "downwards", 10), 2_396_612.91),
        (one_step_case(5_802_294.89, 275.964, 277.373, "upwards", 25), 5_831_944.89),
        # The result is the float nearest the exact decimal, which a binary
        # reading of the inputs misses in the last bit (20021.629999999997).
        (one_step_case(16_021.63, 181.695, 227.408, "downwards", 50), 20_021.63),
        (one_step_case(0.05, 1.0, 1.125, "nearest", 0.01), 0.06),
        # This increase lies 1e-7 intervals below 12,345,678 and still rounds
        # down, even though it is very close to a whole interval.
        (
            one_step_case(123_456_779_999.999, 100.0, 100.01, "downwards", 1),
            123_469_125_676.999,
        ),
        # A whole-number float prints with a trailing ".0", which is no
        # significant digit: 123456779999999.0 has 15. The increase lies 1e-7
        # intervals below 12,345,678 and rounds down.
        (
            one_step_case(123_456_779_999_999.0, 100.0, 100.00001, "downwards", 1),
            123_456_792_345_676.0,
        ),
        # An int is exact at any length: 1,234,567,799,999,999 has 16 digits,
        # and its increase, 1e-7 intervals below 123,456,780, rounds down.
        (
            one_step_case(1_234_567_799_999_999, 100.0, 100.00001, "downwards", 1),
            1_234_567_923_456_778.0,
        ),
        # These increases, $1e-12 (1e-14 intervals) below $100, round down
        # whether the input uses ordinary or exponent notation.
        (one_step_case(100.0, 0.5, 0.999999999999995, "downwards", 100), 100.0),
        (one_step_case(100.0, 5e-05, 9.99999999999995e-05, "downwards", 100), 100.0),
    ],
)
def test_decimal_inputs_round_exactly(case, expected):
    assert case.uprate()("2021-01-01") == expected


def test_a_large_base_does_not_erase_a_decimal_shortfall():
    # This increase lies exactly 5e-6 intervals below 2 and rounds down to 1,
    # even though the base is much larger than the shortfall.
    case = BaseModeCase(
        base_year=2020,
        base_value=10_000_000.0,
        last_explicit_year=2020,
        values={"2020-01-01": 10_000_000.0},
        index={"2020-01-01": 100.0, "2021-01-01": 100.0000001999995},
        rounding={"type": "downwards", "interval": 0.01},
        state_value=True,
    )
    assert case.uprate()("2021-01-01") == 10_000_000.01


@pytest.mark.parametrize(
    "base, index_at_base, index_at_t, rounding_type, interval, expected",
    [
        # Computed inputs use their recorded decimal, on either side of a
        # boundary or tie. Their intended precision cannot be inferred.
        (100.0, 0.1, 0.1 + 0.2, "upwards", 100, 400.0),
        (100.0, 0.4, 0.7 + 0.1, "downwards", 100, 100.0),
        (100.0, 0.1, 0.35 + 0.1, "nearest", 100, 400.0),
        (1_000.0, 100.0, 100.0 + 0.1 + 0.6, "downwards", 1, 1_006.0),
        (1_000.0, 100.0, 100.0 + 0.2 + 0.4, "upwards", 1, 1_007.0),
        (1_000.0, 100.0, 100.699999999 + 1e-14, "downwards", 1, 1_006.0),
        (100.0, 0.1 + 0.2, 0.6, "downwards", 100, 100.0),
        (0.1 + 0.7, 1.0, 2.0, "downwards", 0.1, 1.5),
        (300.0, 100.0, 103.0, "downwards", 0.1 + 0.2, 308.7),
        (1_000.0, 100.0, 100.69999999999713, "downwards", 1, 1_006.0),
        (1_000.0, 100.0, 100.69999999999928, "downwards", 1, 1_006.0),
        (1_000.0, 100.0, 299.99999999999784, "downwards", 1, 2_999.0),
    ],
)
def test_computed_inputs_round_their_recorded_decimal(
    base, index_at_base, index_at_t, rounding_type, interval, expected
):
    case = one_step_case(base, index_at_base, index_at_t, rounding_type, interval)
    assert case.uprate()("2021-01-01") == expected


def test_exact_rounding_does_not_depend_on_the_decimal_context():
    case = one_step_case(1_000.0, 100.0, 100.0 + 0.1 + 0.6, "downwards", 1)
    with localcontext() as context:
        context.prec = 10
        context.traps[Inexact] = True
        assert case.uprate()("2021-01-01") == 1_006.0


@pytest.mark.parametrize(
    "rounding_type, high, expected",
    [
        ("downwards", 100.699999999999, 50_300.0),
        ("nearest", 100.749999999999, 50_350.0),
        ("upwards", 100.60000000000102, 50_350.0),
    ],
)
def test_adjacent_indices_at_decimal_precision_boundary_are_monotone(
    rounding_type, high, expected
):
    low = math.nextafter(high, 0)
    # The old digit-count heuristic reversed these adjacent floats in all
    # three rounding modes, changing the amount by a whole $50 interval.
    values = [
        one_step_case(50_000.0, 100.0, index, rounding_type, 50).uprate()("2021-01-01")
        for index in (low, high)
    ]
    assert values == [expected, expected]


@PROPERTY_SETTINGS
@given(
    base_value=MONEY,
    index_at_base=st.floats(min_value=1.0, max_value=1_000.0),
    index_at_t=st.floats(min_value=1.0, max_value=3_000.0),
    rise=st.floats(min_value=0.0, max_value=1_000.0),
    rounding=st.one_of(st.none(), roundings()),
)
@example(
    base_value=50_000.0,
    index_at_base=100.0,
    index_at_t=100.69999999999898,
    rise=1.4210854715202004e-14,
    rounding={"type": "downwards", "interval": 50},
)
@example(
    base_value=50_000.0,
    index_at_base=100.0,
    index_at_t=100.74999999999899,
    rise=1.4210854715202004e-14,
    rounding={"type": "nearest", "interval": 50},
)
@example(
    base_value=50_000.0,
    index_at_base=100.0,
    index_at_t=100.600000000001,
    rise=1.4210854715202004e-14,
    rounding={"type": "upwards", "interval": 50},
)
def test_a_higher_index_never_gives_a_lower_value(
    base_value, index_at_base, index_at_t, rise, rounding
):
    def value_at(index_value: float) -> float:
        case = BaseModeCase(
            base_year=2020,
            base_value=base_value,
            last_explicit_year=2020,
            values={"2020-01-01": base_value},
            index={"2020-01-01": index_at_base, "2021-01-01": index_value},
            rounding=rounding,
            state_value=False,
        )
        return case.uprate()("2021-01-01")

    assert value_at(index_at_t + rise) >= value_at(index_at_t)


@PROPERTY_SETTINGS
@given(base_mode_cases(rounded=False), st.booleans())
def test_without_rounding_base_and_chained_mode_agree(case, explicit_path):
    base_index = case.index[year_instant(case.base_year)]
    values = {year_instant(case.base_year): case.base_value}
    if explicit_path:
        # Explicit values after the base that follow the index exactly.
        for instant_str, index_value in case.index.items():
            if (
                year_instant(case.base_year)
                < instant_str
                <= year_instant(case.last_explicit_year)
            ):
                values[instant_str] = case.base_value * (index_value / base_index)
        last_explicit_year = case.last_explicit_year
    else:
        last_explicit_year = case.base_year
    root = ParameterNode(
        data={
            "base_mode": {
                "values": dict(values),
                "metadata": {
                    "uprating": {
                        "parameter": "index",
                        "base": {"period": case.base_year},
                    }
                },
            },
            "chained_mode": {
                "values": dict(values),
                "metadata": {"uprating": {"parameter": "index"}},
            },
            "index": {"values": dict(case.index)},
        }
    )
    uprated = uprate_parameters(root)

    def pairs(parameter):
        return [(value.instant_str, value.value) for value in parameter.values_list]

    base_pairs = pairs(uprated.base_mode)
    chained_pairs = pairs(uprated.chained_mode)
    assert [instant for instant, _ in base_pairs] == [
        instant for instant, _ in chained_pairs
    ]
    last = year_instant(last_explicit_year)
    for (instant_str, base_value), (_, chained_value) in zip(base_pairs, chained_pairs):
        if instant_str > last and last_explicit_year == case.base_year:
            assert base_value == chained_value
        else:
            assert base_value == pytest.approx(chained_value, rel=1e-12)


# The index falls below its base level in 2021 and 2023 and rises above it
# in 2022: the floor binds in two years and not in the third.
FALLING_INDEX_CASE = BaseModeCase(
    base_year=2020,
    base_value=1_000.0,
    last_explicit_year=2020,
    values={"2020-01-01": 1_000.0},
    index={
        "2020-01-01": 100.0,
        "2021-01-01": 97.0,
        "2022-01-01": 106.0,
        "2023-01-01": 99.9,
    },
    rounding={"type": "downwards", "interval": 50},
    state_value=False,
)


@PROPERTY_SETTINGS
@given(base_mode_cases())
@example(HEAD_OF_HOUSEHOLD_CASE)
@example(FALLING_INDEX_CASE)
@example(replace(FALLING_INDEX_CASE, rounding=None))
def test_floor_holds_the_index_ratio_at_one(case):
    unfloored = case.projected(case.uprate())
    floored_case = replace(case, floor=True)
    floored = floored_case.projected(floored_case.uprate())
    assert [instant_str for instant_str, _ in floored] == [
        instant_str for instant_str, _ in unfloored
    ]
    index_at_base = case.index[year_instant(case.base_year)]
    for (instant_str, value), (_, unfloored_value) in zip(floored, unfloored):
        if case.index[instant_str] >= index_at_base:
            assert value == unfloored_value
        else:
            assert value == case.base_value
        assert value >= case.base_value


@PROPERTY_SETTINGS
@given(base_mode_cases())
@example(HEAD_OF_HOUSEHOLD_CASE)
def test_explicit_values_win(case):
    parameter = case.uprate()
    last = year_instant(case.last_explicit_year)
    kept = {
        value.instant_str: value.value
        for value in parameter.values_list
        if value.instant_str <= last
    }
    assert kept == case.values
    projected_instants = [instant_str for instant_str, _ in case.projected(parameter)]
    assert projected_instants == sorted(
        instant_str for instant_str in case.index if instant_str > last
    )
    # One value per instant, newest first.
    instants = [value.instant_str for value in parameter.values_list]
    assert instants == sorted(set(instants), reverse=True)


@PROPERTY_SETTINGS
@given(base_mode_cases(), st.data())
def test_projected_values_depend_only_on_the_base_and_the_last_instant(case, data):
    base_instant = year_instant(case.base_year)
    other_values = {
        instant_str: (
            value
            if instant_str == base_instant
            else data.draw(MONEY, label=instant_str)
        )
        for instant_str, value in case.values.items()
    }
    assert case.projected(case.uprate(values=other_values)) == case.projected(
        case.uprate()
    )


@dataclass
class ChainedModeCase:
    data: dict


@st.composite
def chained_mode_cases(draw) -> ChainedModeCase:
    first_year = draw(st.integers(1995, 2010))
    last_year = draw(st.integers(first_year + 1, 2040))
    index = draw(index_paths(first_year, last_year))
    years = draw(
        st.sets(st.integers(first_year - 3, last_year + 2), min_size=1, max_size=6)
    )
    values = {year_instant(year): draw(MONEY) for year in years}
    form = draw(
        st.sampled_from(
            ["string", "mapping", "rounded", "bare", "int", "start", "self"]
        )
    )
    if form == "string":
        meta = "index"
    elif form == "mapping":
        meta = {"parameter": "index"}
    elif form == "rounded":
        meta = {"parameter": "index", "rounding": draw(roundings())}
    elif form == "bare":
        meta = {"parameter": "index", "rounding": float(draw(INTERVALS))}
    elif form == "int":
        # Chained mode fails on an int interval; both implementations must.
        meta = {"parameter": "index", "rounding": 50}
    elif form == "start":
        meta = {
            "parameter": "index",
            "start_instant": year_instant(draw(st.integers(first_year, last_year))),
        }
        if draw(st.booleans()):
            meta["rounding"] = draw(roundings())
    else:
        meta = {"parameter": "self"}
        if draw(st.booleans()):
            meta["number"] = draw(st.integers(1, 5))
    data = {
        "amount": {"values": values, "metadata": {"uprating": meta}},
        "index": {"values": index},
    }
    if draw(st.booleans()):
        # A parameter uprated by the first, so dependency order matters.
        data["derived"] = {
            "values": {year_instant(min(years)): draw(MONEY)},
            "metadata": {"uprating": "amount"},
        }
    return ChainedModeCase(data=data)


def chained_case(value: float, index: tuple, rounding: dict) -> ChainedModeCase:
    return ChainedModeCase(
        data={
            "amount": {
                "values": {"2020-01-01": value},
                "metadata": {"uprating": {"parameter": "index", "rounding": rounding}},
            },
            "index": {"values": {"2020-01-01": index[0], "2021-01-01": index[1]}},
        }
    )


# Rounding boundaries, where random cases rarely land and where any change to
# chained mode's arithmetic before rounding shows.
# 1,400 * (140 / 100) is 1,959.9999999999998, which rounds down to 1,950.
CHAINED_JUST_BELOW_A_MULTIPLE = chained_case(
    1_400.0, (100.0, 140.0), {"type": "downwards", "interval": 10}
)
# 100 * (110 / 100) is 110.00000000000001, which rounds up to 120.
CHAINED_JUST_ABOVE_A_MULTIPLE = chained_case(
    100.0, (100.0, 110.0), {"type": "upwards", "interval": 10}
)
# 125 at an unchanged index is a tie between 100 and 150; nearest takes the
# even multiple, 100.
CHAINED_EXACT_TIE = chained_case(
    125.0, (100.0, 100.0), {"type": "nearest", "interval": 50}
)
# 100 * (117 / 100) is 117, which nearest takes up to 120.
CHAINED_NEAREST_UP = chained_case(
    100.0, (100.0, 117.0), {"type": "nearest", "interval": 10}
)


def uprating_outcome(data: dict, uprate) -> object:
    """The values each parameter ends up with, or the error uprating raises."""
    try:
        root = uprate(ParameterNode(data=copy.deepcopy(data)))
    except Exception as error:  # noqa: BLE001 - the error is the outcome
        return ("error", type(error), str(error))
    return {
        name: [
            (value.instant_str, type(value.value), repr(value.value))
            for value in getattr(root, name).values_list
        ]
        for name in data
    }


@PROPERTY_SETTINGS
@given(chained_mode_cases())
@example(CHAINED_JUST_BELOW_A_MULTIPLE)
@example(CHAINED_JUST_ABOVE_A_MULTIPLE)
@example(CHAINED_EXACT_TIE)
@example(CHAINED_NEAREST_UP)
def test_chained_mode_matches_the_implementation_before_base_mode(case):
    assert uprating_outcome(case.data, uprate_parameters) == uprating_outcome(
        case.data, reference_uprate_parameters
    )


@pytest.mark.parametrize(
    "case, expected",
    [
        (CHAINED_JUST_BELOW_A_MULTIPLE, 1_950.0),
        (CHAINED_JUST_ABOVE_A_MULTIPLE, 120.0),
        (CHAINED_EXACT_TIE, 100.0),
        (CHAINED_NEAREST_UP, 120.0),
    ],
)
def test_chained_mode_keeps_its_float_rounding(case, expected):
    """Chained mode rounds the float product, so float error and half-to-even
    stay as they were; only base mode rounds exact decimals."""
    root = uprate_parameters(ParameterNode(data=copy.deepcopy(case.data)))
    assert root.amount("2021-01-01") == expected


CADENCE_UPRATER = {
    "2015-10-01": 2,
    "2015-12-01": 1,
    "2016-10-01": 4,
    "2016-12-01": 1,
    "2017-10-01": 6,
    "2017-12-01": 1,
    "2018-10-01": 3,
    "2019-10-01": 16,
    "2020-10-01": 32,
}


@pytest.mark.parametrize(
    "cadence",
    [
        {"enactment": "2002-04-01", "start": "2000-10-01", "end": "2001-10-01"},
        {"enactment": "2003-04-01", "start": "2000-10-01", "end": "2002-10-01"},
        {"enactment": "2003-04-01", "start": "2000-10-01", "end": "2001-10-01"},
        {
            "enactment": "2002-04-01",
            "start": "2000-10-01",
            "end": "2001-10-01",
            "effective": "2020-04-01",
        },
        {
            "enactment": "2002-04-01",
            "start": "2000-10-01",
            "end": "2001-10-01",
            "interval": "month",
        },
        {"enactment": "2002-04-01", "start": "2000-10-01"},
    ],
)
@pytest.mark.parametrize("rounding", [None, {"type": "downwards", "interval": 1}])
def test_cadence_uprating_matches_the_implementation_before_base_mode(
    cadence, rounding
):
    meta = {"parameter": "uprater", "at_defined_interval": cadence}
    if rounding is not None:
        meta["rounding"] = rounding
    data = {
        "amount": {
            "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
            "metadata": {"uprating": meta},
        },
        "uprater": {"values": CADENCE_UPRATER},
    }
    assert uprating_outcome(data, uprate_parameters) == uprating_outcome(
        data, reference_uprate_parameters
    )
