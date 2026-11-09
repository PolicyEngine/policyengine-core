"""Properties of base-mode uprating, and of chained mode beside it.

Base mode (``metadata.uprating.base``) gives each index period ``t`` after a
parameter's last explicit value

    base + round(base * (index[t] / index[base] - 1)).

Over the domain generated here:

1. The increase over the base, ``value - base``, is a whole number of rounding
   intervals.
2. Rounding down never puts the increase above the exact increase (the
   rational number the float inputs define) and leaves it less than one
   interval below it. Rounding up mirrors that, and rounding to the nearest
   multiple leaves it within half an interval. An exact increase that is
   already a whole number of intervals stays exactly that, under every
   rule. "Never above" allows the tolerance base mode gives an increase near
   a multiple (1e-9 intervals, or about 3.6e-15 of the amounts if more), plus
   float error of order 1e-14 of the amounts.
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
   (``tests/fixtures/chained_uprating_reference.py``), including at rounding
   boundaries, where a change to the arithmetic before rounding shows.

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
from dataclasses import dataclass  # noqa: E402
from fractions import Fraction  # noqa: E402
from typing import Optional  # noqa: E402

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.parameters import ParameterNode, uprate_parameters  # noqa: E402
from policyengine_core.parameters.operations.uprate_parameters import (  # noqa: E402
    INCREASE_ROUNDING_RELATIVE_TOLERANCE,
    INCREASE_ROUNDING_TOLERANCE,
)
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

    def meta(self) -> dict:
        base = {"period": self.base_year}
        if self.state_value:
            base["value"] = self.base_value
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


def tolerance(case: BaseModeCase, ratio: float, value: float, interval: float):
    """Allowed error in an increase, in currency units, as an exact Fraction.

    Base mode's snap can move an increase by its tolerance: the larger of
    INCREASE_ROUNDING_TOLERANCE intervals and INCREASE_ROUNDING_RELATIVE_TOLERANCE
    times the uprated amount plus the base, which ``amounts`` bounds twice
    over. Float arithmetic adds error of order 1e-16 of the amounts, which
    1e-14 covers.
    """
    amounts = abs(case.base_value) * max(ratio, 1.0) + abs(value)
    snap = max(
        INCREASE_ROUNDING_TOLERANCE * interval,
        INCREASE_ROUNDING_RELATIVE_TOLERANCE * 2 * amounts,
    )
    return Fraction(snap + 1e-14 * amounts)


@PROPERTY_SETTINGS
@given(base_mode_cases(rounded=True))
@example(FLOAT_NOISE_CASE)
@example(LARGE_MULTIPLE_CASE)
@example(HEAD_OF_HOUSEHOLD_CASE)
def test_the_increase_is_a_whole_number_of_intervals(case):
    _, interval = rounding_rule(case.rounding)
    projected = case.projected(case.uprate())
    assert projected
    for _, value in projected:
        intervals = (value - case.base_value) / interval
        assert abs(intervals - round(intervals)) <= 1e-6


@PROPERTY_SETTINGS
@given(base_mode_cases(rounded=True))
@example(FLOAT_NOISE_CASE)
@example(LARGE_MULTIPLE_CASE)
@example(HEAD_OF_HOUSEHOLD_CASE)
def test_the_rounded_increase_brackets_the_exact_increase(case):
    rounding_type, interval = rounding_rule(case.rounding)
    base = Fraction(case.base_value)
    index_at_base = Fraction(case.index[year_instant(case.base_year)])
    for instant_str, value in case.projected(case.uprate()):
        index_at_t = Fraction(case.index[instant_str])
        exact = base * (index_at_t / index_at_base - 1)
        rounded = Fraction(value) - base
        slack = tolerance(case, float(index_at_t / index_at_base), value, interval)
        if (exact / Fraction(interval)).denominator == 1:
            # An exact whole number of intervals survives float error intact.
            assert abs(rounded - exact) <= slack
        if rounding_type == "downwards":
            assert rounded <= exact + slack
            assert exact - rounded < interval + slack
        elif rounding_type == "upwards":
            assert rounded >= exact - slack
            assert rounded - exact < interval + slack
        else:
            assert abs(rounded - exact) <= Fraction(interval) / 2 + slack


@PROPERTY_SETTINGS
@given(
    base_value=MONEY,
    index_at_base=st.floats(min_value=1.0, max_value=1_000.0),
    index_at_t=st.floats(min_value=1.0, max_value=3_000.0),
    rise=st.floats(min_value=0.0, max_value=1_000.0),
    rounding=st.one_of(st.none(), roundings()),
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
def test_chained_mode_matches_the_implementation_before_base_mode(case):
    assert uprating_outcome(case.data, uprate_parameters) == uprating_outcome(
        case.data, reference_uprate_parameters
    )


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
