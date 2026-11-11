"""Base-mode uprating (``metadata.uprating.base``): examples and errors.

Base mode projects each period after a parameter's last explicit value as

    base + round(base * (index[t] / index[base] - 1))

rounding the increase over the base, where chained mode rounds the uprated
amount. The properties are in ``test_base_uprating_property.py``.

The index values below are made up. The statutory examples use values inside
the range that Rev. Proc. 2026-35's 2027 amounts imply; they are not BLS data.
"""

import pytest

from policyengine_core.parameters import ParameterNode, uprate_parameters
from policyengine_core.periods import instant


def uprated_root(data: dict) -> ParameterNode:
    return uprate_parameters(ParameterNode(data=data))


def base_mode(base: dict, rounding=None, index: str = "index") -> dict:
    meta = {"parameter": index, "base": base}
    if rounding is not None:
        meta["rounding"] = rounding
    return {"uprating": meta}


DOWN_50 = {"type": "downwards", "interval": 50}


def test_standard_deduction_for_2027_matches_rev_proc_2026_35():
    # 26 U.S.C. 63(c)(7)(B)(ii) increases the $15,750 and $23,625 amounts by
    # the cost-of-living adjustment with 2024 as the base year, rounding each
    # increase down to a multiple of $50. Rev. Proc. 2026-35, section 3.15,
    # prints $16,600, $24,925 and, for joint returns (twice the single
    # amount, 63(c)(2)(A)), $33,200. Here the index at 2025 stands for the
    # base year, and the ratio 1.0553 lies in the range those amounts imply.
    index = {"2025-01-01": 100.0, "2026-01-01": 102.6, "2027-01-01": 105.53}
    root = uprated_root(
        {
            "single": {
                "values": {"2025-01-01": 15_750, "2026-01-01": 16_100},
                "metadata": base_mode({"period": 2025, "value": 15_750}, DOWN_50),
            },
            "head_of_household": {
                "values": {"2025-01-01": 23_625, "2026-01-01": 24_150},
                "metadata": base_mode({"period": 2025, "value": 23_625}, DOWN_50),
            },
            # Twice the single amount: twice an increase rounded down to $50
            # is the doubled increase rounded down to $100.
            "joint": {
                "values": {"2025-01-01": 31_500, "2026-01-01": 32_200},
                "metadata": base_mode(
                    {"period": 2025, "value": 31_500},
                    {"type": "downwards", "interval": 100},
                ),
            },
            "head_of_household_chained": {
                "values": {"2025-01-01": 23_625, "2026-01-01": 24_150},
                "metadata": {"uprating": {"parameter": "index", "rounding": DOWN_50}},
            },
            "index": {"values": index},
        }
    )

    assert root.single("2027-01-01") == 16_600
    assert root.head_of_household("2027-01-01") == 24_925
    assert root.joint("2027-01-01") == 33_200
    # Chained mode rounds the whole amount, so it lands on a multiple of $50
    # and cannot print $24,925.
    chained = root.head_of_household_chained("2027-01-01")
    assert chained % 50 == 0
    assert chained != 24_925


def test_long_term_care_premium_limits_for_2027_match_rev_proc_2026_35():
    # 26 U.S.C. 213(d)(10)(B)(i) increases the 1997 amounts in (A) by the
    # medical care cost adjustment and rounds each increase to the nearest
    # $10. Rev. Proc. 2026-35, section 3.29, prints $500, $945, $1,890,
    # $5,030 and $6,290. The parameter's history starts after 1997, so each
    # bracket states its base amount.
    bases = [200, 375, 750, 2_000, 2_500]
    published_2026 = [500, 930, 1_860, 4_960, 6_200]
    thresholds = [-float("inf"), 40, 50, 60, 70]
    rounding = {"type": "nearest", "interval": 10}
    brackets = []
    for threshold, base, value_2026 in zip(thresholds, bases, published_2026):
        brackets.append(
            {
                "threshold": {"values": {"2013-01-01": threshold}},
                "amount": {
                    "values": {"2013-01-01": base * 2, "2026-01-01": value_2026},
                    "metadata": base_mode({"period": 1997, "value": base}, rounding),
                },
            }
        )
    root = uprated_root(
        {
            "limit": {"brackets": brackets},
            "band_41_to_50_chained": {
                "values": {"2026-01-01": 930},
                "metadata": {"uprating": {"parameter": "index", "rounding": rounding}},
            },
            "index": {
                "values": {
                    "1997-01-01": 100.0,
                    "2026-01-01": 248.0,
                    "2027-01-01": 251.55,
                }
            },
        }
    )

    amounts = [bracket.amount("2027-01-01") for bracket in root.limit.brackets]
    assert amounts == [500, 945, 1_890, 5_030, 6_290]
    # Rounding the whole amount puts the $375 band on a multiple of $10.
    assert root.band_41_to_50_chained("2027-01-01") == 940


def test_explicit_values_win_and_projection_starts_after_the_last():
    root = uprated_root(
        {
            "amount": {
                # 2027 departs from the formula, as a published value can.
                "values": {
                    "2024-01-01": 1_000,
                    "2025-01-01": 1_050,
                    "2027-01-01": 1_234,
                },
                "metadata": base_mode({"period": 2024}, DOWN_50),
            },
            "index": {
                "values": {
                    "2024-01-01": 100.0,
                    "2025-01-01": 110.0,
                    "2026-01-01": 120.0,
                    "2027-01-01": 130.0,
                    "2028-01-01": 140.0,
                    "2029-01-01": 150.0,
                }
            },
        }
    )

    assert root.amount("2024-01-01") == 1_000
    assert root.amount("2025-01-01") == 1_050
    # 2026 has no explicit value but precedes the last one: it keeps 2025's.
    assert root.amount("2026-01-01") == 1_050
    assert root.amount("2027-01-01") == 1_234
    # From the base, not from 2027's value: 1,000 + 400 and 1,000 + 500.
    assert root.amount("2028-01-01") == 1_400
    assert root.amount("2029-01-01") == 1_500
    assert [value.instant_str for value in root.amount.values_list] == [
        "2029-01-01",
        "2028-01-01",
        "2027-01-01",
        "2025-01-01",
        "2024-01-01",
    ]


def test_base_value_defaults_to_the_parameters_value_at_the_base_period():
    root = uprated_root(
        {
            "amount": {
                "values": {"2017-01-01": 900, "2018-01-01": 1_400},
                "metadata": base_mode(
                    {"period": 2018}, {"type": "downwards", "interval": 100}
                ),
            },
            "index": {
                "values": {
                    "2018-01-01": 100.0,
                    "2019-01-01": 103.0,
                    "2024-01-01": 122.0,
                }
            },
        }
    )

    # 1,400 * 0.03 = 42 rounds down to 0; 1,400 * 0.22 = 308 to 300.
    assert root.amount("2019-01-01") == 1_400
    assert root.amount("2024-01-01") == 1_700


def test_a_stated_base_value_must_match_the_value_at_the_base_period():
    with pytest.raises(ValueError, match="a year off"):
        uprated_root(
            {
                "amount": {
                    "values": {"2024-01-01": 21_900, "2025-01-01": 23_625},
                    # One year early: the parameter's 2024 value is 21,900.
                    "metadata": base_mode({"period": 2024, "value": 23_625}),
                },
                "index": {"values": {"2024-01-01": 100.0, "2026-01-01": 103.0}},
            }
        )


def test_a_stated_base_value_matching_the_parameter_is_accepted():
    root = uprated_root(
        {
            "amount": {
                "values": {"2025-01-01": 23_625},
                "metadata": base_mode({"period": 2025, "value": 23_625.0}, DOWN_50),
            },
            "index": {"values": {"2025-01-01": 100.0, "2026-01-01": 103.0}},
        }
    )

    # 23,625 * 0.03 = 708.75 rounds down to 700.
    assert root.amount("2026-01-01") == 24_325


@pytest.mark.parametrize(
    "rounding, expected",
    [
        ({"type": "downwards", "interval": 10}, 1_160),
        ({"type": "upwards", "interval": 10}, 1_170),
        ({"type": "nearest", "interval": 10}, 1_170),
        # A bare interval rounds to the nearest multiple, as in chained mode.
        (10, 1_170),
        (10.0, 1_170),
        ({"type": "downwards", "interval": 0.25}, 1_166.5),
    ],
)
def test_rounding_applies_to_the_increase(rounding, expected):
    # The increase is 1,000 * 0.1666 = 166.6.
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": 1_000},
                "metadata": base_mode({"period": 2020}, rounding),
            },
            "index": {"values": {"2020-01-01": 100.0, "2021-01-01": 116.66}},
        }
    )

    assert root.amount("2021-01-01") == pytest.approx(expected)


@pytest.mark.parametrize(
    "base, index, expected",
    [
        # 500 * 5% = $25, 2.5 intervals of $10: a tie, which goes up (Python's
        # round would take it to the even 2).
        (500, (100.0, 105.0), 530),
        # 100 * 5% = $5, half an interval: up to $10, not to the even 0.
        (100, (100.0, 105.0), 110),
        # 900 * 5% = $45, 4.5 intervals: up to $50, not to the even 4.
        (900, (100.0, 105.0), 950),
        # 375 * (523.8 / 202.5 - 1) = $595 exactly, but $594.9999999999999
        # in floats, a hair below the tie.
        (375, (202.5, 523.8), 975),
        # 375 * 132% = $495 exactly, but $494.9999999999999 in floats.
        (375, (100.0, 232.0), 875),
        # 85 * (111 / 102 - 1) = $7.50 exactly, but $7.499999999999986 in
        # floats, on a $1 interval.
        (85, (102.0, 111.0), 93),
    ],
)
def test_nearest_rounds_an_exact_half_up(base, index, expected):
    interval = 1 if base == 85 else 10
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": base},
                "metadata": base_mode(
                    {"period": 2020}, {"type": "nearest", "interval": interval}
                ),
            },
            "index": {"values": {"2020-01-01": index[0], "2021-01-01": index[1]}},
        }
    )

    assert root.amount("2021-01-01") == expected


def test_without_rounding_base_mode_scales_the_base_by_the_index():
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": 1_000, "2021-01-01": 1_111},
                "metadata": base_mode({"period": 2020}),
            },
            "index": {
                "values": {
                    "2020-01-01": 100.0,
                    "2021-01-01": 110.0,
                    "2022-01-01": 125.0,
                }
            },
        }
    )

    assert root.amount("2022-01-01") == pytest.approx(1_250)


def test_an_increase_on_a_multiple_is_not_rounded_down_a_full_interval():
    # A 40% rise on $1,400 is $560 exactly, but 1,400 * (140 / 100) - 1,400 is
    # 559.9999999999998 in binary floating point. Rounded down to $10 in
    # floats, it would lose $10; base mode rounds the exact decimal increase.
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": 1_400},
                "metadata": base_mode(
                    {"period": 2020}, {"type": "downwards", "interval": 10}
                ),
            },
            "index": {"values": {"2020-01-01": 100.0, "2021-01-01": 140.0}},
        }
    )

    assert root.amount("2021-01-01") == 1_960


def test_a_large_increase_on_a_multiple_is_not_rounded_down_an_interval():
    # $572,903.25 * (2,032 / 186 - 1) is $5,685,910.75 exactly, 22,743,643
    # quarter-dollar intervals, but floats put it 4e-9 intervals short.
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": 572_903.25},
                "metadata": base_mode(
                    {"period": 2020}, {"type": "downwards", "interval": 0.25}
                ),
            },
            "index": {"values": {"2020-01-01": 186.0, "2021-01-01": 2_032.0}},
        }
    )

    assert root.amount("2021-01-01") == 572_903.25 + 5_685_910.75


def test_a_scale_level_base_reads_each_brackets_value_at_the_base_period():
    root = uprated_root(
        {
            "scale": {
                "metadata": {
                    **base_mode({"period": 2025}, DOWN_50),
                    "uprate_thresholds": True,
                },
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 0}},
                        "rate": {"values": {"2025-01-01": 0.1}},
                    },
                    {
                        "threshold": {"values": {"2025-01-01": 11_925}},
                        "rate": {"values": {"2025-01-01": 0.12}},
                    },
                ],
            },
            "index": {"values": {"2025-01-01": 100.0, "2026-01-01": 102.6}},
        }
    )

    thresholds = [bracket.threshold("2026-01-01") for bracket in root.scale.brackets]
    # 11,925 * 0.026 = 310.05 rounds down to 300.
    assert thresholds == [0, 12_225]


@pytest.mark.parametrize(
    "floor, index_2026, expected",
    [
        # 11,925 * 0.026 = 310.05 rounds to 300.
        (False, 102.6, 12_225),
        (True, 102.6, 12_225),
        # 11,925 * -0.03 = -357.75 rounds to -350; the floor holds the base.
        (False, 97.0, 11_575),
        (True, 97.0, 11_925),
    ],
)
def test_an_infinite_base_stays_infinite(floor, index_2026, expected):
    # A scale's lowest threshold is often -inf, and indexing keeps it there.
    root = uprated_root(
        {
            "scale": {
                "metadata": {
                    **base_mode(
                        {"period": 2025, "floor": floor},
                        {"type": "nearest", "interval": 50},
                    ),
                    "uprate_thresholds": True,
                },
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": -float("inf")}},
                        "rate": {"values": {"2025-01-01": 0.1}},
                    },
                    {
                        "threshold": {"values": {"2025-01-01": 11_925}},
                        "rate": {"values": {"2025-01-01": 0.12}},
                    },
                ],
            },
            "index": {"values": {"2025-01-01": 100.0, "2026-01-01": index_2026}},
        }
    )

    thresholds = [bracket.threshold("2026-01-01") for bracket in root.scale.brackets]
    assert thresholds == [-float("inf"), expected]


def test_base_mode_in_a_cloned_subtree_reads_that_subtrees_index():
    root = ParameterNode(
        data={
            "gov": {
                "amount": {
                    "values": {"2025-01-01": 1_000},
                    "metadata": base_mode({"period": 2025}, DOWN_50, index="gov.index"),
                },
                "index": {"values": {"2025-01-01": 100.0, "2026-01-01": 110.0}},
            },
        }
    )
    root.add_child("baseline", root.gov.clone())
    root.gov.index.update(start="2026-01-01", value=120.0)

    uprated = uprate_parameters(root)

    assert uprated.gov.amount("2026-01-01") == 1_200
    assert uprated.baseline.amount("2026-01-01") == 1_100


def test_no_values_are_projected_before_the_base_period():
    # The base period follows the last explicit value, so the base amount is
    # the value in force there: 2020's.
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": 500},
                "metadata": base_mode({"period": 2023}, DOWN_50),
            },
            "index": {
                "values": {
                    "2021-01-01": 90.0,
                    "2022-01-01": 95.0,
                    "2023-01-01": 100.0,
                    "2024-01-01": 110.0,
                }
            },
        }
    )

    assert [value.instant_str for value in root.amount.values_list] == [
        "2024-01-01",
        "2023-01-01",
        "2020-01-01",
    ]
    assert root.amount("2022-01-01") == 500
    assert root.amount("2023-01-01") == 500
    assert root.amount("2024-01-01") == 550


def test_a_stated_base_after_the_last_explicit_value_must_match_it():
    # The parameter's value in force at 2023 is 2020's 500, so a stated
    # 1,000 contradicts it: the statutory amount belongs in the values.
    with pytest.raises(ValueError, match="value at the base period 2023-01-01 is 500"):
        uprated_root(
            {
                "amount": {
                    "values": {"2020-01-01": 500},
                    "metadata": base_mode({"period": 2023, "value": 1_000}),
                },
                "index": {"values": {"2023-01-01": 100.0, "2024-01-01": 110.0}},
            }
        )


# Core's TaxBenefitSystem applies a constructor reform before it uprates:
# it loads the parameters, clones the baseline, applies the reform, then
# uprates. A reform's change reaches the parameter through Parameter.update.


def standard_deduction_after_reform(
    reform_start: str, base: dict, reform_stop: str = None
) -> ParameterNode:
    root = ParameterNode(
        data={
            "amount": {
                "values": {"2025-01-01": 23_625, "2026-01-01": 24_150},
                "metadata": base_mode(base, DOWN_50),
            },
            "index": {
                "values": {
                    "2025-01-01": 100.0,
                    "2026-01-01": 102.6,
                    "2027-01-01": 105.53,
                }
            },
        }
    )
    # A reform entry: "from this date on" without a stop, and a range
    # ("2026-01-01.2100-12-31" in Reform.from_dict) with one.
    root.amount.update(
        start=instant(reform_start),
        stop=instant(reform_stop) if reform_stop else None,
        value=30_000,
    )
    return uprate_parameters(root)


def test_a_reform_before_uprating_changes_projections_only_through_the_base():
    def path(reform_start: str) -> list:
        amount = standard_deduction_after_reform(reform_start, {"period": 2025}).amount
        return [amount(f"{year}-01-01") for year in (2025, 2026, 2027)]

    # From 2026: the base is still 23,625, so 2027 follows the statute.
    assert path("2026-01-01") == [23_625, 30_000, 24_925]
    # From 2020: 30,000 is the value at the base period, so it is the base.
    # 30,000 * 0.026 = 780 rounds down to 750, and 30,000 * 0.0553 = 1,659
    # to 1,650.
    assert path("2020-01-01") == [30_000, 30_750, 31_650]


def test_a_bounded_reform_before_uprating_holds_its_value_over_its_range():
    # The docs' advice for a reform that should hold in later years. Its
    # range ends after the index's last period, so base mode adds nothing;
    # Parameter.update restores the prior value after the range.
    amount = standard_deduction_after_reform(
        "2026-01-01", {"period": 2025}, reform_stop="2100-12-31"
    ).amount
    years = (2025, 2026, 2027, 2100, 2101)
    assert [amount(f"{year}-01-01") for year in years] == [
        23_625,
        30_000,
        30_000,
        30_000,
        24_150,
    ]


def test_a_reform_before_uprating_that_contradicts_a_stated_base_raises():
    with pytest.raises(ValueError, match="omit `base.value`"):
        standard_deduction_after_reform("2020-01-01", {"period": 2025, "value": 23_625})


def test_an_index_below_its_base_level_gives_a_value_below_the_base():
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": 1_000},
                "metadata": base_mode({"period": 2020}, DOWN_50),
            },
            "index": {"values": {"2020-01-01": 100.0, "2021-01-01": 97.0}},
        }
    )

    # The increase, -30, rounds down to -50.
    assert root.amount("2021-01-01") == 950


def test_floor_holds_a_value_at_its_base_while_the_index_is_below_its_base_level():
    # 26 U.S.C. 1(f)(3)(A) makes the cost-of-living adjustment "the
    # percentage (if any)" by which the index exceeds its base-year level, so
    # an index below that level leaves the amount at its base.
    def amount(base: dict, rounding=None) -> dict:
        return {"values": {"2020-01-01": 1_000}, "metadata": base_mode(base, rounding)}

    root = uprated_root(
        {
            "floored": amount({"period": 2020, "floor": True}, DOWN_50),
            "floor_false": amount({"period": 2020, "floor": False}, DOWN_50),
            "unfloored": amount({"period": 2020}, DOWN_50),
            "floored_unrounded": amount({"period": 2020, "floor": True}),
            "index": {
                "values": {
                    "2020-01-01": 100.0,
                    "2021-01-01": 97.0,
                    "2022-01-01": 106.0,
                    "2023-01-01": 99.9,
                }
            },
        }
    )

    for year in ["2021", "2023"]:
        assert root.floored(f"{year}-01-01") == 1_000
        assert root.floored_unrounded(f"{year}-01-01") == 1_000
    # Without the floor, -30 and -1 round down to -50.
    assert root.unfloored("2021-01-01") == root.floor_false("2021-01-01") == 950
    assert root.unfloored("2023-01-01") == root.floor_false("2023-01-01") == 950
    # Above the base level the floor does not bind: $60 rounds down to $50.
    assert root.floored("2022-01-01") == root.unfloored("2022-01-01") == 1_050
    assert root.floored_unrounded("2022-01-01") == pytest.approx(1_060, rel=1e-12)


def test_floor_holds_the_index_ratio_at_one_not_the_increase_at_zero():
    # The statutes floor the adjustment percentage, so a negative base follows
    # an index above its base level and holds while the index is below it.
    # Flooring the increase at zero instead would give -1,000 and then -900.
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": -1_000},
                "metadata": base_mode({"period": 2020, "floor": True}, DOWN_50),
            },
            "index": {
                "values": {"2020-01-01": 100.0, "2021-01-01": 110.0, "2022-01-01": 90.0}
            },
        }
    )

    assert root.amount("2021-01-01") == -1_100
    assert root.amount("2022-01-01") == -1_000


def test_the_base_period_accepts_instant_strings():
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": 1_000},
                "metadata": base_mode({"period": "2020-01-01"}, DOWN_50),
            },
            "index": {"values": {"2020-01-01": 100.0, "2021-01-01": 110.0}},
        }
    )

    assert root.amount("2021-01-01") == 1_100


@pytest.mark.parametrize("rounding", [None, {"type": "nearest", "interval": 50}])
@pytest.mark.parametrize("base", [float("inf"), -float("inf")])
def test_an_infinite_base_stays_infinite_when_the_index_ratio_underflows(
    base, rounding
):
    # 1e-200 / 1e200 underflows to 0.0, and an infinite base times zero would
    # be NaN. The ratio of two positive index values is positive, so the base
    # stays infinite.
    root = uprated_root(
        {
            "amount": {
                "values": {"2020-01-01": base},
                "metadata": base_mode({"period": 2020}, rounding),
            },
            "index": {"values": {"2020-01-01": 1e200, "2021-01-01": 1e-200}},
        }
    )
    assert root.amount("2021-01-01") == base


def error_case(meta: dict, values=None, index_values=None) -> dict:
    return {
        "amount": {
            "values": values or {"2020-01-01": 1_000},
            "metadata": {"uprating": meta},
        },
        "index": {"values": index_values or {"2020-01-01": 100.0, "2021-01-01": 110.0}},
    }


@pytest.mark.parametrize(
    "data, message",
    [
        (error_case({"parameter": "index", "base": 2020}), "must be a mapping"),
        (error_case({"parameter": "index", "base": None}), "must be a mapping"),
        (error_case({"parameter": "index", "base": {}}), "has no `period`"),
        (
            error_case({"parameter": "index", "base": {"year": 2020}}),
            "not `year`",
        ),
        (
            error_case({"parameter": "index", "base": {"period": "next year"}}),
            "not a period or instant",
        ),
        (
            error_case({"parameter": "index", "base": {"period": True}}),
            "not a period or instant",
        ),
        # Years past the C int range overflow when read as a date.
        (
            error_case({"parameter": "index", "base": {"period": 2**31}}),
            "not a period or instant",
        ),
        (
            error_case({"parameter": "index", "base": {"period": 10**10}}),
            "not a period or instant",
        ),
        # Unknown keys of mixed types are listed, not compared.
        (
            error_case({"parameter": "index", "base": {"period": 2020}, 1: 0, "z": 0}),
            "not `1`, `z`",
        ),
        (
            error_case({"parameter": "index", "base": {"period": 2020, 1: 0, "z": 0}}),
            "`base` takes only `period`, `value` and `floor`, not `1`, `z`",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": "nearest", "interval": 1, 1: 0, "z": 0},
                }
            ),
            "`rounding` takes only `type` and `interval`, not `1`, `z`",
        ),
        # A stated base must match to one part in a billion, however small
        # the amount: 0.0500000005 is 1e-8 off 0.05.
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020, "value": 0.0500000005}},
                values={"2020-01-01": 0.05},
            ),
            "but the parameter's value at the base period",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020, "value": "1,000"}}
            ),
            "not a finite number",
        ),
        (
            error_case({"parameter": "index", "base": {"period": 2020, "value": True}}),
            "not a finite number",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020, "floor": "yes"}}
            ),
            "is not `true` or `false`",
        ),
        (
            error_case({"parameter": "index", "base": {"period": 2020, "floor": 1}}),
            "is not `true` or `false`",
        ),
        (
            error_case({"parameter": "index", "base": {"period": 2020, "floor": None}}),
            "is not `true` or `false`",
        ),
        (
            error_case({"parameter": "self", "base": {"period": 2020}}),
            "names no index",
        ),
        (
            error_case({"base": {"period": 2020}}),
            "has no `parameter` that names an index",
        ),
        (
            error_case({"parameter": None, "base": {"period": 2020}}),
            "has no `parameter` that names an index",
        ),
        (
            error_case({"parameter": "index", "base": {"period": "2020-02-30"}}),
            "not a period or instant",
        ),
        (
            error_case({"parameter": "index", "base": {"period": 0}}),
            "not a period or instant",
        ),
        (
            error_case({"parameter": "index", "base": {"period": [2020]}}),
            "not a period or instant",
        ),
        (
            error_case({"parameter": "index", "base": {"period": 2020.0}}),
            "not a period or instant",
        ),
        (
            error_case({"parameter": "index", "base": {"period": "year:2020"}}),
            "not a period or instant",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020, "value": 10**400}}
            ),
            "not a finite number",
        ),
        (
            error_case({"parameter": "missing.index", "base": {"period": 2020}}),
            "Failed to uprate amount from its base: its index",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020, "value": 23_650}},
                values={"2020-01-01": 23_625},
            ),
            "but the parameter's value at the base period",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "start_instant": "2020-01-01",
                }
            ),
            "not `start_instant`",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "at_defined_interval": {
                        "start": "2000-10-01",
                        "end": "2001-10-01",
                        "enactment": "2002-04-01",
                    },
                }
            ),
            "not `at_defined_interval`",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rouding": {"type": "downwards", "interval": 50},
                }
            ),
            "not `rouding`",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": "down", "interval": 50},
                }
            ),
            "is not one of",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"interval": 50},
                }
            ),
            "is not one of",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": "downwards", "interval": 0},
                }
            ),
            "not a positive number",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": "downwards", "interval": -50},
                }
            ),
            "not a positive number",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020}, "rounding": True}
            ),
            "not a positive number",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": "downwards", "interval": 50, "round": 1},
                }
            ),
            "takes only `type` and `interval`, not `round`",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": ["downwards"], "interval": 50},
                }
            ),
            "is not one of",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": "downwards", "interval": 10**400},
                }
            ),
            "not a positive number",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2019}},
                index_values={"2019-01-01": 100.0, "2021-01-01": 110.0},
            ),
            "has no numeric value at the base period",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020}},
                values={"2020-01-01": float("nan")},
            ),
            "has no numeric value at the base period",
        ),
        (
            error_case({"parameter": "index", "base": {"period": 2019, "value": 1}}),
            "is None at 2019-01-01",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020}},
                index_values={"2020-01-01": 0.0, "2021-01-01": 1.0},
            ),
            "is 0.0 at 2020-01-01",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020}},
                index_values={"2020-01-01": -1.0, "2021-01-01": 1.0},
            ),
            "is -1.0 at 2020-01-01; base mode needs a finite, positive value",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020}},
                index_values={"2020-01-01": 1.0, "2021-01-01": float("inf")},
            ),
            "is inf at 2021-01-01",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020}},
                index_values={"2020-01-01": 1.0, "2021-01-01": float("nan")},
            ),
            "is nan at 2021-01-01",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": "nearest", "interval": 1e-10},
                },
                values={"2020-01-01": 1e300},
                index_values={"2020-01-01": 1.0, "2021-01-01": 1e10},
            ),
            "its projected value at 2021-01-01 is inf",
        ),
        (
            error_case(
                {
                    "parameter": "index",
                    "base": {"period": 2020},
                    "rounding": {"type": "nearest", "interval": 1e-10},
                },
                values={"2020-01-01": -1e300},
                index_values={"2020-01-01": 1.0, "2021-01-01": 1e10},
            ),
            "its projected value at 2021-01-01 is -inf",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020}},
                values={"2020-01-01": 1e300},
                index_values={"2020-01-01": 1.0, "2021-01-01": 1e10},
            ),
            "its projected value at 2021-01-01 is inf",
        ),
        (
            error_case(
                {"parameter": "index", "base": {"period": 2020}},
                values={"2020-01-01": 1_000, "2021-01-01": None},
                index_values={"2020-01-01": 1.0, "2022-01-01": 1.1},
            ),
            "is null",
        ),
    ],
)
def test_malformed_base_mode_metadata_raises(data, message):
    with pytest.raises(ValueError, match=message):
        uprated_root(data)


def test_errors_name_the_parameter():
    with pytest.raises(ValueError, match="Failed to uprate amount from its base"):
        uprated_root(error_case({"parameter": "index", "base": {}}))
    with pytest.raises(ValueError, match="Failed to uprate amount: "):
        uprated_root(error_case({"base": {"period": 2020}}))


def test_a_parameter_with_no_values_and_a_stated_base_raises():
    from policyengine_core.parameters import Parameter
    from policyengine_core.parameters.operations.uprate_parameters import (
        uprate_parameter_from_base,
    )

    root = ParameterNode(data={"index": {"values": {"2020-01-01": 1.0}}})
    parameter = Parameter("empty", data={})
    with pytest.raises(ValueError, match="no values to project after"):
        uprate_parameter_from_base(
            parameter,
            root,
            {"parameter": "index", "base": {"period": 2020, "value": 1}},
        )
