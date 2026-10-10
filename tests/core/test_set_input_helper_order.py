"""An input set over a longer period does not depend on what was calculated first.

``set_input_divide_by_period`` and ``set_input_dispatch_by_period`` spread an
input given for a longer period over a variable's own periods, leaving the
sub-periods that already have an input as they are. They used to treat any
stored value as such an input, including one the simulation had calculated
(a default, a formula result), so the same input gave different values
depending on what had been calculated before it was set.

``test_set_input_helper_order_property.py`` checks the same rule over random
sequences.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.holders import (
    set_input_dispatch_by_period,
    set_input_divide_by_period,
)
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.tools.simulation_dumper import (
    dump_simulation,
    restore_simulation,
)
from tests.fixtures.set_input_helper_order import (
    BRANCH_NAME,
    FormulaReturns999,
    MONTHS,
    Status,
    build_simulation,
    build_system,
    read,
)

MONTHS_2013 = MONTHS[:12]
YEARLY_INPUT = np.array([1200.0, 2400.0], dtype=np.float32)
MONTHLY_SHARE = np.array([100.0, 200.0], dtype=np.float32)
STATUS_INDEX = {status.name: status.index for status in Status}


def assert_reads(simulation, name, expected_by_period):
    for period, expected in expected_by_period.items():
        np.testing.assert_array_equal(
            read(simulation, name, period), expected, err_msg=f"{name} {period}"
        )


def assert_matches_fresh_inputs_before_and_after_replay(simulation, fresh, name):
    """Compare helper outputs and exported inputs before and after a cache wipe."""
    for replay in (False, True):
        if replay:
            simulation._invalidate_all_caches()
        assert simulation._user_input_keys == fresh._user_input_keys
        assert_reads(
            simulation,
            name,
            {period: read(fresh, name, period) for period in [*MONTHS_2013, "2013"]},
        )
        assert simulation.to_input_dataframe().equals(fresh.to_input_dataframe())


# Divided inputs


def test_divided_input_replaces_a_month_calculated_before_it():
    simulation = build_simulation()
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0]})

    simulation.set_input("flow_m", "2013", YEARLY_INPUT)

    assert_reads(
        simulation,
        "flow_m",
        {**{month: MONTHLY_SHARE for month in MONTHS_2013}, "2013": YEARLY_INPUT},
    )


def test_divided_input_is_accepted_after_the_year_was_calculated():
    simulation = build_simulation()
    assert_reads(simulation, "flow_m", {"2013": [0, 0]})

    # Every month now holds a calculated default, and the year their sum.
    simulation.set_input("flow_m", "2013", YEARLY_INPUT)

    assert_reads(
        simulation,
        "flow_m",
        {"2013-05": MONTHLY_SHARE, "2013-12": MONTHLY_SHARE, "2013": YEARLY_INPUT},
    )


def test_divided_input_keeps_month_inputs():
    simulation = build_simulation()
    simulation.set_input("flow_m", "2013-01", np.array([300.0, 600.0]))

    simulation.set_input("flow_m", "2013", YEARLY_INPUT)

    rest = (np.array([900.0, 1800.0], dtype=np.float32) / 11).astype(np.float32)
    assert_reads(
        simulation,
        "flow_m",
        {"2013-01": [300, 600], **{month: rest for month in MONTHS_2013[1:]}},
    )


def test_divided_input_keeps_month_inputs_and_replaces_calculated_months():
    simulation = build_simulation()
    simulation.set_input("flow_m", "2013-01", np.array([300.0, 600.0]))
    assert_reads(simulation, "flow_m", {"2013-02": [0, 0], "2013-01": [300, 600]})

    simulation.set_input("flow_m", "2013", YEARLY_INPUT)

    rest = (np.array([900.0, 1800.0], dtype=np.float32) / 11).astype(np.float32)
    assert_reads(
        simulation,
        "flow_m",
        {"2013-01": [300, 600], **{month: rest for month in MONTHS_2013[1:]}},
    )


def test_divided_input_is_set_from_a_situation_with_a_month_and_its_year():
    simulation = SimulationBuilder().build_from_entities(
        build_system(),
        {"persons": {"a": {"flow_m": {"2013": 1200, "2013-01": 300}}}},
    )

    rest = (np.array([900.0], dtype=np.float32) / 11).astype(np.float32)
    assert_reads(simulation, "flow_m", {"2013-01": [300], "2013-07": rest})


def test_divided_input_that_contradicts_the_inputs_is_refused():
    simulation = build_simulation()
    simulation.set_input("flow_m", "2013", YEARLY_INPUT)
    # Reading the months does not turn them into calculated values.
    assert_reads(simulation, "flow_m", {month: MONTHLY_SHARE for month in MONTHS_2013})

    with pytest.raises(ValueError, match="Inconsistent input"):
        simulation.set_input("flow_m", "2013", YEARLY_INPUT + 100)

    simulation.set_input("flow_m", "2013", YEARLY_INPUT)
    assert_reads(simulation, "flow_m", {"2013-06": MONTHLY_SHARE, "2013": YEARLY_INPUT})


def test_divided_input_replaces_formula_results():
    simulation = build_simulation()
    assert_reads(
        simulation, "formula_flow_m", {"2013-01": [11, 11], "2013": [198, 198]}
    )

    simulation.set_input("formula_flow_m", "2013", YEARLY_INPUT)

    assert_reads(
        simulation,
        "formula_flow_m",
        {"2013-01": MONTHLY_SHARE, "2013": YEARLY_INPUT, "2014-01": [11, 11]},
    )


def test_divided_input_to_a_yearly_variable_replaces_a_calculated_year():
    simulation = build_simulation()
    assert_reads(simulation, "flow_y", {"2013": [0, 0]})

    simulation.set_input("flow_y", "month:2013-01:24", YEARLY_INPUT * 2)

    assert_reads(simulation, "flow_y", {"2013": YEARLY_INPUT, "2014": YEARLY_INPUT})


def test_divided_input_to_a_yearly_variable_drops_the_twelfth_calculated_before():
    simulation = build_simulation()
    # A month of a yearly flow is a twelfth of the year, cached at the month.
    assert_reads(simulation, "flow_y", {"2013-05": [0, 0]})

    simulation.set_input("flow_y", "month:2013-01:12", YEARLY_INPUT)

    assert_reads(simulation, "flow_y", {"2013": YEARLY_INPUT, "2013-05": MONTHLY_SHARE})


# Dispatched inputs


@pytest.mark.parametrize("calculated_first", ["2013-01", "2013-05", "2013-12", "2013"])
def test_dispatched_input_replaces_months_calculated_before_it(calculated_first):
    simulation = build_simulation()
    assert_reads(simulation, "count_m", {calculated_first: [0, 0]})

    simulation.set_input("count_m", "2013", np.array([7, 9]))

    assert_reads(
        simulation,
        "count_m",
        {**{month: [7, 9] for month in MONTHS_2013}, "2013": [7, 9]},
    )


def test_dispatched_input_applies_an_earlier_month_input_to_the_months_after_it():
    simulation = build_simulation()
    simulation.set_input("count_m", "2013-03", np.array([3, 4]))

    simulation.set_input("count_m", "2013", np.array([7, 9]))

    assert_reads(
        simulation,
        "count_m",
        {
            "2013-01": [7, 9],
            "2013-02": [7, 9],
            **{month: [3, 4] for month in MONTHS_2013[2:]},
        },
    )


def test_dispatched_input_keeps_month_inputs_and_replaces_calculated_months():
    simulation = build_simulation()
    simulation.set_input("count_m", "2013-03", np.array([3, 4]))
    assert_reads(simulation, "count_m", {"2013-02": [0, 0], "2013-06": [0, 0]})

    simulation.set_input("count_m", "2013", np.array([7, 9]))

    assert_reads(
        simulation,
        "count_m",
        {
            "2013-01": [7, 9],
            "2013-02": [7, 9],
            **{month: [3, 4] for month in MONTHS_2013[2:]},
        },
    )


def test_dispatched_input_changes_nothing_when_every_month_has_an_input():
    simulation = build_simulation()
    simulation.set_input("count_m", "2013", np.array([7, 9]))
    assert_reads(simulation, "count_m", {month: [7, 9] for month in MONTHS_2013})

    simulation.set_input("count_m", "2013", np.array([1, 2]))

    assert_reads(simulation, "count_m", {month: [7, 9] for month in MONTHS_2013})


def test_dispatched_flow_input_drops_the_sum_calculated_before_it():
    simulation = build_simulation()
    assert_reads(simulation, "dispatched_flow_m", {"2013": [0, 0]})

    simulation.set_input("dispatched_flow_m", "2013", np.array([5.0, 6.0]))

    assert_reads(simulation, "dispatched_flow_m", {"2013-04": [5, 6], "2013": [60, 72]})


def test_dispatched_input_replaces_calculated_booleans_and_enums():
    simulation = build_simulation()
    assert_reads(simulation, "flag_m", {"2013-01": [False, False]})
    assert_reads(simulation, "status_m", {"2013-01": [STATUS_INDEX["none"]] * 2})

    simulation.set_input("flag_m", "2013", np.array([True, False]))
    simulation.set_input("status_m", "2013", np.array(["all", "some"]))

    assert_reads(
        simulation, "flag_m", {"2013-01": [True, False], "2013-08": [True, False]}
    )
    assert_reads(
        simulation,
        "status_m",
        {
            month: [STATUS_INDEX["all"], STATUS_INDEX["some"]]
            for month in ("2013-01", "2013-08")
        },
    )


def test_dispatched_input_to_a_yearly_variable_replaces_a_calculated_year():
    simulation = build_simulation()
    assert_reads(simulation, "count_y", {"2013": [0, 0], "2014-06": [0, 0]})

    simulation.set_input("count_y", "month:2013-01:24", np.array([7, 9]))

    assert_reads(
        simulation, "count_y", {"2013": [7, 9], "2014": [7, 9], "2014-06": [7, 9]}
    )


# What the helpers leave alone


def test_values_calculated_outside_the_input_period_are_kept():
    simulation = build_simulation()
    assert_reads(simulation, "flow_m", {"2014-03": [0, 0], "2014": [0, 0]})
    assert_reads(simulation, "flow_m", {"2012-12": [0, 0], "2012": [0, 0]})
    holder = simulation.get_holder("flow_m")

    simulation.set_input("flow_m", "2013", YEARLY_INPUT)

    for period in ("2012-12", "2012", "2014-03", "2014"):
        assert holder.get_array(period) is not None, period


def test_a_sum_calculated_over_a_period_that_overlaps_the_input_is_dropped():
    simulation = build_simulation()
    assert_reads(simulation, "flow_m", {"year:2013:2": [0, 0]})

    simulation.set_input("flow_m", "2014", YEARLY_INPUT)

    assert_reads(
        simulation, "flow_m", {"year:2013:2": YEARLY_INPUT, "2014": YEARLY_INPUT}
    )


def test_an_input_stored_at_a_longer_period_is_kept():
    simulation = build_simulation()
    # Twelve months from January have the unit of a monthly variable, so the
    # input is stored as given, under the year.
    simulation.set_input("flow_m", "month:2013-01:12", np.array([50.0, 60.0]))

    simulation.set_input("flow_m", "year:2013:2", YEARLY_INPUT * 2)

    holder = simulation.get_holder("flow_m")
    np.testing.assert_array_equal(holder.get_array("2013"), [50, 60])
    assert_reads(
        simulation, "flow_m", {"2013-01": MONTHLY_SHARE, "2014-12": MONTHLY_SHARE}
    )


def test_an_input_stored_for_twelve_months_from_another_month_is_kept():
    simulation = build_simulation()
    # Storage keys twelve months from March as the year starting in March.
    simulation.set_input("flow_m", "month:2013-03:12", np.array([50.0, 60.0]))

    simulation.set_input("flow_m", "year:2013:2", YEARLY_INPUT * 2)

    holder = simulation.get_holder("flow_m")
    np.testing.assert_array_equal(holder.get_array("year:2013-03"), [50, 60])
    assert_reads(simulation, "flow_m", {"2013-05": MONTHLY_SHARE})


# The record of inputs, and calculate's fast cache


def test_a_replaced_value_is_recorded_as_an_input():
    simulation = build_simulation()
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0]})

    simulation.set_input("flow_m", "2013", YEARLY_INPUT)

    january = periods.period("2013-01")
    assert ("flow_m", "default", january) in simulation._user_input_keys
    # Inputs survive the cache wipe that follows a reform; calculated values do not.
    simulation._invalidate_all_caches()
    np.testing.assert_array_equal(
        simulation.get_holder("flow_m").get_array(january), MONTHLY_SHARE
    )


def test_calculate_returns_the_input_after_a_repeated_read():
    simulation = build_simulation()
    # The second read is answered from calculate's fast cache.
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0]})
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0]})
    assert_reads(simulation, "count_m", {"2013-01": [0, 0]})
    assert_reads(simulation, "count_m", {"2013-01": [0, 0]})

    simulation.set_input("flow_m", "2013", YEARLY_INPUT)
    simulation.set_input("count_m", "2013", np.array([7, 9]))

    assert_reads(simulation, "flow_m", {"2013-01": MONTHLY_SHARE})
    assert_reads(simulation, "count_m", {"2013-01": [7, 9]})


def test_a_dropped_sum_leaves_calculates_fast_cache():
    simulation = build_simulation()
    assert_reads(simulation, "flow_m", {"year:2013:2": [0, 0]})
    # ``calculate`` does not put a sum in its fast cache today; a value held
    # there for a period whose stored value is dropped goes with it.
    two_years = periods.period("year:2013:2")
    simulation._fast_cache[("flow_m", two_years)] = np.zeros(2, dtype=np.float32)

    simulation.set_input("flow_m", "2014", YEARLY_INPUT)

    assert ("flow_m", two_years) not in simulation._fast_cache
    assert_reads(simulation, "flow_m", {"year:2013:2": YEARLY_INPUT})


def test_an_input_under_a_branch_the_simulation_does_not_read_keeps_its_fast_cache():
    simulation = build_simulation()
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0]})
    cached = simulation._fast_cache[("flow_m", periods.period("2013-01"))]

    simulation.get_holder("flow_m").set_input(
        periods.period("2013"), YEARLY_INPUT, "elsewhere"
    )

    # The simulation reads only ``default``, so what it returned is unchanged.
    assert simulation._fast_cache[("flow_m", periods.period("2013-01"))] is cached
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0]})


@pytest.mark.parametrize("on_disk", [False, True])
def test_a_helper_called_directly_stores_inputs(on_disk):
    simulation = build_simulation(on_disk=on_disk)
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0]})
    assert_reads(simulation, "count_m", {"2013-01": [0, 0]})
    year = periods.period("2013")

    set_input_divide_by_period(simulation.get_holder("flow_m"), year, YEARLY_INPUT)
    set_input_dispatch_by_period(
        simulation.get_holder("count_m"), year, np.array([7, 9])
    )

    january = periods.period("2013-01")
    assert ("flow_m", "default", january) in simulation._user_input_keys
    assert ("count_m", "default", january) in simulation._user_input_keys
    location = "disk" if on_disk else "memory"
    for name in ("flow_m", "count_m"):
        holder = simulation.get_holder(name)
        for month in MONTHS_2013:
            month_period = periods.period(month)
            assert holder._user_input_storage[("default", month)] == (
                month_period,
                frozenset({location}),
            )
            assert holder._stores_user_input(month_period, "default")
    assert_reads(simulation, "flow_m", {"2013-01": MONTHLY_SHARE})
    assert_reads(simulation, "count_m", {"2013-01": [7, 9]})
    # A second call finds those inputs, as it would after ``set_input``.
    with pytest.raises(ValueError, match="Inconsistent input"):
        set_input_divide_by_period(
            simulation.get_holder("flow_m"), year, YEARLY_INPUT + 100
        )
    if on_disk:
        simulation.memory_config.max_memory_occupation_pc = 101
        for name in ("flow_m", "count_m"):
            simulation.get_holder(name).put_in_cache(
                np.array([700, 900]), january, derived=False
            )
    set_input_divide_by_period(simulation.get_holder("flow_m"), year, YEARLY_INPUT)
    set_input_dispatch_by_period(
        simulation.get_holder("count_m"), year, np.array([1, 2])
    )
    assert_reads(simulation, "count_m", {"2013-01": [7, 9]})
    simulation._invalidate_all_caches()
    assert_reads(simulation, "flow_m", {"2013-01": MONTHLY_SHARE, "2013": YEARLY_INPUT})
    assert_reads(simulation, "count_m", {"2013-01": [7, 9], "2013": [7, 9]})
    assert simulation.to_input_dataframe()["flow_m__2013-01"].tolist() == [100, 200]
    assert simulation.to_input_dataframe()["count_m__2013-01"].tolist() == [7, 9]


@pytest.mark.parametrize(
    "name, month_value, year_value",
    [
        ("flow_m", [300.0, 600.0], YEARLY_INPUT),
        ("count_m", [3, 4], [7, 9]),
    ],
)
@pytest.mark.parametrize("on_disk", [False, True])
def test_helper_ignores_unregistered_carry_over_cache(
    name, month_value, year_value, on_disk
):
    simulation = build_simulation(on_disk=on_disk)
    holder = simulation.get_holder(name)
    march = periods.period("2013-03")
    holder.put_in_cache(np.array(month_value), march, derived=False)
    # Carry-over provenance does not make a cache write a supplied input.
    assert not holder.is_derived(march)
    assert (name, "default", march) not in simulation._user_input_keys
    read(simulation, name, "2013")
    fresh = build_simulation()

    simulation.set_input(name, "2013", np.array(year_value))
    fresh.set_input(name, "2013", np.array(year_value))

    assert_matches_fresh_inputs_before_and_after_replay(simulation, fresh, name)


@pytest.mark.parametrize(
    "name, month_value, year_value",
    [
        ("flow_m", [300.0, 600.0], YEARLY_INPUT),
        ("count_m", [3, 4], [7, 9]),
    ],
)
@pytest.mark.parametrize("registered", [False, True])
def test_helper_ignores_a_record_after_cache_replaced_the_only_input(
    name, month_value, year_value, registered
):
    simulation = build_simulation()
    holder = simulation.get_holder(name)
    march = periods.period("2013-03")
    if registered:
        holder._memory_storage.put(
            np.array(month_value, dtype=holder.variable.dtype), march
        )
        simulation._user_input_keys.add((name, "default", march))
        assert not hasattr(holder, "_user_input_storage")
    else:
        simulation.set_input(name, march, np.array(month_value))
    holder.put_in_cache(np.array([700, 900]), march, derived=False)
    assert (name, "default", march) in simulation._user_input_keys
    assert not holder._stores_user_input(march, "default")
    read(simulation, name, "2013")
    # The replaced input no longer survives in either tier.
    fresh = build_simulation()

    simulation.set_input(name, "2013", np.array(year_value))
    fresh.set_input(name, "2013", np.array(year_value))

    assert_matches_fresh_inputs_before_and_after_replay(simulation, fresh, name)


@pytest.mark.parametrize(
    "name, month_value, year_value",
    [
        ("flow_m", [300.0, 600.0], YEARLY_INPUT),
        ("count_m", [3, 4], [7, 9]),
    ],
)
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("derived", [False, True])
def test_helper_keeps_disk_input_hidden_by_memory_cache(
    name, month_value, year_value, registered, derived
):
    simulation = build_simulation(on_disk=True)
    holder = simulation.get_holder(name)
    march = periods.period("2013-03")
    if registered:
        holder._disk_storage.put(
            np.array(month_value, dtype=holder.variable.dtype), march
        )
        simulation._user_input_keys.add((name, "default", march))
        assert not hasattr(holder, "_user_input_storage")
    else:
        simulation.set_input(name, march, np.array(month_value))
    simulation.memory_config.max_memory_occupation_pc = 101
    if derived:
        # Model a derived shadow already stored above the supplied disk input.
        holder._memory_storage.put(
            np.array([700, 900], dtype=holder.variable.dtype), march, derived=True
        )
    else:
        holder.put_in_cache(np.array([700, 900]), march, derived=False)
    assert holder._stores_user_input(march, "default")
    np.testing.assert_array_equal(holder.get_array(march), [700, 900])
    read(simulation, name, "2013")
    fresh = build_simulation()
    fresh.set_input(name, march, np.array(month_value))

    simulation.set_input(name, "2013", np.array(year_value))
    fresh.set_input(name, "2013", np.array(year_value))

    np.testing.assert_array_equal(holder._disk_storage.get(march), month_value)
    assert holder._memory_storage.get(march) is None
    assert_matches_fresh_inputs_before_and_after_replay(simulation, fresh, name)


@pytest.mark.parametrize(
    "name, year_value",
    [("flow_m", YEARLY_INPUT), ("count_m", [7, 9])],
)
def test_repeated_helper_clears_cache_shadow_when_every_month_has_a_disk_input(
    name, year_value
):
    simulation = build_simulation(on_disk=True)
    fresh = build_simulation()
    for candidate in (simulation, fresh):
        candidate.set_input(name, "2013", np.array(year_value))
    holder = simulation.get_holder(name)
    january = periods.period("2013-01")
    simulation.memory_config.max_memory_occupation_pc = 101
    holder.put_in_cache(np.array([700, 900]), january, derived=False)
    read(simulation, name, "2013")

    # The repeated helper writes no new months, but the cache shadow still
    # must be removed so reads agree with the supplied disk inputs.
    simulation.set_input(name, "2013", np.array(year_value))

    assert holder._memory_storage.get(january) is None
    assert_matches_fresh_inputs_before_and_after_replay(simulation, fresh, name)


@pytest.mark.parametrize("on_disk", [False, True])
def test_helper_drops_overlapping_early_year_cache(on_disk):
    simulation = build_simulation(on_disk=on_disk)
    holder = simulation.get_holder("flow_m")
    year = periods.period("0025")
    holder.put_in_cache(np.array([5.0, 6.0]), year, derived=True)
    # Plain reads do not fast-cache an aggregate today; model an existing
    # fast-cache entry to prove cleanup uses the early-year Period itself.
    simulation._fast_cache[("flow_m", year)] = np.array([5.0, 6.0])
    fresh = build_simulation()

    set_input_divide_by_period(holder, year, YEARLY_INPUT)
    fresh.set_input("flow_m", year, YEARLY_INPUT)

    assert holder.get_array(year) is None
    assert ("flow_m", year) not in simulation._fast_cache
    january = year.start.period(periods.MONTH)
    for month in [january.offset(index) for index in range(12)]:
        np.testing.assert_array_equal(
            simulation.calculate("flow_m", month), fresh.calculate("flow_m", month)
        )
    np.testing.assert_array_equal(
        simulation.calculate("flow_m", year), fresh.calculate("flow_m", year)
    )


def test_every_stored_value_counts_as_an_input_without_a_record():
    simulation = build_simulation()
    holder = simulation.get_holder("flow_m")
    holder.put_in_cache(np.array([300.0, 600.0]), periods.period("2013-01"))
    # A simulation with no record of its inputs cannot tell them apart from
    # what it calculated.
    del simulation._user_input_keys

    set_input_divide_by_period(holder, periods.period("2013"), YEARLY_INPUT)

    rest = (np.array([900.0, 1800.0], dtype=np.float32) / 11).astype(np.float32)
    np.testing.assert_array_equal(holder.get_array("2013-01"), [300, 600])
    np.testing.assert_array_equal(holder.get_array("2013-02"), rest)


# Branches


def test_input_on_a_branch_drops_the_sum_its_parent_calculated():
    simulation = build_simulation()
    assert_reads(simulation, "flow_m", {"2013": [0, 0]})
    branch = simulation.get_branch(BRANCH_NAME)

    branch.set_input("flow_m", "2013", YEARLY_INPUT)

    assert_reads(branch, "flow_m", {"2013-01": MONTHLY_SHARE, "2013": YEARLY_INPUT})
    # The simulation the branch came from keeps what it calculated.
    assert simulation.get_holder("flow_m").get_array("2013") is not None
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0], "2013": [0, 0]})


def test_input_on_a_branch_replaces_what_the_branch_calculated():
    simulation = build_simulation()
    branch = simulation.get_branch(BRANCH_NAME)
    assert_reads(branch, "flow_m", {"2013-01": [0, 0], "2013": [0, 0]})
    assert_reads(branch, "count_m", {"2013-06": [0, 0]})

    branch.set_input("flow_m", "2013", YEARLY_INPUT)
    branch.set_input("count_m", "2013", np.array([7, 9]))

    assert_reads(branch, "flow_m", {"2013-01": MONTHLY_SHARE, "2013": YEARLY_INPUT})
    assert_reads(branch, "count_m", {"2013-06": [7, 9], "2013-12": [7, 9]})
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0]})
    assert_reads(simulation, "count_m", {"2013-06": [0, 0]})


def test_input_stored_under_default_replaces_what_the_branch_calculated():
    simulation = build_simulation()
    branch = simulation.get_branch(BRANCH_NAME)
    # The branch stores what it calculates under its own name, which it
    # reads before ``default``.
    assert_reads(branch, "flow_m", {"2013-01": [0, 0], "2013": [0, 0]})

    # ``Holder.set_input`` stores under ``default`` unless given a branch.
    branch.get_holder("flow_m").set_input(periods.period("2013"), YEARLY_INPUT)

    assert_reads(branch, "flow_m", {"2013-01": MONTHLY_SHARE, "2013": YEARLY_INPUT})


def test_input_on_a_nested_branch_drops_the_sum_its_parent_branch_calculated():
    simulation = build_simulation()
    parent = simulation.get_branch(BRANCH_NAME)
    # The parent branch stores the sum under its own name, which the nested
    # branch reads before ``default``.
    assert_reads(parent, "flow_m", {"2013": [0, 0]})
    nested = parent.get_branch("nested")

    nested.set_input("flow_m", "2013", YEARLY_INPUT)

    assert_reads(nested, "flow_m", {"2013-01": MONTHLY_SHARE, "2013": YEARLY_INPUT})
    assert_reads(parent, "flow_m", {"2013": [0, 0]})


# Restored simulations


@pytest.mark.parametrize(
    "name, month, month_value, year_value, later, later_value",
    [
        ("flow_m", "2013-01", [300.0, 600.0], YEARLY_INPUT, "2013-02", None),
        ("count_m", "2013-03", [3, 4], [7, 9], "2013-04", [3, 4]),
    ],
)
def test_a_restored_month_keeps_its_value_under_a_yearly_input(
    name, month, month_value, year_value, later, later_value
):
    simulation = build_simulation()
    simulation.set_input(name, month, np.array(month_value))
    with tempfile.TemporaryDirectory(prefix="core-set-input-helpers-") as directory:
        dump_simulation(simulation, directory)
        restored = restore_simulation(directory, simulation.tax_benefit_system)

    # The dump restores the month as an input, so the yearly helper keeps it.
    restored.set_input(name, "2013", np.array(year_value))

    if later_value is None:
        rest = np.array(year_value, dtype=np.float32) - np.array(
            month_value, dtype=np.float32
        )
        later_value = (rest / 11).astype(np.float32)
    assert_reads(restored, name, {month: month_value, later: later_value})


def test_a_restored_calculated_value_is_recalculated_after_a_reform():
    simulation = build_simulation()
    assert_reads(simulation, "formula_flow_m", {"2013-01": [11, 11]})
    with tempfile.TemporaryDirectory(prefix="core-set-input-helpers-") as directory:
        dump_simulation(simulation, directory)
        restored = restore_simulation(directory, simulation.tax_benefit_system)

    # The dump records which values were inputs, so a calculated one is
    # restored as calculated and a reform recalculates it.
    restored.apply_reform(FormulaReturns999)

    assert_reads(restored, "formula_flow_m", {"2013-01": [999, 999]})


# Values stored on disk


def test_divided_input_replaces_values_calculated_and_stored_on_disk():
    simulation = build_simulation(on_disk=True)
    assert_reads(simulation, "flow_m", {"2013-01": [0, 0], "2013": [0, 0]})
    holder = simulation.get_holder("flow_m")
    assert holder._disk_storage.get(periods.period("2013")) is not None

    simulation.set_input("flow_m", "2013", YEARLY_INPUT)

    assert_reads(simulation, "flow_m", {"2013-01": MONTHLY_SHARE, "2013": YEARLY_INPUT})


def test_dispatched_input_replaces_values_calculated_and_stored_on_disk():
    simulation = build_simulation(on_disk=True)
    assert_reads(simulation, "count_m", {"2013-05": [0, 0]})
    assert_reads(simulation, "dispatched_flow_m", {"2013": [0, 0]})

    simulation.set_input("count_m", "2013", np.array([7, 9]))
    simulation.set_input("dispatched_flow_m", "2013", np.array([5.0, 6.0]))

    assert_reads(simulation, "count_m", {"2013-05": [7, 9], "2013-09": [7, 9]})
    assert_reads(simulation, "dispatched_flow_m", {"2013-04": [5, 6], "2013": [60, 72]})


def test_input_on_a_branch_drops_a_sum_stored_on_disk_under_its_name():
    simulation = build_simulation(on_disk=True)
    # The branch name contains the separator of disk storage keys. (Disk
    # storage cannot list the periods stored under such a name, so the sum
    # is stored directly instead of being calculated on the branch.)
    assert "_" in BRANCH_NAME
    branch = simulation.get_branch(BRANCH_NAME)
    holder = branch.get_holder("flow_m")
    year = periods.period("2013")
    holder._disk_storage.put(
        np.array([5.0, 5.0], dtype=np.float32), year, BRANCH_NAME, derived=True
    )

    branch.set_input("flow_m", "2013", YEARLY_INPUT)

    assert holder._disk_storage.get(year, BRANCH_NAME) is None
    np.testing.assert_array_equal(
        holder._disk_storage.get(periods.period("2013-01"), BRANCH_NAME),
        MONTHLY_SHARE,
    )
