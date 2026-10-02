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

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.holders import (
    set_input_dispatch_by_period,
    set_input_divide_by_period,
)
from policyengine_core.simulations import SimulationBuilder
from tests.fixtures.set_input_helper_order import (
    BRANCH_NAME,
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


def test_a_helper_called_directly_stores_inputs():
    simulation = build_simulation()
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
    assert_reads(simulation, "flow_m", {"2013-01": MONTHLY_SHARE})
    assert_reads(simulation, "count_m", {"2013-01": [7, 9]})
    # A second call finds those inputs, as it would after ``set_input``.
    with pytest.raises(ValueError, match="Inconsistent input"):
        set_input_divide_by_period(
            simulation.get_holder("flow_m"), year, YEARLY_INPUT + 100
        )
    set_input_dispatch_by_period(
        simulation.get_holder("count_m"), year, np.array([1, 2])
    )
    assert_reads(simulation, "count_m", {"2013-01": [7, 9]})


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
    holder._disk_storage.put(np.array([5.0, 5.0], dtype=np.float32), year, BRANCH_NAME)

    branch.set_input("flow_m", "2013", YEARLY_INPUT)

    assert holder._disk_storage.get(year, BRANCH_NAME) is None
    np.testing.assert_array_equal(
        holder._disk_storage.get(periods.period("2013-01"), BRANCH_NAME),
        MONTHLY_SHARE,
    )
