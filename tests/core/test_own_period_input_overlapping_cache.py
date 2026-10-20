"""An input for one of a variable's own periods updates its sums and twelfths.

``calculate`` adds up a monthly flow over a year and divides a yearly flow
into a month, and caches the result at that period. An input set for one
month (or one year) of the variable replaced the value stored there but left
those results in place, so a later ``calculate`` returned the old sum or
twelfth (policyengine-core#579). The result depended on whether the longer
(or shorter) period had been read before the input was set.

``set_input`` now drops what the simulation calculated for the variable over
periods that overlap the input's, by the rule the ``set_input`` helpers
follow (``test_set_input_helper_order.py``). Inputs are kept.
``test_set_input_helper_order_property.py`` compares such reads with a
simulation that calculated nothing first.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.set_input_helper_order import (
    BRANCH_NAME,
    build_simulation,
    read,
)


def _sequence(simulation, warm):
    """Issue #579's table: read the other period or not, set, read again."""
    if warm:
        read(simulation, "flow_m", "2013")
        read(simulation, "flow_y", "2013-05")
    simulation.set_input("flow_m", "2013-05", np.array([50.0, 0.0]))
    simulation.set_input("flow_y", "2013", np.array([1200.0, 0.0]))
    return read(simulation, "flow_m", "2013"), read(simulation, "flow_y", "2013-05")


@pytest.mark.parametrize("on_disk", [False, True])
def test_sum_and_twelfth_follow_an_input_set_after_them(on_disk):
    cold = _sequence(build_simulation(on_disk=on_disk), warm=False)
    warm = _sequence(build_simulation(on_disk=on_disk), warm=True)

    for result in (cold, warm):
        np.testing.assert_array_equal(result[0], [50.0, 0.0])
        np.testing.assert_array_equal(result[1], [100.0, 0.0])


def test_annual_sum_follows_a_january_input():
    # The audit witness: a warm annual flow of 0, then January set to 1.
    simulation = build_simulation()
    np.testing.assert_array_equal(read(simulation, "flow_m", "2013"), [0.0, 0.0])

    simulation.set_input("flow_m", "2013-01", np.array([1.0, 1.0]))

    np.testing.assert_array_equal(read(simulation, "flow_m", "2013"), [1.0, 1.0])


def test_inputs_for_other_months_are_kept():
    simulation = build_simulation()
    simulation.set_input("flow_m", "2013-04", np.array([7.0, 0.0]))
    np.testing.assert_array_equal(read(simulation, "flow_m", "2013"), [7.0, 0.0])

    simulation.set_input("flow_m", "2013-05", np.array([50.0, 0.0]))

    np.testing.assert_array_equal(read(simulation, "flow_m", "2013-04"), [7.0, 0.0])
    np.testing.assert_array_equal(read(simulation, "flow_m", "2013"), [57.0, 0.0])
    assert ("flow_m", "default", "2013-04") in {
        (name, branch, str(period))
        for name, branch, period in simulation._user_input_keys
    }


def test_sum_over_two_years_follows_an_input():
    simulation = build_simulation()
    np.testing.assert_array_equal(read(simulation, "flow_m", "year:2013:2"), [0.0, 0.0])

    simulation.set_input("flow_m", "2014-12", np.array([3.0, 0.0]))

    np.testing.assert_array_equal(read(simulation, "flow_m", "year:2013:2"), [3.0, 0.0])


def test_branch_input_updates_the_sum_the_branch_reads():
    # The branch reads the sum its parent calculated until it is dropped;
    # the parent keeps its own sum, which its inputs still give.
    simulation = build_simulation()
    np.testing.assert_array_equal(read(simulation, "flow_m", "2013"), [0.0, 0.0])
    branch = simulation.get_branch(BRANCH_NAME)

    branch.set_input("flow_m", "2013-05", np.array([50.0, 0.0]))

    np.testing.assert_array_equal(read(branch, "flow_m", "2013"), [50.0, 0.0])
    np.testing.assert_array_equal(read(simulation, "flow_m", "2013"), [0.0, 0.0])


def test_input_through_the_holder_updates_the_sum():
    # ``Simulation.set_input`` drops the month from ``calculate``'s fast
    # cache; an input set on the holder has to as well.
    simulation = build_simulation()
    np.testing.assert_array_equal(read(simulation, "flow_m", "2013"), [0.0, 0.0])

    simulation.get_holder("flow_m").set_input("2013-05", np.array([50.0, 0.0]))

    np.testing.assert_array_equal(read(simulation, "flow_m", "2013"), [50.0, 0.0])
