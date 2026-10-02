"""What a simulation calculated before its inputs were set does not change them.

Each example sets some inputs, has one simulation calculate a random list of
requests, then sets more inputs on that simulation and on one that
calculated nothing, and compares the two. It also compares both with a model
that holds inputs only (``tests/fixtures/set_input_helper_order.py``).
``test_set_input_helper_order.py`` pins the same behaviour with examples.

Outside the properties, because they are other order dependences:

- Other variables. A value calculated earlier from the variable an input is
  set for stays cached, as after any ``set_input``, so the properties read
  only the variables the inputs are set for, whose one formula reads nothing.
- Auto-carry-over and uprating, which read whatever periods are stored
  (policyengine-core#562, #563): the test system uses neither.
- ``calculate_add`` and ``calculate_divide`` called directly, whose results
  are cached where a plain read does not return them (#571): the requests
  are plain ``calculate`` calls.
- An input set for one of the variable's own periods (one month, or one
  year) after a sum or a twelfth of the variable was calculated: the helpers
  are not involved and that sum stays cached (policyengine-core#579). The
  periods of another size are compared only for variables given no such
  input after the calculations.
"""

from __future__ import annotations

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from tests.fixtures.set_input_helper_order import (
    BRANCH_NAME,
    COUNT,
    DIVIDED,
    FLOWS,
    HELPER_PERIODS,
    NAMES,
    OTHER_READS,
    OWN_PERIODS,
    OWN_READS,
    Reference,
    Status,
    apply_input,
    build_simulation,
    build_system,
    input_record,
    read,
    read_all,
    same_arrays,
    sub_periods,
)

SYSTEM = build_system()
STATUSES = [status.name for status in Status]


def _values(name):
    if name == "flag_m":
        element = st.booleans()
    elif name == "status_m":
        element = st.sampled_from(STATUSES)
    elif name in ("count_m", "count_y"):
        element = st.integers(-50, 50)
    else:
        # Whole numbers, so that every stored float32 is the same on any
        # platform; the shares of a divided input are not whole.
        element = st.integers(-1200, 1200).map(float)
    return st.lists(element, min_size=COUNT, max_size=COUNT)


def _input(periods_of):
    return st.sampled_from(NAMES).flatmap(
        lambda name: st.tuples(
            st.just(name), st.sampled_from(periods_of(name)), _values(name)
        )
    )


own_input = _input(lambda name: OWN_PERIODS[name])
helper_input = _input(lambda name: HELPER_PERIODS[name])
any_input = st.one_of(own_input, helper_input, helper_input)
request = st.sampled_from(NAMES).flatmap(
    lambda name: st.tuples(
        st.just(name), st.sampled_from(OWN_READS[name] + OTHER_READS[name])
    )
)

SETTINGS = dict(
    deadline=None,
    suppress_health_check=[
        hypothesis.HealthCheck.too_slow,
        hypothesis.HealthCheck.data_too_large,
    ],
)


def _run(first_inputs, requests_before_branch, requests, later_inputs, on_disk, branch):
    """Run the example; return the simulation the later inputs were set on,
    each later input's outcome, and the model's."""
    simulation = build_simulation(SYSTEM, on_disk=on_disk)
    reference = Reference()
    for name, period, values in first_inputs:
        assert apply_input(simulation, name, period, values) == reference.set_input(
            name, period, values
        )
    for name, period in requests_before_branch:
        read(simulation, name, period)
    branch_name = "default"
    if branch:
        simulation = simulation.get_branch(BRANCH_NAME)
        branch_name = BRANCH_NAME
    for name, period in requests:
        read(simulation, name, period)
    outcomes = [
        apply_input(simulation, name, period, values)
        for name, period, values in later_inputs
    ]
    expected = [
        reference.set_input(name, period, values, branch_name)
        for name, period, values in later_inputs
    ]
    return simulation, outcomes, expected, reference, branch_name


@hypothesis.settings(max_examples=500, **SETTINGS)
@hypothesis.given(
    first_inputs=st.lists(any_input, max_size=4),
    requests_before_branch=st.lists(request, max_size=6),
    requests=st.lists(request, max_size=8),
    later_inputs=st.lists(any_input, min_size=1, max_size=6),
    on_disk=st.booleans(),
    branch=st.booleans(),
)
def test_inputs_do_not_depend_on_what_was_calculated_before(
    first_inputs, requests_before_branch, requests, later_inputs, on_disk, branch
):
    # Branches of a simulation that stores on disk share its files
    # (policyengine-core#558), so the two are not combined.
    on_disk = on_disk and not branch
    calculated, outcomes, expected, reference, branch_name = _run(
        first_inputs, requests_before_branch, requests, later_inputs, on_disk, branch
    )
    fresh, fresh_outcomes, _, _, _ = _run(
        first_inputs, [], [], later_inputs, on_disk, branch
    )

    # The same inputs are accepted or refused, and recorded.
    assert outcomes == fresh_outcomes == expected
    assert input_record(calculated) == input_record(fresh)

    # A sum or a twelfth calculated before an input for one of the
    # variable's own periods stays cached (policyengine-core#579).
    given_own_period_input = {
        name for name, period, _ in later_inputs if period in OWN_PERIODS[name]
    }
    other_periods_of = [name for name in NAMES if name not in given_own_period_input]
    values = read_all(calculated, other_periods_of=other_periods_of)
    fresh_values = read_all(fresh, other_periods_of=other_periods_of)
    assert values.keys() == fresh_values.keys()
    for key in values:
        assert same_arrays(values[key], fresh_values[key]), (
            key,
            values[key],
            fresh_values[key],
        )

    # Both hold the inputs the inputs-only model holds.
    for name in NAMES:
        for period in OWN_READS[name]:
            expected_value = reference.value(name, period, branch_name)
            assert same_arrays(values[(name, period)], expected_value), (
                name,
                period,
                values[(name, period)],
                expected_value,
            )


@hypothesis.settings(max_examples=300, **SETTINGS)
@hypothesis.given(
    first_inputs=st.lists(own_input, max_size=4),
    requests=st.lists(request, max_size=8),
    name=st.sampled_from(DIVIDED),
    data=st.data(),
)
def test_a_divided_input_adds_up_to_the_value_given(first_inputs, requests, name, data):
    """Conservation: a divided input is accepted while a sub-period is left
    without an input, the variable's inputs over the period then add up to
    it, and a plain read of the period returns that sum, whatever was
    calculated before."""
    period = data.draw(st.sampled_from(HELPER_PERIODS[name]))
    values = data.draw(_values(name))
    simulation = build_simulation(SYSTEM)
    for input_name, input_period, input_values in first_inputs:
        assert apply_input(simulation, input_name, input_period, input_values) is None
    for request_name, request_period in requests:
        read(simulation, request_name, request_period)

    # The input is refused only when every sub-period already has one.
    with_input = {
        input_period
        for input_name, input_period, _ in first_inputs
        if input_name == name
    }
    hypothesis.assume(set(sub_periods(name, period)) - with_input)
    assert apply_input(simulation, name, period, values) is None

    given = np.array(values, dtype=np.float32)
    total = sum(
        read(simulation, name, sub_period).astype(np.float64)
        for sub_period in sub_periods(name, period)
    )
    np.testing.assert_allclose(total, given, rtol=1e-5, atol=1e-2)
    if name in FLOWS and period in OTHER_READS[name]:
        # The same period, read as ``calculate`` adds it up.
        np.testing.assert_allclose(
            read(simulation, name, period), given, rtol=1e-5, atol=1e-2
        )


@hypothesis.settings(max_examples=300, **SETTINGS)
@hypothesis.given(
    first_inputs=st.lists(own_input, max_size=6),
    requests=st.lists(request, max_size=8),
    later_input=helper_input,
)
def test_a_helper_leaves_the_inputs_it_finds_unchanged(
    first_inputs, requests, later_input
):
    """An input already set for one of the variable's own periods is never
    changed by an input set over a longer period, and every sub-period of
    that longer period holds an input afterwards."""
    simulation = build_simulation(SYSTEM)
    for name, period, values in first_inputs:
        assert apply_input(simulation, name, period, values) is None
    for name, period in requests:
        read(simulation, name, period)
    before = {
        (name, period): read(simulation, name, period).copy()
        for name, period, _ in first_inputs
    }

    name, period, values = later_input
    apply_input(simulation, name, period, values)

    for key, value in before.items():
        assert same_arrays(read(simulation, *key), value), key
    recorded = input_record(simulation)
    for sub_period in sub_periods(name, period):
        assert (name, "default", sub_period) in recorded
