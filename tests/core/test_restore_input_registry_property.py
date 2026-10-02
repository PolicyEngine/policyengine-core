"""Property: a restored simulation keeps and drops what the dumped one would.

For random inputs (including inputs replaced after calculating, inputs split
by a ``set_input`` helper, and ETERNITY inputs set for a year) and random
calculations before the dump:

1. **Round trip.** The restored simulation stores exactly the dumped
   simulation's values (default branch), byte for byte, and records the same
   storage keys as inputs.
2. **Same answers.** Calculating any sequence of requests on the restored
   simulation gives what the dumped simulation gives.
3. **Reforms keep the same values.** After ``apply_reform`` with a reform
   that changes nothing, the restored simulation and the dumped simulation
   store the same values and give the same answers to the same requests,
   byte for byte. So does a new simulation given only the inputs, except in
   the one case below.

The exception is an existing difference between the dumped simulation and a
new one, not a restore one: a ``set_input`` helper splitting an annual input
counts a month the simulation already calculated as already set, so the
split depends on what was calculated before the input (chip task_fd52e7d5).
The comparison with a new simulation leaves those examples out.

``test_restore_input_registry.py`` pins the same behaviour with examples.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.enums import EnumArray
from policyengine_core.tools.simulation_dumper import (
    dump_simulation,
    restore_simulation,
)
from tests.fixtures.uprated_inputs import (
    NoOp,
    build_simulation,
    build_system,
    input_values,
)

YEARS = ["2012", "2013", "2014", "2015"]
MONTHS = ["2013-01", "2013-02", "2014-06"]

INPUT_PERIODS = {
    "uprated_count": YEARS,
    "uprated_amount": YEARS,
    "masked_amount": YEARS,
    "eligible": YEARS,
    # Annual inputs go through the helper that splits them into months.
    "monthly_amount": MONTHS + ["2013"],
    "eternal_code": ["2013", "ETERNITY"],
}
REQUESTS = [
    ("uprated_count", YEARS),
    ("uprated_amount", YEARS),
    ("doubled_amount", YEARS),
    ("masked_amount", YEARS),
    ("eligible", YEARS),
    ("monthly_amount", MONTHS),
    ("monthly_doubled", MONTHS),
    ("eternal_code", ["2013"]),
    ("eternal_code_plus_one", ["2015"]),
]

_input = st.tuples(
    st.sampled_from(sorted(INPUT_PERIODS)),
    st.integers(0, 3),
    st.integers(0, 20),
).map(
    lambda draw: (
        draw[0],
        INPUT_PERIODS[draw[0]][draw[1] % len(INPUT_PERIODS[draw[0]])],
        input_values(draw[0], draw[2]),
    )
)
_request = st.tuples(st.integers(0, len(REQUESTS) - 1), st.integers(0, 3)).map(
    lambda draw: (
        REQUESTS[draw[0]][0],
        REQUESTS[draw[0]][1][draw[1] % len(REQUESTS[draw[0]][1])],
    )
)
# Before the dump: set an input or calculate, in any order.
_step = st.one_of(
    st.tuples(st.just("input"), _input),
    st.tuples(st.just("calculate"), _request),
)


@pytest.fixture(scope="module")
def system():
    return build_system()


def _result(function):
    try:
        value = function()
    except Exception as error:  # Compare failures as well as values.
        return ("error", type(error).__name__, str(error))
    if isinstance(value, EnumArray):
        value = value.view(np.ndarray)
    value = np.asarray(value)
    return ("array", value.dtype.str, value.shape, value.tobytes())


def _stored(simulation):
    """Every default-branch value, as bytes, keyed by variable and period."""
    stored = {}
    for population in simulation.populations.values():
        for name, holder in population._holders.items():
            for branch_name, period in holder.get_known_branch_periods():
                if branch_name != "default":
                    continue
                value = holder.get_array(period)
                stored[(name, str(period))] = (value.dtype.str, value.tobytes())
    return stored


def _split_after_calculating(steps):
    """Whether an annual input to the monthly variable follows a calculation."""
    calculated = False
    for kind, step in steps:
        if kind == "calculate":
            calculated = True
        elif calculated and step[0] == "monthly_amount" and step[1] == "2013":
            return True
    return False


def _input_keys(simulation):
    """Storage keys the input record names, as storage writes them."""
    keys = set()
    for name, branch_name, period in simulation._user_input_keys:
        if simulation.get_holder(name).variable.definition_period == periods.ETERNITY:
            period = periods.ETERNITY
        keys.add((name, branch_name, str(periods.period(period))))
    return keys


@hypothesis.settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[
        hypothesis.HealthCheck.too_slow,
        hypothesis.HealthCheck.data_too_large,
        hypothesis.HealthCheck.function_scoped_fixture,
    ],
)
@hypothesis.given(
    steps=st.lists(_step, max_size=12),
    requests=st.lists(_request, max_size=10),
)
def test_restored_simulation_keeps_and_drops_what_the_dumped_one_would(
    system, steps, requests
):
    dumped = build_simulation(system)
    inputs = []
    for kind, step in steps:
        if kind == "input":
            # Some inputs are refused (an annual input twice for a variable
            # split into months): the new simulation gets the same refusal.
            inputs.append(step)
            _result(lambda: dumped.set_input(*step))
        else:
            _result(lambda: dumped.calculate(*step))

    with tempfile.TemporaryDirectory(prefix="core-restore-") as directory:
        dump_simulation(dumped, directory)
        restored = restore_simulation(directory, system)

    # 1. Round trip.
    assert _stored(restored) == _stored(dumped)
    stored_keys = {(name, "default", period) for name, period in _stored(dumped).keys()}
    assert _input_keys(restored) == _input_keys(dumped) & stored_keys

    # 2. Same answers.
    for request in requests:
        assert _result(lambda: restored.calculate(*request)) == _result(
            lambda: dumped.calculate(*request)
        ), request

    # 3. Reforms keep the same values.
    fresh = build_simulation(system)
    for step in inputs:
        _result(lambda: fresh.set_input(*step))
    compare_with_fresh = not _split_after_calculating(steps)
    for simulation in (dumped, restored, fresh):
        simulation.apply_reform(NoOp)
    assert _stored(restored) == _stored(dumped)
    if compare_with_fresh:
        assert _stored(restored) == _stored(fresh)
    for request in requests:
        answers = {
            label: _result(lambda: simulation.calculate(*request))
            for label, simulation in (
                ("dumped", dumped),
                ("restored", restored),
                ("fresh", fresh),
            )
        }
        assert answers["restored"] == answers["dumped"], (request, answers)
        if compare_with_fresh:
            assert answers["restored"] == answers["fresh"], (request, answers)
