"""Property: a restored simulation keeps and drops what the dumped one would.

For random inputs (including inputs replaced after calculating, inputs split
by a ``set_input`` helper, ETERNITY inputs set for a year, and inputs set on
a branch of the dumped simulation), random calculations, values cached with
``put_in_cache`` but not marked derived, deletions and reforms before the
dump:

1. **Round trip.** The restored simulation stores exactly the dumped
   simulation's values (default branch), byte for byte, with the same
   derived marks, and records as inputs exactly the values not marked
   derived.
2. **Same answers.** Calculating any sequence of requests on the restored
   simulation gives what the dumped simulation gives.
3. **Reforms keep the same values.** When the dumped simulation's input
   record agrees with its derived marks (it records exactly the stored
   values not marked derived), the restored simulation records the same
   inputs, and after ``apply_reform`` with a reform that changes nothing,
   the restored simulation and the dumped simulation store the same values
   and give the same answers to the same requests, byte for byte. So does a
   new simulation given only the explicit inputs, when the sequence contains
   only input writes, calculations and branch input writes.

The record and the marks disagree only after a value is cached without
``derived=True`` (an input to carry-over that ``set_input`` did not record)
or after an input is deleted (``delete_arrays`` leaves its record, which
policyengine-core#561 drops). The restored simulation then follows the marks,
by property 1.

The new simulation is left out after caching, deletion or a reform, since
it is given only the explicit inputs. Period-splitting helpers ignore
calculated values when applying an input, as master requires.

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
# Before the dump: set an input, calculate, set an input on a branch, cache
# a value without marking it derived, delete a variable's values for a
# period, or apply a reform that changes nothing, in any order.
_step = st.one_of(
    st.tuples(st.just("input"), _input),
    st.tuples(st.just("calculate"), _request),
    st.tuples(st.just("branch_input"), _input),
    st.tuples(st.just("cache"), _input),
    st.tuples(st.just("delete"), _request),
    st.tuples(st.just("reform"), st.just(None)),
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


def _derived_marks(simulation):
    """Every default-branch value's derived mark, keyed as ``_stored``."""
    return {
        (name, str(period)): holder.is_derived(period)
        for population in simulation.populations.values()
        for name, holder in population._holders.items()
        for branch_name, period in holder.get_known_branch_periods()
        if branch_name == "default"
    }


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
        elif kind == "branch_input":
            # Stays on the branch: the dumped simulation's own values and
            # inputs are what a new simulation is given.
            _result(lambda: dumped.get_branch("reform").set_input(*step))
        elif kind == "cache":
            name, period, values = step
            holder = dumped.get_holder(name)
            _result(
                lambda: holder.put_in_cache(
                    holder._to_array(values), periods.period(period)
                )
            )
        elif kind == "delete":
            _result(lambda: dumped.delete_arrays(*step))
        elif kind == "reform":
            dumped.apply_reform(NoOp)
        else:
            _result(lambda: dumped.calculate(*step))

    stored_keys = {(name, "default", period) for name, period in _stored(dumped)}
    marked_inputs = {
        (name, "default", period)
        for (name, period), derived in _derived_marks(dumped).items()
        if not derived
    }
    records_agree = _input_keys(dumped) & stored_keys == marked_inputs

    with tempfile.TemporaryDirectory(prefix="core-restore-") as directory:
        dump_simulation(dumped, directory)
        restored = restore_simulation(directory, system)

    # 1. Round trip.
    assert _stored(restored) == _stored(dumped)
    assert _derived_marks(restored) == _derived_marks(dumped)
    assert _input_keys(restored) == marked_inputs
    if records_agree:
        assert _input_keys(restored) == _input_keys(dumped) & stored_keys

    # 2. Same answers.
    for request in requests:
        assert _result(lambda: restored.calculate(*request)) == _result(
            lambda: dumped.calculate(*request)
        ), request

    if not records_agree:
        return

    # 3. Reforms keep the same values.
    fresh = build_simulation(system)
    for step in inputs:
        _result(lambda: fresh.set_input(*step))
    compare_with_fresh = all(
        kind in ("input", "calculate", "branch_input") for kind, _ in steps
    )
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
