"""Property: restores preserve values and reforms keep their own leaf inputs.

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
3. **Reforms keep the expected inputs.** Successful leaf writes made by
   explicit input operations, and not subsequently overwritten by cache or
   calculation writes or deleted, are exactly what the dumped simulation
   keeps after a reform that changes nothing. The restored simulation keeps
   exactly the dumped values not marked derived: restoring promotes these
   values to new explicit inputs. These two independent references are
   checked even when they differ. When they agree, the simulations also
   store the same values and give the same answers after reform, byte for
   byte. So does a new simulation given only the explicit inputs, when the
   sequence contains only input writes, calculations and branch input writes.

An external cache write is not a supplied input, even when it replaces a
supplied value with identical bytes. Its non-derived mark makes it available
to carry-over and uprating, and a dump restores it as a new recorded input.
The leaf-write reference observes storage writes by operation kind; it does
not consult the input registry or the supplied-input tier metadata.

The new simulation is left out after caching, deletion or a reform, since
it is given only the explicit inputs. Period-splitting helpers ignore
calculated values when applying an input, as master requires.

``test_restore_input_registry.py`` pins the same behaviour with examples.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage
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
@hypothesis.example(
    steps=[
        ("input", ("eligible", "2012", [False, False])),
        ("cache", ("eligible", "2012", [False, False])),
    ],
    requests=[],
)
@hypothesis.example(
    steps=[("input", ("eligible", "2012", [False, False]))],
    requests=[],
)
def test_restored_simulation_keeps_and_drops_what_the_dumped_one_would(
    system, steps, requests
):
    dumped = build_simulation(system)
    inputs = []
    supplied = {}
    kind = None
    original_put = InMemoryStorage.put

    def observe_put(
        storage,
        value,
        period,
        branch_name="default",
        derived=False,
        sequence_number=None,
    ):
        result = original_put(
            storage,
            value,
            period,
            branch_name,
            derived=derived,
            sequence_number=sequence_number,
        )
        if branch_name != "default" or kind not in ("input", "cache", "calculate"):
            return result
        # Observe successful writes to this simulation's leaf storage, not
        # the registry/provenance helpers that cache invalidation uses.
        for population in dumped.populations.values():
            for name, holder in population._holders.items():
                if holder._memory_storage is storage:
                    stored_period = periods.ETERNITY if storage.is_eternal else period
                    key = (name, str(periods.period(stored_period)))
                    if kind == "input" and not derived:
                        supplied[key] = (value.dtype.str, value.tobytes())
                    else:
                        supplied.pop(key, None)
                    return result
        return result

    with patch.object(InMemoryStorage, "put", observe_put):
        for kind, step in steps:
            if kind == "input":
                # Observe individual helper writes, including partial writes
                # before a refused input, without claiming skipped slots.
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
                remaining = _stored(dumped)
                supplied = {
                    key: value for key, value in supplied.items() if key in remaining
                }
            elif kind == "reform":
                dumped.apply_reform(NoOp)
                assert _stored(dumped) == supplied
            else:
                _result(lambda: dumped.calculate(*step))

    dumped_values = _stored(dumped)
    stored_keys = {(name, "default", period) for name, period in dumped_values}
    derived_marks = _derived_marks(dumped)
    restored_inputs = {
        key: value for key, value in dumped_values.items() if not derived_marks[key]
    }
    marked_inputs = {(name, "default", period) for name, period in restored_inputs}
    references_agree = supplied == restored_inputs

    with tempfile.TemporaryDirectory(prefix="core-restore-") as directory:
        dump_simulation(dumped, directory)
        restored = restore_simulation(directory, system)

    # 1. Round trip.
    assert _stored(restored) == _stored(dumped)
    assert _derived_marks(restored) == _derived_marks(dumped)
    assert _input_keys(restored) == marked_inputs
    if references_agree:
        assert _input_keys(restored) == _input_keys(dumped) & stored_keys

    # 2. Same answers.
    for request in requests:
        assert _result(lambda: restored.calculate(*request)) == _result(
            lambda: dumped.calculate(*request)
        ), request

    # 3. Every case checks both reform contracts, including cache overwrites.
    fresh = build_simulation(system)
    for step in inputs:
        _result(lambda: fresh.set_input(*step))
    compare_with_fresh = all(
        kind in ("input", "calculate", "branch_input") for kind, _ in steps
    )
    # A separate fresh simulation receives the serialized leaf snapshot;
    # no restore/provenance helper defines this reference's input set.
    snapshot_reference = build_simulation(system)
    for (name, period), (dtype, data) in restored_inputs.items():
        values = np.frombuffer(data, dtype=np.dtype(dtype)).copy()
        snapshot_reference.get_holder(name)._set(
            periods.period(period), values, is_input=True
        )
    for simulation in (dumped, restored, fresh, snapshot_reference):
        simulation.apply_reform(NoOp)
    assert _stored(dumped) == supplied
    assert _stored(restored) == restored_inputs
    assert _input_keys(dumped) == {
        (name, "default", period) for name, period in supplied
    }
    assert _input_keys(restored) == marked_inputs
    assert _stored(restored) == _stored(snapshot_reference)
    if references_agree:
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
                ("snapshot", snapshot_reference),
            )
        }
        assert answers["restored"] == answers["snapshot"], (request, answers)
        if references_agree:
            assert answers["restored"] == answers["dumped"], (request, answers)
        if compare_with_fresh:
            assert answers["restored"] == answers["fresh"], (request, answers)
