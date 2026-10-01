"""Branch results do not depend on what was calculated before.

Random sequences of calculations, branches, inputs and
``drop_computed_arrays`` run on a synthetic system, with values held in
memory, on disk, or not kept. Every calculation in a branch must equal the
same calculation in a new simulation given the branch's inputs, its own and
those it inherited when it was created, before calculating anything.
``test_branch_input_invalidation.py`` pins the same behaviour with examples.
"""

from __future__ import annotations

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.experimental import MemoryConfig
from tests.fixtures.branch_input_invalidation import (
    PEOPLE,
    ROOT_INPUTS,
    SYNTHETIC_SYSTEM,
    YEARS,
    synthetic_simulation,
)

SETTABLE = [
    "p_a",
    "p_b",
    "p_up",
    "p_c",
    "p_m",
    "p_sum",
    "p_prod",
    "p_switch",
    "p_inner_only",
]
CALCULABLE = [
    "p_sum",
    "p_prod",
    "p_cond",
    "p_lag",
    "p_months",
    "p_inner",
    "p_inner_only",
    "p_up",
    "p_c",
]
MONTHS = ["2013-01", "2013-07", "2014-12", "2015-03"]
MEMORY_CONFIGS = {
    "memory": lambda: None,
    "disk": lambda: MemoryConfig(max_memory_occupation=0),
    "not_kept": lambda: MemoryConfig(
        max_memory_occupation=1, variables_to_drop=["p_sum", "p_inner_only"]
    ),
}

# Indices pick a simulation modulo how many exist (-1: the newest); small ones
# keep most operations on the root and the first few branches, where they
# interact.
simulation_index = st.integers(0, 3)
values = st.tuples(*[st.integers(0, 100).map(float)] * PEOPLE)
calculate = st.one_of(
    st.tuples(
        st.just("calculate"),
        simulation_index,
        st.sampled_from(CALCULABLE),
        st.sampled_from(YEARS),
    ),
    st.tuples(
        st.just("calculate"),
        simulation_index,
        st.just("p_month"),
        st.sampled_from(MONTHS),
    ),
)
set_input = st.one_of(
    st.tuples(
        st.just("set"),
        simulation_index,
        st.sampled_from(SETTABLE),
        st.sampled_from(YEARS),
        values,
    ),
    st.tuples(
        st.just("set"),
        simulation_index,
        st.sampled_from(["p_m", "p_month"]),
        st.sampled_from(MONTHS),
        values,
    ),
)
single_operation = st.one_of(
    calculate,
    calculate,
    set_input,
    set_input,
    st.tuples(st.just("branch"), simulation_index),
    st.tuples(st.just("drop"), simulation_index),
)
# Pairs of a calculated value and an input it depends on, each reaching it a
# different way: through a formula's own branch, uprating from an earlier
# year, a month of a year, a value the holder may not keep, a lagged year,
# and a year summed from months.
DEPENDENCIES = [
    ("p_inner", "2013", "p_inner_only", "2013"),
    ("p_prod", "2015", "p_up", "2014"),
    ("p_month", "2013-07", "p_m", "2013"),
    ("p_prod", "2013", "p_sum", "2013"),
    ("p_lag", "2014", "p_a", "2013"),
    ("p_cond", "2014", "p_b", "2014"),
    ("p_months", "2014", "p_m", "2014-12"),
]
# Calculate a value, branch, override an input it depends on in the new
# branch (index -1), and calculate the value again there; or branch first and
# calculate the value in the branch before and after overriding the input.
override_after_calculating = st.tuples(
    simulation_index, st.sampled_from(DEPENDENCIES), values, st.booleans()
).map(
    lambda drawn: (
        [
            ("calculate", drawn[0], drawn[1][0], drawn[1][1]),
            ("branch", drawn[0]),
            ("set", -1, drawn[1][2], drawn[1][3], drawn[2]),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
        ]
        if drawn[3]
        else [
            ("branch", drawn[0]),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
            ("set", -1, drawn[1][2], drawn[1][3], drawn[2]),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
        ]
    )
)
operations = st.lists(
    st.one_of(
        single_operation.map(lambda operation: [operation]),
        override_after_calculating,
    ),
    min_size=1,
    max_size=15,
).map(lambda chunks: [operation for chunk in chunks for operation in chunk])


def _stored_periods(variable, period):
    """The periods ``set_input`` stores ``variable`` at, given ``period``."""
    period = periods.period(period)
    if SYNTHETIC_SYSTEM.get_variable(variable).definition_period == periods.MONTH:
        return [str(month) for month in period.get_subperiods(periods.MONTH)]
    return [str(period)]


def _own_inputs(simulation, variable, period):
    """The values stored on the branch itself for ``variable`` over ``period``."""
    holder = simulation.get_holder(variable)
    return {
        (variable, stored_period): tuple(
            holder._get_array_from_storage(stored_period, simulation.branch_name)
        )
        for stored_period in _stored_periods(variable, period)
    }


def _run(program, memory_config):
    """Run ``program``; return each calculation with the inputs it should reflect.

    A branch's inputs are those of the simulation it was created from, as
    they were then, and those set on it since, as stored (an input for a
    year is divided between the months the branch has not set itself).
    """
    root = synthetic_simulation(ROOT_INPUTS, memory_config=memory_config)
    simulations = [root]
    inputs = [dict(ROOT_INPUTS)]
    own_inputs = [set()]
    results = []
    for operation in program:
        kind, index = operation[0], operation[1] % len(simulations)
        if operation[1] == -1:
            index = len(simulations) - 1  # The newest simulation.
        simulation = simulations[index]
        if kind == "branch":
            simulations.append(simulation.get_branch(f"b{len(simulations)}"))
            inputs.append(dict(inputs[index]))
            own_inputs.append(set())
        elif kind == "set":
            if index == 0:
                continue  # Inputs on the root after calculating are out of scope.
            _, _, variable, period, value = operation
            stored = {(variable, p) for p in _stored_periods(variable, period)}
            if len(stored) > 1 and stored <= own_inputs[index]:
                # Dividing a year between months the branch has all set
                # itself is an error unless the totals match.
                continue
            simulation.set_input(variable, period, np.asarray(value))
            stored = _own_inputs(simulation, variable, period)
            inputs[index].update(stored)
            own_inputs[index].update(stored)
        elif kind == "calculate":
            _, _, variable, period = operation
            value = np.array(simulation.calculate(variable, period), copy=True)
            results.append((dict(inputs[index]), variable, period, value))
        else:
            simulation.drop_computed_arrays()
    return results


@hypothesis.settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(program=operations, mode=st.sampled_from(sorted(MEMORY_CONFIGS)))
def test_branch_calculations_match_a_simulation_given_its_inputs_first(program, mode):
    for branch_inputs, variable, period, value in _run(program, MEMORY_CONFIGS[mode]()):
        expected = synthetic_simulation(branch_inputs).calculate(variable, period)
        # Uprating chains may round differently in float32 depending on which
        # earlier periods were calculated first.
        np.testing.assert_allclose(
            value, expected, rtol=1e-5, err_msg=f"{variable} {period}"
        )
