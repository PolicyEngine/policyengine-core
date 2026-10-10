"""Branch results do not depend on what was calculated before.

Random sequences of calculations, branches (including reused and forgotten
names), inputs (some setting again the value already read),
``drop_computed_arrays`` and dumps restored as new simulations run on a
synthetic system, with
values held in memory, on disk, not kept, or blacklisted. Every calculation
in a branch must equal the same calculation in a new simulation given the
branch's inputs, its own and those it inherited when it was created, before
calculating anything. The inputs a branch should hold are modelled here, not
read back from the branch. ``test_branch_input_invalidation.py`` pins the
same behaviour with examples.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.experimental import MemoryConfig
from policyengine_core.tools.simulation_dumper import (
    dump_simulation,
    restore_simulation,
)
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
    "p_imported_up",
    "p_up",
    "p_c",
]
MONTHS = ["2013-01", "2013-07", "2014-12", "2015-03"]
# How the root simulation (and so every branch) stores values.
MODES = {
    "memory": dict(),
    "disk": dict(memory_config=lambda: MemoryConfig(max_memory_occupation=0)),
    "not_kept": dict(
        memory_config=lambda: MemoryConfig(
            max_memory_occupation=1, variables_to_drop=["p_sum", "p_inner_only"]
        )
    ),
    # The synthetic system blacklists p_sum and p_inner_only.
    "blacklist": dict(opt_out_cache=True),
}

# Indices pick a simulation modulo how many exist (-1: the newest); small ones
# keep most operations on the root and the first few branches, where they
# interact. A branch is named "x" or "y" (so two lineages can share a name,
# and a forgotten name can be reused) or given a name of its own (None).
simulation_index = st.integers(0, 3)
branch_name = st.sampled_from(["x", "y", None, None])
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
    st.tuples(st.just("branch"), simulation_index, branch_name),
    st.tuples(st.just("forget"), simulation_index),
    st.tuples(st.just("drop"), simulation_index),
    # Set again, on a branch, an input it reads, with the value it reads.
    st.tuples(st.just("reset"), simulation_index, st.integers(0, 50)),
    # Dump a simulation (a branch, say) and restore it as a new one.
    st.tuples(st.just("dump"), simulation_index),
)
# Pairs of a calculated value and an input it depends on, each reaching it a
# different way: through a formula's own branch, uprating from an earlier
# year (directly, and through a formula's own branch), a month of a year, a
# value the holder may not keep, a lagged year, and a year summed from months.
DEPENDENCIES = [
    ("p_inner", "2013", "p_inner_only", "2013"),
    ("p_prod", "2015", "p_up", "2014"),
    ("p_imported_up", "2015", "p_up", "2014"),
    ("p_month", "2013-07", "p_m", "2013"),
    ("p_month", "2015-03", "p_m", "2015"),
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
            ("branch", drawn[0], None),
            ("set", -1, drawn[1][2], drawn[1][3], drawn[2]),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
        ]
        if drawn[3]
        else [
            ("branch", drawn[0], None),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
            ("set", -1, drawn[1][2], drawn[1][3], drawn[2]),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
        ]
    )
)
# Calculate the value in a new branch, set again there the input it depends
# on, with the value the branch reads, and calculate the value again.
reset_after_calculating = st.tuples(
    simulation_index, st.sampled_from(DEPENDENCIES)
).map(
    lambda drawn: [
        ("branch", drawn[0], None),
        ("calculate", -1, drawn[1][0], drawn[1][1]),
        ("reset", -1, (drawn[1][2], drawn[1][3])),
        ("calculate", -1, drawn[1][0], drawn[1][1]),
    ]
)
# The same through a dump: calculate the value, dump that simulation, restore
# it, branch from the restored one, override the input and calculate again.
override_after_restoring = st.tuples(
    simulation_index, st.sampled_from(DEPENDENCIES), values
).map(
    lambda drawn: [
        ("calculate", drawn[0], drawn[1][0], drawn[1][1]),
        ("dump", drawn[0]),
        ("branch", -1, None),
        ("set", -1, drawn[1][2], drawn[1][3], drawn[2]),
        ("calculate", -1, drawn[1][0], drawn[1][1]),
    ]
)
operations = st.lists(
    st.one_of(
        single_operation.map(lambda operation: [operation]),
        override_after_calculating,
        reset_after_calculating,
        override_after_restoring,
    ),
    min_size=1,
    max_size=15,
).map(lambda chunks: [operation for chunk in chunks for operation in chunk])


def _stored_inputs(own_inputs, variable, period, value):
    """What ``set_input(variable, period, value)`` stores on a branch, by its own rule.

    Monthly ``p_m`` divides a yearly value between the months the branch has
    not set as inputs itself (what is left after the months it has set).
    Returns ``None`` when every month is already set (an error unless the
    totals match, so the program skips it).
    """
    period = periods.period(period)
    variable_period = SYNTHETIC_SYSTEM.get_variable(variable).definition_period
    if variable_period != periods.MONTH or period.unit == periods.MONTH:
        return {(variable, str(period)): tuple(value)}
    months = [str(month) for month in period.get_subperiods(periods.MONTH)]
    unset = [month for month in months if (variable, month) not in own_inputs]
    if not unset:
        return None
    remaining = np.asarray(value, dtype=float) - sum(
        (
            np.asarray(own_inputs[(variable, month)])
            for month in months
            if month not in unset
        ),
        np.zeros(PEOPLE),
    )
    return {(variable, month): tuple(remaining / len(unset)) for month in unset}


def _run(program, mode):
    """Run ``program``; return each calculation with the inputs it should reflect.

    A branch's inputs are those of the simulation it was created from, as
    they were then, and those set on it since. A simulation restored from a
    dump has the inputs the dumped one had.
    """
    options = MODES[mode]
    root = synthetic_simulation(
        ROOT_INPUTS,
        memory_config=options.get("memory_config", lambda: None)(),
        opt_out_cache=options.get("opt_out_cache", False),
    )
    simulations = [root]
    inputs = [dict(ROOT_INPUTS)]
    own_inputs = [{}]
    children = {}  # (parent index, name) -> index, while the parent keeps it
    roots = {0}  # the root and simulations restored from dumps
    results = []
    for operation in program:
        kind, index = operation[0], operation[1] % len(simulations)
        if operation[1] == -1:
            index = len(simulations) - 1  # The newest simulation.
        simulation = simulations[index]
        if kind == "branch":
            name = operation[2] or f"b{len(simulations)}"
            if (index, name) in children or name == simulation.branch_name:
                continue  # get_branch would return an existing simulation.
            simulations.append(simulation.get_branch(name))
            inputs.append(dict(inputs[index]))
            own_inputs.append({})
            children[(index, name)] = len(simulations) - 1
        elif kind == "forget":
            # The parent forgets a branch (formulas delete theirs); the branch
            # object keeps working, and its name can be reused.
            names = sorted(name for parent, name in children if parent == index)
            if names:
                del simulation.branches[names[0]]
                del children[(index, names[0])]
        elif kind == "set":
            if index in roots:
                continue  # Inputs on a root after calculating are out of scope.
            _, _, variable, period, value = operation
            stored = _stored_inputs(own_inputs[index], variable, period, value)
            if stored is None:
                continue
            simulation.set_input(variable, period, np.asarray(value))
            inputs[index].update(stored)
            own_inputs[index].update(stored)
        elif kind == "reset":
            if index in roots:
                continue
            # Only inputs stored for the variable's own period unit: a helper
            # spreading a year over months stores something else.
            keys = sorted(
                key
                for key in inputs[index]
                if periods.period(key[1]).unit
                == SYNTHETIC_SYSTEM.get_variable(key[0]).definition_period
            )
            selector = operation[2]
            if isinstance(selector, tuple):
                keys = [selector] if selector in keys else []
            if not keys:
                continue
            key = keys[selector % len(keys)] if isinstance(selector, int) else keys[0]
            simulation.set_input(key[0], key[1], np.asarray(inputs[index][key]))
            own_inputs[index][key] = inputs[index][key]
        elif kind == "calculate":
            _, _, variable, period = operation
            value = np.array(simulation.calculate(variable, period), copy=True)
            results.append((dict(inputs[index]), variable, period, value))
        elif kind == "dump":
            directory = tempfile.mkdtemp(prefix="policyengine-branch-dump-")
            try:
                dump_simulation(simulation, directory)
                restored = restore_simulation(directory, SYNTHETIC_SYSTEM)
            finally:
                shutil.rmtree(directory, ignore_errors=True)
            simulations.append(restored)
            inputs.append(dict(inputs[index]))
            own_inputs.append({})
            roots.add(len(simulations) - 1)
        else:
            simulation.drop_computed_arrays()
    return results


# CI runs a fixed set of programs; set POLICYENGINE_BRANCH_PROPERTY_EXAMPLES
# to explore new ones (e.g. 5000 for a soak).
_SOAK_EXAMPLES = os.environ.get("POLICYENGINE_BRANCH_PROPERTY_EXAMPLES")


@hypothesis.settings(
    max_examples=int(_SOAK_EXAMPLES or 200),
    derandomize=not _SOAK_EXAMPLES,
    database=None,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(program=operations, mode=st.sampled_from(sorted(MODES)))
def test_branch_calculations_match_a_simulation_given_its_inputs_first(program, mode):
    for branch_inputs, variable, period, value in _run(program, mode):
        expected = synthetic_simulation(branch_inputs).calculate(variable, period)
        # Uprating chains, and months divided from a year, may round
        # differently in float32 depending on what was calculated first.
        np.testing.assert_allclose(
            value, expected, rtol=1e-5, atol=1e-4, err_msg=f"{variable} {period}"
        )
