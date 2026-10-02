"""Properties of the fast cache under holder writes and deletes.

Random sequences of operations run on a simulation and the branches made
from it: ``calculate``; ``set_input``, ``put_in_cache`` and ``delete_arrays``
on a holder, under the simulation's own branch name, ``default``, or a
branch it does not read; ``Simulation.set_input`` and
``Simulation.delete_arrays``; and creating a branch.

1. **The fast cache never changes what ``calculate`` returns.** A second
   tree runs the same operations but empties every fast cache before each
   ``calculate``. Every result agrees (bytes, or error), and so does every
   stored value at the end.
2. **A holder write or delete drops only what it changes.** The entries it
   removes all belong to its own simulation and variable, were stored under
   a branch that simulation reads, and are for the period written (any
   period, for an ETERNITY variable) or for a period the deleted one
   contains. Every other simulation's fast cache is untouched.

``test_holder_write_fast_cache.py`` pins the same behaviour with examples.
"""

from __future__ import annotations

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.enums import EnumArray
from tests.fixtures.uprated_inputs import (
    INPUT_VALUE_TYPES,
    PERSON_COUNT,
    build_simulation,
    build_system,
)

YEARS = ["2012", "2013", "2014"]
MONTHS = ["2013-01", "2013-02", "2014-06"]
VARIABLES = {
    # name: (periods to calculate, periods to write or delete)
    "uprated_count": (YEARS, YEARS),
    "uprated_amount": (YEARS, YEARS),
    "doubled_amount": (YEARS, YEARS),
    "masked_amount": (YEARS, YEARS),
    "eligible": (YEARS, YEARS),
    "monthly_amount": (MONTHS, MONTHS + ["2013"]),
    "monthly_doubled": (MONTHS, MONTHS + ["2013"]),
    "eternal_code": (YEARS, YEARS + ["ETERNITY"]),
    "eternal_code_plus_one": (YEARS, YEARS + ["ETERNITY"]),
}
NAMES = sorted(VARIABLES)
BRANCH_NAMES = ["a", "b"]
# Where a holder operation stores or deletes: the simulation's own branch
# name, the default one, or one the simulation does not read.
TARGETS = ["own", "default", "unread"]

_node = st.integers(0, 5)
_name = st.sampled_from(NAMES)
_index = st.integers(0, 3)
_seed = st.integers(0, 9)
_operation = st.one_of(
    st.tuples(st.just("calculate"), _node, _name, _index),
    st.tuples(st.just("calculate"), _node, _name, _index),
    st.tuples(
        st.just("holder_set_input"),
        _node,
        _name,
        _index,
        st.sampled_from(TARGETS),
        _seed,
    ),
    st.tuples(
        st.just("holder_put_in_cache"),
        _node,
        _name,
        _index,
        st.sampled_from(TARGETS),
        _seed,
    ),
    st.tuples(
        st.just("holder_delete"),
        _node,
        _name,
        st.one_of(st.none(), _index),
        st.sampled_from(TARGETS),
    ),
    st.tuples(st.just("set_input"), _node, _name, _index, _seed),
    st.tuples(st.just("delete_arrays"), _node, _name, st.one_of(st.none(), _index)),
    st.tuples(st.just("branch"), _node, st.sampled_from(BRANCH_NAMES)),
)


@pytest.fixture(scope="module")
def system():
    return build_system()


def _values(name, seed):
    value_type = INPUT_VALUE_TYPES.get(name, float)
    if value_type is bool:
        return np.array([seed % 2 == 1, seed % 3 != 0])
    base = np.arange(PERSON_COUNT) + 1
    if value_type is int:
        return (base * (seed + 3)).astype(np.int32)
    return (base * (seed + 0.5) * 12).astype(np.float32)


def _result(function):
    try:
        value = function()
    except Exception as error:  # Compare failures as well as values.
        return ("error", type(error).__name__, str(error))
    if value is None:
        return ("none",)
    if isinstance(value, EnumArray):
        value = value.view(np.ndarray)
    value = np.asarray(value)
    return ("array", value.dtype.str, value.shape, value.tobytes())


class _Tree:
    """A root simulation and the branches made from it, in creation order."""

    def __init__(self, system, empty_fast_caches):
        self.nodes = [build_simulation(system)]
        self.empty_fast_caches = empty_fast_caches

    def node(self, index):
        return self.nodes[index % len(self.nodes)]

    def fast_cache_keys(self):
        return [set(simulation._fast_cache) for simulation in self.nodes]


def _target_branch(simulation, target):
    if target == "own":
        return simulation.branch_name
    if target == "default":
        return "default"
    return "unread"


def _apply(tree, operation):
    """Run ``operation``; return its result and what it may drop.

    The second value is ``None``, or ``(node index, predicate)`` for a holder
    write or delete: the predicate says which of that simulation's
    fast-cache keys the operation is allowed to remove.
    """
    kind = operation[0]
    simulation = tree.node(operation[1])
    node_index = tree.nodes.index(simulation)
    name = operation[2] if len(operation) > 2 else None
    if kind == "branch":
        branch = simulation.get_branch(name)
        if all(branch is not node for node in tree.nodes):
            tree.nodes.append(branch)
        return ("branch", tree.nodes.index(branch)), None

    calculate_periods, write_periods = VARIABLES[name]
    variable = simulation.tax_benefit_system.get_variable(name)
    eternal = variable.definition_period == periods.ETERNITY
    if kind == "calculate":
        period = calculate_periods[operation[3] % len(calculate_periods)]
        if tree.empty_fast_caches:
            for node in tree.nodes:
                node._fast_cache.clear()
        return _result(lambda: simulation.calculate(name, period)), None

    period = (
        None
        if operation[3] is None
        else periods.period(write_periods[operation[3] % len(write_periods)])
    )
    if kind == "set_input":
        values = _values(name, operation[4])
        return _result(lambda: simulation.set_input(name, period, values)), None
    if kind == "delete_arrays":
        return _result(lambda: simulation.delete_arrays(name, period)), None

    holder = simulation.get_holder(name)
    branch_name = _target_branch(simulation, operation[4])
    read = branch_name in simulation._get_visible_branch_names()
    if kind == "holder_delete":

        def may_drop(key):
            return (
                read
                and key[0] == name
                and (period is None or eternal or period.contains(key[1]))
            )

        return (
            _result(lambda: holder.delete_arrays(period, branch_name)),
            (node_index, may_drop),
        )

    values = _values(name, operation[5])
    # A ``set_input`` helper stores every month of an annual input.
    written = (
        period.get_subperiods(variable.definition_period)
        if kind == "holder_set_input"
        and not eternal
        and period.unit != variable.definition_period
        else [period]
    )

    def may_drop(key):
        return read and key[0] == name and (eternal or key[1] in written)

    if kind == "holder_set_input":
        result = _result(lambda: holder.set_input(period, values, branch_name))
    else:
        result = _result(lambda: holder.put_in_cache(values, period, branch_name))
    return result, (node_index, may_drop)


def _stored(tree):
    stored = []
    for simulation in tree.nodes:
        values = {}
        for population in simulation.populations.values():
            for name, holder in population._holders.items():
                for branch_name, period in holder.get_known_branch_periods():
                    value = holder._memory_storage.get(period, branch_name)
                    values[(name, branch_name, str(period))] = (
                        value.dtype.str,
                        value.tobytes(),
                    )
        stored.append(values)
    return stored


@hypothesis.settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[
        hypothesis.HealthCheck.too_slow,
        hypothesis.HealthCheck.data_too_large,
        hypothesis.HealthCheck.function_scoped_fixture,
    ],
)
@hypothesis.given(operations=st.lists(_operation, max_size=40))
def test_fast_cache_is_transparent_and_drops_only_what_changes(system, operations):
    cached = _Tree(system, empty_fast_caches=False)
    uncached = _Tree(system, empty_fast_caches=True)

    for step, operation in enumerate(operations):
        before = cached.fast_cache_keys()
        result, drop = _apply(cached, operation)
        after = cached.fast_cache_keys()

        # 1. The fast cache never changes what ``calculate`` returns.
        assert result == _apply(uncached, operation)[0], (step, operation)

        # 2. A holder write or delete drops only what it changes.
        if drop is not None:
            node_index, may_drop = drop
            for index, (keys_before, keys_after) in enumerate(zip(before, after)):
                removed = keys_before - keys_after
                if index != node_index:
                    assert not removed, (step, operation, index, removed)
                for key in removed:
                    assert may_drop(key), (step, operation, key)

    assert _stored(cached) == _stored(uncached)
    for index, simulation in enumerate(cached.nodes):
        twin = uncached.nodes[index]
        twin._fast_cache.clear()
        for name, (calculate_periods, _) in VARIABLES.items():
            for period in calculate_periods:
                assert _result(lambda: simulation.calculate(name, period)) == _result(
                    lambda: twin.calculate(name, period)
                ), (index, name, period)
                twin._fast_cache.clear()
