"""Branches that share their parent's arrays read what copied branches read.

Random sequences of operations run on a simulation whose branches share
arrays and on one whose branches are copies (the previous ``get_branch``),
and every read must agree. ``test_branch_shared_arrays.py`` pins the same
behaviour with examples.
"""

from __future__ import annotations

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.enums import EnumArray
from tests.fixtures.branch_shared_arrays import (
    build_simulation,
    shared_keys,
    stored_arrays,
)


def _deep_copy_branch(simulation, name, clone_system=False):
    """``get_branch`` as it was before branches shared arrays."""
    if name == simulation.branch_name:
        return simulation
    if name in simulation.branches:
        return simulation.branches[name]
    branch = simulation.clone(clone_tax_benefit_system=clone_system)
    simulation.branches[name] = branch
    branch.branch_name = name
    branch.parent_branch = simulation
    if simulation.trace:
        branch.trace = True
        branch.tracer = simulation.tracer
    return branch


PERSON_INPUTS = ["salary"]
HOUSEHOLD_INPUTS = ["rent", "accommodation_size"]
READ_VARIABLES = [
    ("salary", "month"),
    ("salary", "year"),
    ("age", "month"),
    ("basic_income", "month"),
    ("income_tax", "month"),
    ("social_security_contribution", "month"),
    ("pension", "month"),
    ("disposable_income", "month"),
    ("rent", "month"),
    ("accommodation_size", "month"),
    ("housing_occupancy_status", "month"),
    ("housing_allowance", "month"),
    ("parenting_allowance", "month"),
    ("household_income", "month"),
    ("housing_tax", "year"),
]
FLOAT_VARIABLES = [
    ("salary", "month"),
    ("basic_income", "month"),
    ("income_tax", "month"),
    ("disposable_income", "month"),
    ("rent", "month"),
    ("housing_tax", "year"),
]
MONTHS = ["2017-01", "2017-02", "2018-01"]
YEARS = ["2017", "2018"]
BRANCH_NAMES = ["a", "b", "c"]

_operation = st.one_of(
    st.tuples(
        st.just("set_input"),
        st.integers(0, 7),
        st.sampled_from(PERSON_INPUTS + HOUSEHOLD_INPUTS),
        st.sampled_from(MONTHS + YEARS),
        st.integers(0, 5_000),
    ),
    st.tuples(
        st.just("read"),
        st.integers(0, 7),
        st.sampled_from(READ_VARIABLES),
        st.integers(0, 1),
        st.booleans(),
    ),
    st.tuples(
        st.just("write_in_place"),
        st.integers(0, 7),
        st.sampled_from(FLOAT_VARIABLES),
        st.integers(0, 1),
        st.booleans(),
    ),
    st.tuples(
        st.just("delete"),
        st.integers(0, 7),
        st.sampled_from([name for name, _ in READ_VARIABLES]),
        st.sampled_from(MONTHS + YEARS + [None]),
    ),
    st.tuples(
        st.just("branch"),
        st.integers(0, 7),
        st.sampled_from(BRANCH_NAMES),
        st.booleans(),
    ),
    st.tuples(st.just("drop"), st.integers(0, 7)),
    st.tuples(st.just("invalidate"), st.integers(0, 7)),
)


class _Tree:
    """A root simulation and the branches made from it, in creation order."""

    def __init__(self, tax_benefit_system, make_branch):
        self.nodes = [build_simulation(tax_benefit_system)]
        self.make_branch = make_branch
        # Simulations a branch has been made from. Writing in place into one
        # of these is the documented difference, so the test leaves it out.
        self.branched_from = []

    def node(self, index):
        return self.nodes[index % len(self.nodes)]


def _result(function):
    try:
        value = function()
    except Exception as error:  # Compare failures as well as values.
        return ("error", type(error).__name__, str(error))
    if value is None:
        return ("none",)
    if isinstance(value, EnumArray):
        return ("enum", value.possible_values, np.asarray(value.view(np.ndarray)))
    return ("array", np.asarray(value))


def _same_bytes(left, right):
    """Same dtype, shape and bytes (values, for object arrays)."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype.kind == "O":
        return left.tolist() == right.tolist()
    return left.tobytes() == right.tobytes()


def _same(left, right):
    if left[0] != right[0]:
        return False
    if left[0] == "array":
        return _same_bytes(left[1], right[1])
    if left[0] == "enum":
        return left[1] is right[1] and _same_bytes(left[2], right[2])
    return left == right


def _apply(tree, operation):
    kind, index = operation[0], operation[1]
    simulation = tree.node(index)
    if kind == "set_input":
        _, _, variable, period, seed = operation
        count = (
            simulation.persons.count
            if variable in PERSON_INPUTS
            else simulation.household.count
        )
        values = (np.arange(count, dtype=float) + 1) * seed
        return _result(lambda: simulation.set_input(variable, period, values))
    if kind in ("read", "write_in_place"):
        _, _, (variable, unit), period_index, get_only = operation
        period = (MONTHS if unit == "month" else YEARS)[period_index]
        if get_only:
            result = _result(lambda: simulation.get_array(variable, period))
        else:
            result = _result(lambda: simulation.calculate(variable, period))
        if (
            kind == "write_in_place"
            and result[0] == "array"
            and all(simulation is not parent for parent in tree.branched_from)
        ):
            # What a formula does when it writes into a value it read.
            value = (
                simulation.get_array(variable, period)
                if get_only
                else simulation.calculate(variable, period)
            )
            value += 1
            value[0] = -7
            return _result(lambda: simulation.get_array(variable, period))
        return result
    if kind == "delete":
        _, _, variable, period = operation
        return _result(lambda: simulation.delete_arrays(variable, period))
    if kind == "branch":
        _, _, name, clone_system = operation
        branch = tree.make_branch(simulation, name, clone_system)
        if branch is not simulation and all(branch is not n for n in tree.nodes):
            tree.nodes.append(branch)
            tree.branched_from.append(simulation)
        return ("branch", [n is branch for n in tree.nodes].index(True))
    if kind == "drop":
        parent = simulation.parent_branch
        if parent is None:
            return ("root",)
        del parent.branches[simulation.branch_name]
        for population in simulation.populations.values():
            for holder in population._holders.values():
                holder._memory_storage._arrays.clear()
        tree.nodes = [n for n in tree.nodes if n is not simulation]
        return ("dropped",)
    if kind == "invalidate":
        simulation._invalidate_all_caches()
        return ("invalidated",)
    raise AssertionError(kind)


def _assert_arrays_shared_with_the_parent_are_views(tree):
    """A branch can only write into an array once it has copied it."""
    for simulation in tree.nodes[1:]:
        parent_arrays = list(stored_arrays(simulation.parent_branch).values())
        shared = shared_keys(simulation)
        for key, array in stored_arrays(simulation).items():
            if any(np.shares_memory(array, other) for other in parent_arrays):
                assert key in shared, key
                assert not array.flags.writeable, key


@hypothesis.settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[
        hypothesis.HealthCheck.too_slow,
        hypothesis.HealthCheck.data_too_large,
    ],
)
@hypothesis.given(operations=st.lists(_operation, max_size=40))
def test_shared_array_branches_match_copied_branches(tax_benefit_system, operations):
    shared = _Tree(
        tax_benefit_system,
        lambda simulation, name, clone_system: simulation.get_branch(
            name, clone_system
        ),
    )
    copied = _Tree(tax_benefit_system, _deep_copy_branch)

    for step, operation in enumerate(operations):
        assert _same(_apply(shared, operation), _apply(copied, operation)), (
            step,
            operation,
        )
        assert len(shared.nodes) == len(copied.nodes)
        _assert_arrays_shared_with_the_parent_are_views(shared)

    # Every simulation in both trees reads the same value for everything.
    for shared_simulation, copied_simulation in zip(shared.nodes, copied.nodes):
        for variable, unit in READ_VARIABLES:
            for period in MONTHS if unit == "month" else YEARS:
                assert _same(
                    _result(lambda: shared_simulation.calculate(variable, period)),
                    _result(lambda: copied_simulation.calculate(variable, period)),
                ), (shared_simulation.branch_name, variable, period)
