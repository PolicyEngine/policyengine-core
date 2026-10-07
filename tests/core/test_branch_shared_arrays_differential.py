"""Randomized branch isolation checks for immutable cached arrays."""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from tests.fixtures.branch_shared_arrays import build_simulation

PERIODS = ("2017-01", "2017-02", "2018-06")
VARIABLES = ("salary", "rent", "income_tax", "disposable_income")

operation = st.one_of(
    st.tuples(
        st.just("branch"),
        st.integers(min_value=0, max_value=10**6),
        st.integers(min_value=0, max_value=10**6),
    ),
    st.tuples(
        st.just("calculate"),
        st.integers(min_value=0, max_value=10**6),
        st.sampled_from(VARIABLES),
        st.sampled_from(PERIODS),
    ),
    st.tuples(
        st.just("set"),
        st.integers(min_value=0, max_value=10**6),
        st.sampled_from(("salary", "rent")),
        st.sampled_from(PERIODS),
        st.integers(min_value=-1000, max_value=1000),
    ),
    st.tuples(
        st.just("delete"),
        st.integers(min_value=0, max_value=10**6),
        st.sampled_from(VARIABLES),
        st.none() | st.sampled_from(PERIODS),
    ),
)


def _snapshot(simulation):
    return {
        (name, key): entry.read().copy()
        for population in simulation.populations.values()
        for name, holder in population._holders.items()
        for key, entry in holder._memory_storage._arrays.items()
    }


def _assert_snapshot(simulation, expected) -> None:
    actual = _snapshot(simulation)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)


@hypothesis.settings(max_examples=300, deadline=None)
@hypothesis.given(operations=st.lists(operation, max_size=50))
def test_random_branch_operations_never_mutate_root_storage(
    tax_benefit_system,
    operations,
) -> None:
    root = build_simulation(tax_benefit_system)
    root.calculate("disposable_income", "2017-01")
    root_snapshot = _snapshot(root)
    branches = [root.get_branch("initial")]

    for sequence, item in enumerate(operations):
        kind = item[0]
        branch = branches[item[1] % len(branches)]
        if kind == "branch":
            branches.append(branch.get_branch(f"branch-{sequence}-{item[2]}"))
        elif kind == "calculate":
            _, _, variable, input_period = item
            value = branch.calculate(variable, input_period)
            assert not value.flags.writeable
        elif kind == "set":
            _, _, variable, input_period, seed = item
            size = 4 if variable == "salary" else 2
            branch.set_input(variable, input_period, np.full(size, seed, dtype=float))
        else:
            _, _, variable, input_period = item
            branch.delete_arrays(variable, input_period)

        _assert_snapshot(root, root_snapshot)
