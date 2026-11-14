"""Property: ``subsample`` keeps every kept member's loaded role.

For any households of explicit role keys (bounded, unbounded and unmatched),
any person order, weights, default role, role source and sample, every kept
household is a whole source household whose role counts, by key and by
``nb_persons``, equal the source household's, whether or not the system
defines a role variable. An integer role variable either keeps the roles or
rejects members with no role before changing anything.
"""

import logging
from contextlib import contextmanager

import pytest

from tests.fixtures.subsample_roles import (
    DATASET_YEAR,
    FLATTENED_KEYS,
    ROLE,
    UNKNOWN_COLUMN_WARNING,
    assert_roles_kept,
    load,
    role_keys,
)

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

# ``parent`` is the template's compound key and ``lodger`` no key at all:
# both load as no role.
EXTRA_KEYS = ["child", "lodger", "parent"]


def household(first_parent, second_parent):
    parents = ["first_parent"] * first_parent + ["second_parent"] * second_parent
    return st.lists(
        st.sampled_from(EXTRA_KEYS), min_size=0 if parents else 1, max_size=3
    ).map(lambda others: parents + others)


HOUSEHOLD = st.tuples(st.booleans(), st.booleans()).flatmap(
    lambda parents: household(*parents)
)
HOUSEHOLDS = st.lists(HOUSEHOLD, min_size=1, max_size=6)
DEFAULT_ROLES = st.sampled_from(
    ["first_parent", "second_parent", "child", "parent", None]
)
SEEDS = st.sampled_from([0, 1, "a", "b"])


@contextmanager
def unknown_column_warnings():
    class Collect(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.messages = []

        def emit(self, record):
            if UNKNOWN_COLUMN_WARNING in record.getMessage():
                self.messages.append(record.getMessage())

    handler = Collect()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler.messages
    finally:
        root.removeHandler(handler)


def draw_layout(data, households):
    people = sum(len(household) for household in households)
    order = data.draw(st.permutations(range(people)), label="order")
    weights = data.draw(
        st.lists(st.integers(1, 9), min_size=len(households), max_size=len(households)),
        label="weights",
    )
    return order, [float(weight) for weight in weights]


@hypothesis.settings(max_examples=200, deadline=None)
@hypothesis.example(
    households=[["first_parent", "child", "child"], ["first_parent"]],
    role_variable="absent",
    role_column=ROLE,
    default_role="first_parent",
    n=2,
    seed=0,
    quantize_weights=True,
    again=1,
    data=None,
).via("review P3-1 on #595")
@hypothesis.given(
    households=HOUSEHOLDS,
    role_variable=st.sampled_from(["absent", "string"]),
    role_column=st.sampled_from([ROLE, "role"]),
    default_role=DEFAULT_ROLES,
    n=st.integers(1, 8),
    seed=SEEDS,
    quantize_weights=st.booleans(),
    again=st.integers(1, 4),
    data=st.data(),
)
def test_subsample_keeps_explicit_roles(
    households,
    role_variable,
    role_column,
    default_role,
    n,
    seed,
    quantize_weights,
    again,
    data,
):
    order, weights = (None, None) if data is None else draw_layout(data, households)
    with unknown_column_warnings() as warnings:
        simulation = load(
            households,
            role_variable=role_variable,
            role_column=role_column,
            default_role=default_role,
            order=order,
            weights=weights,
        )
        # The loader itself agrees with the table, so later checks test
        # what subsample keeps rather than what was loaded.
        assert_roles_kept(simulation, households)
        assert simulation.household.count == len(households)

        simulation.subsample(
            n=n, seed=seed, time_period=DATASET_YEAR, quantize_weights=quantize_weights
        )
        assert 1 <= simulation.household.count <= min(n, len(households))
        assert_roles_kept(simulation, households)

        # The rebuilt simulation carries them on through another subsample.
        simulation.subsample(n=again, seed=seed, time_period=DATASET_YEAR)
        assert_roles_kept(simulation, households)

    assert warnings == []


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    households=HOUSEHOLDS,
    default_role=DEFAULT_ROLES,
    n=st.integers(1, 8),
    seed=SEEDS,
    data=st.data(),
)
def test_integer_role_variable_keeps_roles_or_changes_nothing(
    households, default_role, n, seed, data
):
    order, weights = draw_layout(data, households)
    simulation = load(
        households,
        role_variable="integer",
        role_column="role",
        default_role=default_role,
        order=order,
        weights=weights,
    )
    roleless = any(
        key not in FLATTENED_KEYS for household in households for key in household
    )
    dataset = simulation.dataset
    persons = simulation.persons.ids.tolist()
    roles = role_keys(simulation).tolist()
    household_weights = simulation.calculate("household_weight", DATASET_YEAR).tolist()

    if roleless:
        with pytest.raises(ValueError, match="integer role variable"):
            simulation.subsample(n=n, seed=seed, time_period=DATASET_YEAR)
        assert simulation.dataset is dataset
        assert simulation.persons.ids.tolist() == persons
        assert role_keys(simulation).tolist() == roles
        assert (
            simulation.calculate("household_weight", DATASET_YEAR).tolist()
            == household_weights
        )
    else:
        simulation.subsample(n=n, seed=seed, time_period=DATASET_YEAR)
        assert_roles_kept(simulation, households)
