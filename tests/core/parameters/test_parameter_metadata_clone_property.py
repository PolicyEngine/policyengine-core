"""Properties of parameter tree clones: each copy is a tree of its own.

For any chain of clones of a parameter tree (a clone of the source, or of
another clone) and any edits made along the way to any tree in the chain,
through ``Parameter.update``, a value assignment, a value removal, the
metadata dict's own methods, replacing nested metadata values, and Core's
``propagate_parameter_metadata`` and ``interpolate_parameters``:

* every tree reads as a tree built fresh from the same data and given the
  same edits (its own, and those of the trees it was cloned from before it
  was cloned), so no edit reaches a tree it was not made on;
* ``clone``, which shares nested metadata values, and ``deep_clone``, which
  deep-copies them as Core used to, leave every tree in the same state
  (differential).

A nested metadata value edited in place, rather than replaced, is visible
through every tree that shares it: that is the documented contract and is
not one of the edits here. ``test_parameter_metadata_clone.py`` has the
examples.
"""

import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tests.fixtures.parameter_metadata_clones import (  # noqa: E402
    apply_edit,
    build,
    deep_clone,
    snapshot,
)

# Positions are taken modulo the number of objects in the tree.
POSITIONS = st.integers(min_value=0, max_value=60)
KEYS = st.sampled_from(
    ["label", "unit", "reference", "notes", "added", "propagate_metadata_to_children"]
)
LEAVES = st.one_of(st.text(max_size=3), st.integers(-3, 3), st.booleans())
METADATA_VALUES = st.recursive(
    LEAVES,
    lambda inner: st.one_of(
        st.lists(inner, max_size=2),
        st.dictionaries(st.sampled_from(["title", "href", "x"]), inner, max_size=2),
    ),
    max_leaves=4,
)
NUMBERS = st.one_of(st.integers(0, 500), st.floats(0, 500, allow_nan=False))

EDITS = st.one_of(
    st.tuples(st.just("update"), POSITIONS, st.integers(2014, 2024), NUMBERS),
    st.tuples(st.just("set value"), POSITIONS, POSITIONS, NUMBERS),
    st.tuples(st.just("remove value"), POSITIONS, POSITIONS),
    st.tuples(st.just("set key"), POSITIONS, KEYS, METADATA_VALUES),
    st.tuples(
        st.just("update keys"),
        POSITIONS,
        st.dictionaries(KEYS, METADATA_VALUES, max_size=2),
    ),
    st.tuples(st.just("setdefault key"), POSITIONS, KEYS, METADATA_VALUES),
    st.tuples(st.just("pop key"), POSITIONS, KEYS),
    st.tuples(st.just("replace values"), POSITIONS),
    st.tuples(st.just("propagate")),
    st.tuples(st.just("interpolate")),
)


@st.composite
def chains(draw):
    """Steps on a growing list of trees, the first being the source.

    A step either clones a tree in the list (appending the clone) or edits
    one. Each step names its tree by position in the list.
    """
    steps = []
    count = 1
    for _ in range(draw(st.integers(min_value=1, max_value=8))):
        target = draw(st.integers(min_value=0, max_value=count - 1))
        if draw(st.booleans()):
            steps.append(("clone", target))
            count += 1
        else:
            steps.append(("edit", target, draw(EDITS)))
    return steps


def _run(steps, clone):
    """Run ``steps`` with ``clone``; return each tree and its edit history."""
    trees = [build()]
    histories = [[]]
    for step in steps:
        if step[0] == "clone":
            trees.append(clone(trees[step[1]]))
            histories.append(list(histories[step[1]]))
        else:
            apply_edit(trees[step[1]], step[2])
            histories[step[1]].append(step[2])
    return trees, histories


def _fresh(history):
    tree = build()
    for edit in history:
        apply_edit(tree, edit)
    return tree


_SETTINGS = dict(max_examples=300, deadline=None, derandomize=True)


@settings(**_SETTINGS)
@given(steps=chains())
def test_every_tree_in_a_clone_chain_reads_as_a_fresh_tree(steps):
    trees, histories = _run(steps, lambda tree: tree.clone())

    for tree, history in zip(trees, histories):
        assert snapshot(tree) == snapshot(_fresh(history))


@settings(**_SETTINGS)
@given(steps=chains())
def test_sharing_nested_metadata_matches_deep_copying_it(steps):
    shared, _ = _run(steps, lambda tree: tree.clone())
    deep, _ = _run(steps, deep_clone)

    assert [snapshot(tree) for tree in shared] == [snapshot(tree) for tree in deep]


@settings(**_SETTINGS)
@given(
    edits=st.lists(EDITS, min_size=1, max_size=6),
    edit_clone=st.booleans(),
)
def test_edits_to_one_side_never_change_the_other(edits, edit_clone):
    """The two-tree case, checked against the untouched side's own snapshot."""
    source = build()
    clone = source.clone()
    edited, other = (clone, source) if edit_clone else (source, clone)
    other_before = snapshot(other)

    for edit in edits:
        apply_edit(edited, edit)

    assert snapshot(other) == other_before
    assert snapshot(edited) == snapshot(_fresh(edits))
