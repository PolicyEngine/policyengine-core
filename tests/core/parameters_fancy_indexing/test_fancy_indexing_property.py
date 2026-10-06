"""Property: vector parameter lookups agree row by row with scalar lookups.

For any parameter tree of depth 1 to 3 (1 to 3 children per node, numeric
leaves) and any list of 0 to 20 paths through it, looking the paths up one
level at a time with key arrays returns an array of one value per path, each
equal to the scalar lookup of that path. Zero paths give an empty array; that
case used to raise ``IndexError`` once a level held numeric leaves. A key that
names no child raises ``ParameterNotFoundError`` at any level, including a
one-element key after a selection of no rows. Examples are in
``test_fancy_indexing.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.parameters import (  # noqa: E402
    ParameterNode,
    ParameterNotFoundError,
)


@st.composite
def _tree_and_paths(draw):
    widths = draw(st.lists(st.integers(1, 3), min_size=1, max_size=3))
    leaves = list(itertools.product(*[range(width) for width in widths]))
    values = draw(
        st.lists(
            st.integers(-10_000, 10_000),
            min_size=len(leaves),
            max_size=len(leaves),
        )
    )
    data = {}
    for leaf, value in zip(leaves, values):
        node = data
        for depth, child in enumerate(leaf[:-1]):
            node = node.setdefault(f"k{depth}_{child}", {})
        node[f"k{len(leaf) - 1}_{leaf[-1]}"] = {"2020-01-01": value}
    paths = draw(st.lists(st.sampled_from(leaves), max_size=20))
    return widths, data, paths


@settings(max_examples=300, deadline=None)
@given(_tree_and_paths())
def test_vector_lookup_matches_scalar_lookups(case):
    widths, data, paths = case
    node = ParameterNode(data=data)("2020-01-01")

    result = node
    for depth in range(len(widths)):
        result = result[
            np.asarray([f"k{depth}_{path[depth]}" for path in paths], dtype=str)
        ]

    expected = []
    for path in paths:
        scalar = node
        for depth, child in enumerate(path):
            scalar = scalar[f"k{depth}_{child}"]
        expected.append(scalar)

    assert np.shape(result) == (len(paths),)
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=float))


@settings(max_examples=300, deadline=None)
@given(_tree_and_paths(), st.data())
def test_unknown_key_at_any_level_raises(case, data):
    # The paths select rows (possibly none); then one level is asked for a
    # key that names no child, among valid keys for the selected rows, or
    # alone when no row is selected.
    widths, data_tree, paths = case
    node = ParameterNode(data=data_tree)("2020-01-01")
    level = data.draw(st.integers(0, len(widths) - 1))

    result = node
    for depth in range(level):
        result = result[
            np.asarray([f"k{depth}_{path[depth]}" for path in paths], dtype=str)
        ]
    keys = [f"k{level}_{path[level]}" for path in paths]
    if keys:
        keys[data.draw(st.integers(0, len(keys) - 1))] = "unknown"
    else:
        # No rows selected: a one-element key broadcasts over them.
        keys = ["unknown"]

    with pytest.raises(ParameterNotFoundError):
        result[np.asarray(keys, dtype=str)]
