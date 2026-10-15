"""Reads of a lazily resolved ``ParameterNodeAtInstant`` in any order.

For any instant and any sequence of reads (attribute, item, iteration,
``_children``, ``dir``, ``repr``, vectorial indexing, names that do not
exist), every read returns what the up-front build held, and the node read in
full afterwards equals that build. ``test_parameter_node_at_instant_lazy.py``
pins the same behaviour with examples.
"""

from __future__ import annotations

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import ParameterNodeAtInstant
from tests.fixtures.parameter_nodes_at_instant import (
    INSTANTS,
    build_tree,
    describe,
    materialise,
    ordered,
    snapshot,
)

# Node paths in the fixture tree, and names to ask each for (present at some
# instants, absent at others, or never there).
PATHS = [(), ("group",), ("group", "inner"), ("by_zone",), ("by_zone", "z2")]
NAMES = ["flat_rate", "late", "group", "scale", "by_zone", "x", "y", "inner"]
NAMES += ["1", "2", "z1", "z2", "z3", "amount", "rate", "nope"]
KINDS = ["getattr", "getitem", "iter", "children", "dir", "repr", "vector"]

reads = st.tuples(
    st.sampled_from(PATHS), st.sampled_from(KINDS), st.sampled_from(NAMES)
)


def _comparable(value):
    if isinstance(value, ParameterNodeAtInstant):
        return materialise(value)
    return describe(value)


@hypothesis.given(
    instant=st.sampled_from(INSTANTS), sequence=st.lists(reads, max_size=30)
)
@hypothesis.settings(max_examples=400, deadline=None)
def test_reads_in_any_order_match_the_up_front_build(instant, sequence):
    tree = build_tree()
    expected_root = snapshot(tree, instant)
    root = tree.get_at_instant(instant)

    for path, kind, name in sequence:
        node, expected = root, expected_root
        for step in path:
            node, expected = node[step], expected[step]

        if kind == "getattr":
            if name in expected:
                assert _comparable(getattr(node, name)) == expected[name]
            else:
                with pytest.raises(ParameterNotFoundError):
                    getattr(node, name)
                assert not hasattr(node, name)
        elif kind == "getitem":
            if name in expected:
                assert _comparable(node[name]) == expected[name]
            else:
                with pytest.raises(KeyError):
                    node[name]
        elif kind == "iter":
            assert list(node) == list(expected)
            assert (name in node) == (name in expected)
        elif kind == "children":
            assert ordered(materialise(node)) == ordered(expected)
        elif kind == "dir":
            assert set(expected) <= set(dir(node))
        elif kind == "repr":
            assert all(child in repr(node) for child in expected)
        elif kind == "vector" and path == ("by_zone",) and expected["z1"]:
            zones = np.array(["z3", "z1"])
            np.testing.assert_array_equal(
                node[zones].amount,
                [expected["z3"]["amount"], expected["z1"]["amount"]],
            )

    assert ordered(materialise(root)) == ordered(expected_root)
