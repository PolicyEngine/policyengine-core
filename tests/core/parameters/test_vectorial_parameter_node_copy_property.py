"""A copy of a vectorial parameter node reads what the original reads.

Random paths through ``rate.<family>.<housing>.<zone>`` take each level by
name or by an array of keys (strings, an ``EnumArray`` or enum items). Partway
along, the vectorial node reached so far is copied with ``copy``,
``copy.deepcopy`` or a pickle round trip, and the rest of the path is read
from the copy and from the original. ``RATE`` gives the values each read must
return. ``test_vectorial_parameter_node_copy.py`` pins the same behaviour with
examples.
"""

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.enums import EnumArray
from policyengine_core.parameters import VectorialParameterNodeAtInstant
from policyengine_core.tools import assert_near
from tests.fixtures.vectorial_parameter_nodes import (
    COPIERS,
    DEEP_COPIERS,
    FAMILIES,
    HOUSINGS,
    INSTANT,
    RATE,
    ZONES,
    Family,
    Housing,
    fancy_indexing_parameters,
)

LEVELS = (FAMILIES, HOUSINGS, ZONES)
ENUMS = (Family, Housing, None)


@st.composite
def _paths(draw):
    """Steps for the three levels, and how many to take before copying."""
    size = draw(st.integers(1, 6))
    steps = []
    for children, enum in zip(LEVELS, ENUMS):
        if draw(st.booleans()):
            # One child for every element, read as an attribute or an item.
            steps.append(
                (
                    "name",
                    draw(st.sampled_from(children)),
                    draw(st.sampled_from(["attribute", "item"])),
                )
            )
        else:
            keys = draw(
                st.lists(st.sampled_from(children), min_size=size, max_size=size)
            )
            kinds = ["strings"] + (["enum_array", "enum_items"] if enum else [])
            steps.append(("keys", keys, draw(st.sampled_from(kinds))))
    # Copy a vectorial node: after an array of keys, before the leaf.
    first_keys = next(
        (level for level, step in enumerate(steps) if step[0] == "keys"), None
    )
    hypothesis.assume(first_keys is not None and first_keys < 2)
    copy_after = draw(st.integers(first_keys + 1, 2))
    return size, steps, copy_after


def _read(node, step, enum):
    if step[0] == "name":
        _, child, how = step
        return getattr(node, child) if how == "attribute" else node[child]
    _, keys, kind = step
    if kind == "strings":
        return node[np.array(keys)]
    items = [enum[key] for key in keys]
    if kind == "enum_items":
        return node[np.array(items, dtype=object)]
    return node[EnumArray(np.array([item.index for item in items]), enum)]


def _read_path(node, steps, levels):
    for level in levels:
        node = _read(node, steps[level], ENUMS[level])
    return node


def _expected(size, steps):
    def key(level, element):
        step = steps[level]
        return step[1] if step[0] == "name" else step[1][element]

    return [
        RATE[key(0, element)][key(1, element)][key(2, element)]
        for element in range(size)
    ]


@hypothesis.settings(max_examples=300, deadline=None)
@hypothesis.given(
    path=_paths(),
    copier_name=st.sampled_from(sorted(COPIERS)),
    read_before_copying=st.booleans(),
)
def test_copy_reads_what_the_original_reads(path, copier_name, read_before_copying):
    size, steps, copy_after = path
    node = _read_path(
        fancy_indexing_parameters()(INSTANT).rate, steps, range(copy_after)
    )
    assert type(node) is VectorialParameterNodeAtInstant
    rest = range(copy_after, len(LEVELS))
    expected = _expected(size, steps)
    if read_before_copying:
        # Fills the node's lookup caches before it is copied.
        assert_near(_read_path(node, steps, rest), expected)

    copied = COPIERS[copier_name](node)

    assert type(copied) is VectorialParameterNodeAtInstant
    assert copied._name == node._name
    assert copied._instant_str == node._instant_str
    assert copied.vector.dtype == node.vector.dtype
    assert copied.vector.tobytes() == node.vector.tobytes()
    original_array, copied_array = np.asarray(node), np.asarray(copied)
    assert copied_array.dtype == original_array.dtype
    assert copied_array.tobytes() == original_array.tobytes()
    if copier_name in DEEP_COPIERS:
        assert not np.shares_memory(copied.vector, node.vector)
    else:
        assert copied.vector is node.vector
    assert_near(_read_path(copied, steps, rest), expected)
    assert_near(_read_path(node, steps, rest), expected)
