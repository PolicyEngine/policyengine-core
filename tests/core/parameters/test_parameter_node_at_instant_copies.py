"""Copies, pickles and forks of parameter trees whose nodes at instant are
partly read.

A node at instant resolves a child on first read, so reads change the node,
its shared state and the at-instant caches of the parameter nodes below it.
Copying or pickling must not fail when such a read happens part-way through
(here, a read made by a value that is being copied, as another thread could
make it at that moment), and a copy must read the parameters it was copied
with.
"""

import copy
import os
import pickle
import signal
import threading

import pytest

from policyengine_core.parameters import ParameterNode, ParameterNodeAtInstant
from tests.fixtures.parameter_nodes_at_instant import (
    TREE,
    build_tree,
    materialise,
    ordered,
    snapshot,
)

INSTANT = "2017-03-01"
OTHER_INSTANT = "2018-01-01"
NEW_INSTANT = "2019-01-01"
PROTOCOLS = list(range(pickle.HIGHEST_PROTOCOL + 1))

DUPLICATES = {
    "deepcopy": copy.deepcopy,
    **{
        f"pickle-{protocol}": (
            lambda value, protocol=protocol: pickle.loads(
                pickle.dumps(value, protocol=protocol)
            )
        )
        for protocol in PROTOCOLS
    },
}


def partly_read_tree():
    tree = build_tree()
    held = tree(INSTANT)
    held.group.x
    held.by_zone.z1.amount
    return tree, held


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=DUPLICATES.keys())
def test_a_partly_read_node_copies_with_every_value(duplicate):
    tree, held = partly_read_tree()

    copied = duplicate(held)

    assert ordered(materialise(copied)) == ordered(snapshot(tree, INSTANT))


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=DUPLICATES.keys())
def test_a_tree_holding_partly_read_nodes_copies_with_its_cache(duplicate):
    tree, _ = partly_read_tree()

    copied = duplicate(tree)

    assert INSTANT in copied._at_instant_cache
    assert copied(INSTANT) is copied._at_instant_cache[INSTANT]
    assert ordered(materialise(copied(INSTANT))) == ordered(snapshot(tree, INSTANT))


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=DUPLICATES.keys())
def test_a_copied_tree_reads_its_own_parameters(duplicate):
    tree, held = partly_read_tree()
    copied = duplicate(tree)
    copied_held = copied._at_instant_cache[INSTANT]

    # Changes to either tree after the copy stay in that tree, for children
    # read before the copy and for children not read yet.
    tree.flat_rate.update(period="2017", value=0.5)
    tree.group.x.update(period="2017", value=0.6)
    copied.by_zone.z2.amount.update(period="2017", value=7)

    assert copied_held.flat_rate == 0.1
    assert copied_held.group.x == 0.1
    assert copied(INSTANT).flat_rate == 0.1
    assert copied(INSTANT).by_zone.z2.amount == 7
    assert held.by_zone.z2.amount == 200
    assert tree(INSTANT).flat_rate == 0.5


class ReadsWhenCopied:
    """A parameter value that reads parameters when it is copied or pickled,
    as another thread could while a copy is under way."""

    def __init__(self, read):
        self.read = read

    def __deepcopy__(self, memo):
        self.read()
        return "copied"

    def __reduce__(self):
        self.read()
        return (str, ("copied",))


def tree_with_a_value_that_reads(read_for):
    """A tree whose nodes at instant hold a value that reads parameters when
    it is copied.

    Only nodes at instant hold that value: the root's node at ``INSTANT`` as
    ``first``, and the group's node at ``OTHER_INSTANT`` as ``x``. The
    parameters hold plain values again.
    """
    tree = ParameterNode(
        "",
        data={
            "first": {"values": {"2010-01-01": 0}},
            "second": {"values": {"2010-01-01": 2}},
            "group": {"x": {"values": {"2010-01-01": 3}}},
        },
    )
    held = tree(INSTANT)
    reader = ReadsWhenCopied(lambda: read_for(tree, held))
    first = tree.children["first"]
    x = tree.group.children["x"]
    first.values_list[0].value = x.values_list[0].value = reader
    assert held.first is reader
    assert tree.group(OTHER_INSTANT).x is reader
    first.values_list[0].value = 0
    x.values_list[0].value = 3
    return tree, held


READS_DURING_A_COPY = {
    # Resolves a child of the node being copied.
    "a child of the node": lambda tree, held: held.second,
    # Resolves a node child, which fills that parameter node's cache.
    "a node child": lambda tree, held: held.group.x,
    # A new instant fills the root's cache and the group's.
    "a new instant": lambda tree, held: tree(NEW_INSTANT).group.x,
}


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=DUPLICATES.keys())
@pytest.mark.parametrize(
    "read", READS_DURING_A_COPY.values(), ids=READS_DURING_A_COPY.keys()
)
def test_a_read_during_a_copy_of_a_node_does_not_break_it(read, duplicate):
    _, held = tree_with_a_value_that_reads(read)

    copied = duplicate(held)

    assert copied.first == "copied"
    assert copied.second == 2
    assert copied.group.x == 3


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=DUPLICATES.keys())
@pytest.mark.parametrize(
    "read", READS_DURING_A_COPY.values(), ids=READS_DURING_A_COPY.keys()
)
def test_a_read_during_a_copy_of_a_tree_does_not_break_it(read, duplicate):
    tree, _ = tree_with_a_value_that_reads(read)

    copied = duplicate(tree)

    assert copied(INSTANT).first == "copied"
    assert copied(INSTANT).second == 2
    assert copied(INSTANT).group.x == 3
    assert copied.group(OTHER_INSTANT).x == "copied"
    assert copied(NEW_INSTANT).group.x == 3


def test_a_shallow_copy_of_a_tree_shares_its_cache_as_before():
    tree, held = partly_read_tree()

    shallow = copy.copy(tree)

    assert shallow._at_instant_cache is tree._at_instant_cache
    assert shallow(INSTANT) is held


class BlockingChild:
    """A parameter child whose first resolution waits until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def _get_at_instant(self, instant):
        self.entered.set()
        assert self.release.wait(10), "never released"
        return 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_process_forked_during_a_resolution_can_resolve_children():
    tree = ParameterNode("", data={"other": {"values": {"2010-01-01": 2}}})
    slow = BlockingChild()
    tree.children = {"slow": slow, **tree.children}
    held = tree(INSTANT)
    reader = threading.Thread(target=lambda: held.slow)
    reader.start()
    assert slow.entered.wait(10)
    # The reader now holds the resolution lock, and the forked process gets a
    # copy of the lock but not the reader.
    try:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - runs in the forked process
            signal.alarm(10)
            try:
                os._exit(0 if held.other == 2 else 1)
            except BaseException:
                os._exit(2)
        _, status = os.waitpid(pid, 0)
    finally:
        slow.release.set()
        reader.join(10)

    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status


class SlottedNodeAtInstant(ParameterNodeAtInstant):
    __slots__ = ("marker",)


class SlottedParameterNode(ParameterNode):
    __slots__ = ("marker",)


@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, *DUPLICATES.values()],
    ids=["copy", *DUPLICATES.keys()],
)
def test_slotted_subclasses_keep_their_slots(duplicate):
    tree, held = partly_read_tree()
    node = SlottedNodeAtInstant.__new__(SlottedNodeAtInstant)
    node.__dict__.update(held.__dict__)
    node.marker = "node at instant"
    parameter_node = SlottedParameterNode("", data=TREE)
    parameter_node.marker = "parameter node"
    parameter_node(INSTANT).group.x

    copied_node = duplicate(node)
    copied_parameter_node = duplicate(parameter_node)

    assert copied_node.marker == "node at instant"
    assert ordered(materialise(copied_node)) == ordered(snapshot(tree, INSTANT))
    assert copied_parameter_node.marker == "parameter node"
    assert ordered(materialise(copied_parameter_node(INSTANT))) == ordered(
        snapshot(tree, INSTANT)
    )
