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

from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import (
    Parameter,
    ParameterNode,
    ParameterNodeAtInstant,
)
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


def give_plain_caches(tree):
    """Give every parameter node a plain ``dict`` cache, as policyengine-us's
    ``copy_parameter_nodes`` does, and as a tree pickled by an earlier core
    has."""
    for node in (tree, *tree.get_descendants()):
        if isinstance(node, ParameterNode):
            node._at_instant_cache = {}


def tree_with_a_value_that_reads(read_for, caches="core"):
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
    if caches == "plain":
        give_plain_caches(tree)
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


@pytest.mark.parametrize("caches", ["core", "plain"])
@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=DUPLICATES.keys())
@pytest.mark.parametrize(
    "read", READS_DURING_A_COPY.values(), ids=READS_DURING_A_COPY.keys()
)
def test_a_read_during_a_copy_of_a_tree_does_not_break_it(read, duplicate, caches):
    tree, _ = tree_with_a_value_that_reads(read, caches)

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
def test_a_slotted_node_at_instant_keeps_its_slots(duplicate):
    tree, held = partly_read_tree()
    node = SlottedNodeAtInstant.__new__(SlottedNodeAtInstant)
    node.__dict__.update(held.__dict__)
    node.marker = "node at instant"

    copied = duplicate(node)

    assert copied.marker == "node at instant"
    assert ordered(materialise(copied)) == ordered(snapshot(tree, INSTANT))


SLOTTED_PARAMETER_NODE_DUPLICATES = {
    "copy": copy.copy,
    **{
        name: duplicate
        for name, duplicate in DUPLICATES.items()
        # Protocols 0 and 1 refuse a class with ``__slots__`` and no
        # ``__getstate__``, as before.
        if name not in ("pickle-0", "pickle-1")
    },
}


@pytest.mark.parametrize(
    "duplicate",
    SLOTTED_PARAMETER_NODE_DUPLICATES.values(),
    ids=SLOTTED_PARAMETER_NODE_DUPLICATES.keys(),
)
def test_a_slotted_parameter_node_keeps_its_slots(duplicate):
    parameter_node = SlottedParameterNode("", data=TREE)
    parameter_node.marker = "parameter node"
    parameter_node(INSTANT).group.x

    copied = duplicate(parameter_node)

    assert copied.marker == "parameter node"
    assert ordered(materialise(copied(INSTANT))) == ordered(
        snapshot(parameter_node, INSTANT)
    )


# A deep copy or pickle of a node at instant is a snapshot, as when every
# child was built up front: it holds no parameter nodes.


class RecordsDeepCopies:
    copies = 0

    def __deepcopy__(self, memo):
        RecordsDeepCopies.copies += 1
        return self


def tree_with_an_unrelated_branch():
    tree = build_tree()
    tree.add_child(
        "unrelated_branch_marker",
        ParameterNode(
            "unrelated_branch_marker",
            data={"amount": {"values": {"2010-01-01": 1}}},
        ),
    )
    tree.unrelated_branch_marker.amount.values_list[0].value = RecordsDeepCopies()
    return tree


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_pickled_node_leaves_the_parameter_tree_behind(protocol):
    tree = tree_with_an_unrelated_branch()
    group = tree(INSTANT).group
    group.x

    data = pickle.dumps(group, protocol=protocol)

    assert b"unrelated_branch_marker" not in data
    assert ordered(materialise(pickle.loads(data))) == ordered(
        snapshot(tree.group, INSTANT)
    )


def test_a_deep_copied_node_leaves_the_parameter_tree_behind():
    tree = tree_with_an_unrelated_branch()
    group = tree(INSTANT).group
    group.x
    RecordsDeepCopies.copies = 0

    copied = copy.deepcopy(group)

    assert RecordsDeepCopies.copies == 0
    assert ordered(materialise(copied)) == ordered(snapshot(tree.group, INSTANT))


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=DUPLICATES.keys())
def test_add_child_on_a_copy_of_a_listed_node_is_listed(duplicate):
    tree = ParameterNode(
        "",
        data={"x": {"values": {"2010-01-01": 10}}, "y": {"values": {"2010-01-01": 20}}},
    )
    held = tree(INSTANT)
    list(held)

    copied = duplicate(held)
    copied.add_child("z", 30)

    assert list(copied) == ["x", "y", "z"]
    assert "z" in copied
    assert copied["z"] == 30 and copied.z == 30
    assert list(copied._children) == ["x", "y", "z"]


@pytest.mark.parametrize("read_in_full", [list, lambda node: "x" in node])
def test_add_child_on_a_copy_of_a_fully_read_node_changes_only_shared_items(
    read_in_full,
):
    tree = ParameterNode(
        "",
        data={"x": {"values": {"2010-01-01": 10}}, "y": {"values": {"2010-01-01": 20}}},
    )
    held = tree(INSTANT)
    read_in_full(held)
    duplicate = copy.copy(held)

    duplicate.add_child("x", 99)
    duplicate.add_child("extra", 77)

    # Items are one mapping shared by copies; attributes belong to each copy.
    assert held["x"] == 99 and held.x == 10
    assert held["extra"] == 77
    with pytest.raises(ParameterNotFoundError):
        held.extra
    assert duplicate.x == 99 and duplicate.extra == 77
    assert list(held._children) == ["x", "y", "extra"]


def test_children_written_directly_are_items():
    tree = ParameterNode(
        "",
        data={"x": {"values": {"2010-01-01": 10}}, "y": {"values": {"2010-01-01": 20}}},
    )
    held = tree(INSTANT)
    held.add_child("added", 1)
    list(held)

    held._children["written"] = 2

    assert held["written"] == 2
    assert held["x"] == 10


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=DUPLICATES.keys())
def test_shallow_copies_sharing_a_cache_still_share_it_once_copied(duplicate):
    tree, _ = partly_read_tree()
    alias = copy.copy(tree)

    copied_tree, copied_alias = duplicate((tree, alias))
    copied_tree.flat_rate.update(period="2017", value=0.5)

    assert copied_alias._at_instant_cache is copied_tree._at_instant_cache
    assert copied_alias(INSTANT).flat_rate == 0.5


def test_a_plain_cache_is_replaced_by_a_copyable_one_when_first_filled():
    tree, held = partly_read_tree()
    give_plain_caches(tree)
    plain = tree._at_instant_cache
    plain[INSTANT] = held

    tree(OTHER_INSTANT)

    assert type(tree._at_instant_cache) is not dict
    assert isinstance(tree._at_instant_cache, dict)
    assert tree._at_instant_cache[INSTANT] is held
    assert list(plain) == [INSTANT]  # Not written to.
    assert tree(INSTANT) is held


@pytest.mark.parametrize(
    "replacement",
    [dict, lambda: type(ParameterNode("", data={})._at_instant_cache)()],
    ids=["plain", "core"],
)
@pytest.mark.parametrize("starting_cache", ["plain", "core"])
def test_a_cache_replaced_while_a_node_is_built_is_the_one_filled(
    replacement, starting_cache
):
    # policyengine-us's isolate_parameter_tracing replaces a root's cache to
    # drop an earlier request's nodes. If that happens while the root is read,
    # the node built goes into the cache the root holds afterwards.
    tree = ParameterNode("", data={"a": {"values": {"2010-01-01": 1}}})

    class ReplacesTheRootCache(Parameter):
        replaced = False

        def _get_at_instant(self, instant):
            if not ReplacesTheRootCache.replaced:
                ReplacesTheRootCache.replaced = True
                tree._at_instant_cache = replacement()
            return super()._get_at_instant(instant)

    # Named like a node's own attribute, so it resolves while the node is
    # built.
    tree.add_child(
        "_vectorial_node",
        ReplacesTheRootCache("_vectorial_node", {"values": {"2010-01-01": 5}}),
    )
    if starting_cache == "plain":
        tree._at_instant_cache = {}
    tree._at_instant_cache[OTHER_INSTANT] = "a node from an earlier request"

    node = tree(INSTANT)

    assert list(tree._at_instant_cache) == [INSTANT]
    assert tree(INSTANT) is node


def test_children_written_directly_are_items_and_not_attributes():
    tree = ParameterNode(
        "",
        data={"x": {"values": {"2010-01-01": 10}}, "y": {"values": {"2010-01-01": 20}}},
    )
    held = tree(INSTANT)
    children = held._children

    children["written"] = 2
    children["x"] = 99
    del children["y"]

    assert held["written"] == 2 and held["x"] == 99
    with pytest.raises(KeyError):
        held["y"]
    with pytest.raises(ParameterNotFoundError):
        held.written
    assert held.x == 10 and held.y == 20


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_pickled_cache_is_a_plain_dict_that_earlier_cores_load(protocol):
    tree = build_tree()

    data = pickle.dumps(tree, protocol=protocol)
    restored = pickle.loads(data)

    assert b"_AtInstantCache" not in data
    assert type(restored._at_instant_cache) is dict
    assert ordered(materialise(restored(INSTANT))) == ordered(snapshot(tree, INSTANT))
