"""``ParameterNodeAtInstant`` resolves its children on first read.

It used to build every descendant in its constructor, so each instant a
system was asked about cost a copy of the whole parameter tree (about 34,000
node objects and 16-33 MB per instant for policyengine-us) that the system
then kept. These tests pin that nothing is built before it is read, and that
everything read is what the up-front build held.
"""

import copy
import gc
import pickle
import threading
import weakref

import numpy as np
import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import ParameterNode, ParameterNodeAtInstant
from policyengine_core.parameters.parameter_node_at_instant import _RESOLVED
from policyengine_core.tracers import FullTracer
from tests.fixtures.parameter_nodes_at_instant import (
    INSTANTS,
    build_tree,
    materialise,
    ordered,
    snapshot,
)


@pytest.mark.parametrize("instant", INSTANTS)
def test_fully_read_node_matches_the_up_front_build(instant):
    tree = build_tree()
    assert ordered(materialise(tree.get_at_instant(instant))) == ordered(
        snapshot(tree, instant)
    )


@pytest.mark.parametrize(
    "instant", ["2000-01-01", "2012-06-01", "2015-01-01", "2017-01-01", "2030-01-01"]
)
def test_country_template_matches_the_up_front_build(instant):
    parameters = CountryTaxBenefitSystem().parameters
    assert ordered(materialise(parameters.get_at_instant(instant))) == ordered(
        snapshot(parameters, instant)
    )


def resolved(node_at_instant):
    return list(node_at_instant.__dict__[_RESOLVED])


def test_nothing_is_resolved_until_it_is_read():
    at_instant = build_tree().get_at_instant("2017-03-01")
    assert resolved(at_instant) == []

    assert at_instant.group.x == 0.1
    # Only the path that was read: the root's ``group`` and the group's ``x``.
    assert resolved(at_instant) == ["group"]
    assert resolved(at_instant.group) == ["x"]


def test_asking_for_many_instants_builds_one_node_each():
    tree = build_tree()
    gc.collect()
    before = sum(isinstance(o, ParameterNodeAtInstant) for o in gc.get_objects())
    for year in range(2000, 2100):
        tree.get_at_instant(f"{year}-01-01")
    after = sum(isinstance(o, ParameterNodeAtInstant) for o in gc.get_objects())
    # The up-front build made 9 nodes per instant for this tree (900 here).
    assert after - before == 100


def test_a_read_child_is_resolved_once():
    at_instant = build_tree().get_at_instant("2017-03-01")
    # A scale is rebuilt by every ``_get_at_instant`` call, so identity shows
    # that the node kept what it resolved.
    assert at_instant.scale is at_instant.scale
    assert at_instant["scale"] is at_instant.scale
    assert at_instant._children["scale"] is at_instant.scale
    assert at_instant.group is at_instant["group"]


def test_children_are_in_parameter_order_whatever_was_read_first():
    tree = build_tree()
    at_instant = tree.get_at_instant("2030-01-01")
    at_instant.by_zone
    at_instant["late"]
    at_instant.group
    assert list(at_instant) == list(snapshot(tree, "2030-01-01"))
    assert list(at_instant._children) == list(snapshot(tree, "2030-01-01"))


def test_child_without_a_value_yet_is_absent():
    at_instant = build_tree().get_at_instant("2017-03-01")
    with pytest.raises(ParameterNotFoundError):
        at_instant.late
    with pytest.raises(ParameterNotFoundError) as error:
        at_instant.group.y
    assert str(error.value) == (
        "The parameter 'group[y]' was not found in the 2017-03-01 "
        "tax and benefit system."
    )
    with pytest.raises(KeyError):
        at_instant["late"]
    assert not hasattr(at_instant, "late")
    assert "late" not in at_instant
    assert "late" not in at_instant._children
    assert at_instant.group.inner._children == {}


def test_unknown_names_raise_as_before():
    at_instant = build_tree().get_at_instant("2017-03-01")
    for read_first in (False, True):
        if read_first:
            at_instant._children
        with pytest.raises(ParameterNotFoundError) as error:
            at_instant.group.nope
        assert str(error.value) == (
            "The parameter 'group[nope]' was not found in the 2017-03-01 "
            "tax and benefit system."
        )
        with pytest.raises(KeyError):
            at_instant.group["nope"]
        assert not hasattr(at_instant, "nope")


def test_vectorial_indexing_reads_the_whole_subtree():
    at_instant = build_tree().get_at_instant("2020-01-01")
    zones = np.array(["z1", "z3", "z2"])
    np.testing.assert_array_equal(at_instant.by_zone[zones].amount, [200, 600, 400])
    np.testing.assert_allclose(at_instant.by_zone[zones].rate, [0.1, 0.3, 0.2])
    # Nothing outside the indexed subtree was resolved for it.
    assert resolved(at_instant) == ["by_zone"]


def test_dir_lists_children_for_completion():
    at_instant = build_tree().get_at_instant("2017-03-01")
    names = dir(at_instant)
    assert {"flat_rate", "group", "scale", "by_zone"} <= set(names)
    assert "late" not in names


def test_repr_shows_every_child():
    text = repr(build_tree().get_at_instant("2017-03-01"))
    for name in ("flat_rate", "group", "scale", "by_zone", "z3"):
        assert name in text


def test_add_child_before_and_after_a_full_read():
    at_instant = build_tree().get_at_instant("2017-03-01")
    at_instant.add_child("extra", 1)
    assert at_instant.extra == 1 and at_instant["extra"] == 1
    assert list(at_instant._children)[-1] == "extra"
    at_instant.add_child("later", 2)
    assert at_instant.later == 2 and at_instant._children["later"] == 2


def test_a_node_asked_for_after_an_update_reads_the_update():
    tree = build_tree()
    assert tree.get_at_instant("2017-03-01").flat_rate == 0.1
    tree.flat_rate.update(value=0.5, period="year:2017:1")
    assert tree.get_at_instant("2017-03-01").flat_rate == 0.5


def test_a_child_read_before_an_update_keeps_what_was_read():
    tree = build_tree()
    held = tree.get_at_instant("2017-03-01")
    assert held.flat_rate == 0.1
    tree.flat_rate.update(value=0.5, period="year:2017:1")
    assert held.flat_rate == 0.1


def test_a_child_not_read_before_an_update_reads_the_update():
    """Intended difference from the up-front build, which held 0.1 here: a
    node held across a change to its parameters reads, for each child, the
    value at the time that child is first read."""
    tree = build_tree()
    held = tree.get_at_instant("2017-03-01")
    tree.flat_rate.update(value=0.5, period="year:2017:1")
    assert held.flat_rate == 0.5


def test_deep_copied_tree_keeps_its_own_nodes_at_instant():
    tree = build_tree()
    tree.get_at_instant("2017-03-01").group.x  # cache a partly read node
    clone = copy.deepcopy(tree)
    cached = clone._at_instant_cache["2017-03-01"]
    clone.group.y.update(value=0.9, period="year:2010:30")
    clone.flat_rate.update(value=0.7, period="year:2010:30")
    # The copy's cached node reads the copy, not the original tree.
    assert cached.flat_rate == 0.7
    assert tree.get_at_instant("2017-03-01").flat_rate == 0.1
    assert not hasattr(tree.get_at_instant("2017-03-01").group, "y")


def test_copy_of_a_node_at_instant_reads_the_same_values():
    at_instant = build_tree().get_at_instant("2017-03-01")
    at_instant.group
    for duplicate in (copy.copy(at_instant), copy.deepcopy(at_instant)):
        assert materialise(duplicate) == materialise(at_instant)


@pytest.mark.parametrize(
    "child_name",
    # Not ``add_child``: ParameterNode itself cannot hold a child of that name.
    ["_name", "_instant_str", "_vectorial_node", "_node", "_resolved"]
    + ["_resolve", "_resolve_all", "_store", "_materialise"],
)
def test_children_named_like_the_nodes_own_attributes(child_name):
    """Such a child reads as its value whatever the read order, as when every
    child was set up front, and leaves its siblings readable."""
    tree = ParameterNode(
        "",
        data={
            child_name: {"values": {"2010-01-01": 42}},
            "ok": {"values": {"2010-01-01": 7}},
        },
    )
    for read_name_first in (True, False):
        tree._at_instant_cache.clear()
        at_instant = tree.get_at_instant("2017-03-01")
        if read_name_first:
            assert getattr(at_instant, child_name) == 42
            assert at_instant.ok == 7 and at_instant["ok"] == 7
        else:
            assert at_instant["ok"] == 7 and at_instant.ok == 7
            assert getattr(at_instant, child_name) == 42
        assert at_instant[child_name] == 42
        assert set(at_instant._children) == {child_name, "ok"}


class BlockingChild:
    """A parameter child whose first resolution waits until released."""

    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def _get_at_instant(self, instant):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            assert self.release.wait(10), "never released"
        return object()


@pytest.mark.parametrize(
    "second_read",
    [
        lambda node: node.slow,
        lambda node: node["slow"],
        lambda node: node._children["slow"],
        lambda node: [node[name] for name in node][0],
    ],
)
def test_concurrent_first_reads_resolve_a_child_once(second_read):
    tree = ParameterNode("", data={"other": {"values": {"2010-01-01": 1}}})
    slow = BlockingChild()
    tree.children = {"slow": slow, **tree.children}
    at_instant = tree.get_at_instant("2017-03-01")
    results, errors = [], []

    def read(reader):
        try:
            results.append(reader(at_instant))
        except Exception as error:  # noqa: BLE001 - recorded for the assertion
            errors.append(error)

    first = threading.Thread(target=read, args=(lambda node: node.slow,))
    first.start()
    assert slow.entered.wait(10)
    second = threading.Thread(target=read, args=(second_read,))
    second.start()
    second.join(0.2)  # let the second reader reach the child too
    slow.release.set()
    first.join(10)
    second.join(10)

    assert not errors, errors
    assert len(results) == 2 and results[0] is results[1]
    assert slow.calls == 1


def test_shallow_copies_read_the_same_children():
    held = build_tree().get_at_instant("2017-03-01")
    duplicate = copy.copy(held)
    original_scale = held.scale
    # A scale is rebuilt by each resolution, so identity shows one resolution.
    assert duplicate.scale is original_scale
    assert duplicate["scale"] is original_scale
    assert held["scale"] is original_scale
    duplicate._children
    assert held._children["scale"] is original_scale
    assert held.by_zone is duplicate["by_zone"]


@pytest.mark.parametrize(
    "duplicate", [copy.deepcopy, lambda n: pickle.loads(pickle.dumps(n))]
)
def test_traced_nodes_copy_and_pickle(duplicate):
    tree = build_tree()
    tree.trace = True
    tree.tracer = FullTracer()
    tree.branch_name = "default"
    traced = tree.get_at_instant("2017-03-01")
    traced.group.x  # partly resolved, through the tracing wrapper
    raw = traced.parameter_node_at_instant
    for node in (raw, traced):
        assert duplicate(node).flat_rate == 0.1
        assert duplicate(node).group.x == 0.1


def test_a_child_without_a_value_is_resolved_once():
    tree = ParameterNode("", data={"late": {"values": {"2030-01-01": 1}}})
    held = tree.get_at_instant("2020-01-01")
    calls = []
    original = tree.children["late"]._get_at_instant

    def counting(instant):
        calls.append(instant)
        return original(instant)

    tree.children["late"]._get_at_instant = counting
    for _ in range(3):
        with pytest.raises(ParameterNotFoundError):
            held.late
        with pytest.raises(KeyError):
            held["late"]
    assert list(held) == []
    assert len(calls) == 1

    # A child already read keeps what that read found, value or none.
    tree.children["late"].update(value=2, period="year:2020:1")
    with pytest.raises(KeyError):
        held["late"]
    assert tree.get_at_instant("2020-01-01").late == 2


def test_a_node_lets_go_of_its_parameter_node_once_every_child_is_read():
    tree = ParameterNode(
        "",
        data={
            "group": {
                "x": {"values": {"2010-01-01": 1}},
                "late": {"values": {"2030-01-01": 2}},
            },
            "unused": {"values": {"2010-01-01": 3}},
        },
    )
    group = tree.get_at_instant("2017-03-01").group
    tree_ref = weakref.ref(tree)
    del tree
    gc.collect()
    # Partly read: the node still needs the tree to resolve the rest.
    assert tree_ref() is not None
    assert group.x == 1
    with pytest.raises(ParameterNotFoundError):
        group.late
    gc.collect()
    assert tree_ref() is None
    assert group.x == 1 and list(group) == ["x"]
