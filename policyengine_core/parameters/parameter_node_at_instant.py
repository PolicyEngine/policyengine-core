import os
import sys
import threading
from typing import TYPE_CHECKING, Iterable, Union

import numpy

from policyengine_core import parameters, periods, tools
from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import helpers

if TYPE_CHECKING:
    from policyengine_core.parameters.parameter_node import ParameterNode

from policyengine_core.parameters.vectorial_parameter_node_at_instant import (
    VectorialParameterNodeAtInstant,
)


# The one instance attribute a node keeps for itself; a parameter of this
# name is rejected.
_STATE = "_ParameterNodeAtInstant__state"
# Attributes a node sets on itself, which a child of the same name shadows.
_OWN_NAMES = frozenset({"_name", "_instant_str", "_children", "_vectorial_node"})
_MISSING = object()

# Held while a child is resolved for the first time, so that concurrent
# readers never see a half-resolved node or resolve one child twice. Reads of
# a resolved child never take it. Reentrant because resolving a child can
# build that child's own node.
_RESOLUTION_LOCK = threading.RLock()


def _replace_resolution_lock() -> None:
    """Give a forked process a lock that no thread holds. A thread that held
    the lock when another thread forked does not exist in the forked process,
    so nothing there would release it, and the first resolution would wait for
    ever."""
    global _RESOLUTION_LOCK
    _RESOLUTION_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_replace_resolution_lock)


def _copied_state(instance):
    """What ``object.__getstate__`` gives, with the instance dictionary
    copied: a ``dict``, ``None`` when it is empty, or a ``(dict, slots)`` pair
    for an instance with ``__slots__``.

    Reads change the dictionaries of parameter nodes and nodes at instant, so
    copying or pickling a dictionary itself would fail if another thread read
    meanwhile. Copying a ``dict`` is a single step for other threads.
    """
    state = object.__getstate__(instance)
    if isinstance(state, tuple):
        attributes, slots = state
        return (None if attributes is None else dict(attributes), slots)
    return None if state is None else dict(state)


# Instant strings that ``periods.instant`` has accepted.
_DATES = set()


def _is_date(instant_str) -> bool:
    if not isinstance(instant_str, str):
        return False
    if instant_str in _DATES:
        return True
    try:
        # As a scale's build reads it: ``str`` checks that the day exists.
        str(periods.instant(instant_str))
    except ValueError:
        return False
    _DATES.add(instant_str)
    return True


class _LazyChildren:
    """What a node at instant knows about its children. A shallow copy of the
    node shares it, as copies of a node that built every child up front shared
    their ``_children`` mapping."""

    __slots__ = ("instant", "source", "resolved", "absent", "added", "children")

    def __init__(self, instant: str, source: dict):
        self.instant = instant
        # Each child's parameter node as at construction, in order; None once
        # every child is resolved.
        self.source = source
        # Children resolved from ``source`` so far.
        self.resolved = {}
        # Children of ``source`` with no value at the instant.
        self.absent = set()
        # Children given with ``add_child``, in call order.
        self.added = {}
        # Every child in order, once all of ``source`` is resolved.
        self.children = None

    def __getstate__(self) -> dict:
        # Only a deep copy or a pickle copies this state; a shallow copy of
        # the node shares it. Either is a snapshot, as when every child was
        # built up front: resolving every child first lets go of ``source``,
        # whose parameter nodes would bring the whole tree, through their
        # parents, into the copy. Once every child is resolved, reads no
        # longer change these containers (only ``add_child`` does), so they
        # are given as they are and keep their identities. Pickle protocols 0
        # and 1 also refuse a class with ``__slots__`` that does not define
        # this.
        with _RESOLUTION_LOCK:
            _resolve_remaining(self)
            return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state) -> None:
        if isinstance(state, tuple):
            # ``(None, slots)``, as ``object.__getstate__`` gives for a class
            # with ``__slots__``.
            state = state[1]
        for name in self.__slots__:
            setattr(self, name, state[name])


class ParameterNodeAtInstant:
    """
    Parameter node of the legislation, at a given instant.

    Children resolve on first read and are kept from then on. Building every
    descendant up front made each instant a model is asked about cost a copy of
    the whole parameter tree (16 to 33 MB and several seconds per instant for
    policyengine-us), which every tax-benefit system then kept for its
    lifetime. Which children a node has, and in what order, is fixed when it is
    built. A child's value is taken when the child is first read, so a node
    held across an in-place change to its parameters reads the changed value
    for a child it has not read yet; once read (including found to have no
    value), a child keeps what that read found.

    A child that fails to build at the instant (a scale that refuses its
    brackets, say) fails when it is read, not when the node is built, so the
    node's other children can be read. A deep copy or pickle of the node
    builds every child, so it fails then too.

    Several threads can read a node, and copy or pickle it while others read.
    A deep copy or pickle of a node resolves every child first, so it holds
    the same values as when every child was built up front, and no parameter
    nodes. Its pickle holds this version's lazy state, which earlier versions
    of core can't load. At an instant that is not a date, and in a subclass,
    every child is built up front, as before.
    """

    def __init__(self, name: str, node: "ParameterNode", instant_str: str):
        """
        :param name: Name of the node.
        :param node: Original :any:`ParameterNode` instance.
        :param instant_str: A date in the format `YYYY-MM-DD`.
        """

        # The "technical" attributes are hidden, so that the node children can be easily browsed with auto-completion without pollution
        self._name = name
        self._instant_str = instant_str
        source = dict(node.children.items())
        if _STATE in source:
            raise ValueError(f"{_STATE!r} cannot name a parameter.")
        state = _LazyChildren(instant_str, source)
        self.__dict__[_STATE] = state
        cls = type(self)
        if cls is not ParameterNodeAtInstant:
            # A subclass gets the up-front build, as before: its add_child
            # sees every child, every child is an attribute before its own
            # ``__getattr__`` or ``__getitem__`` runs, and ``_children`` is
            # also the mapping items are read from.
            children = {}
            with _RESOLUTION_LOCK:
                state.source = None
                state.children = children
            self.__dict__["_children"] = children
            for child_name, child in source.items():
                child_at_instant = child._get_at_instant(instant_str)
                if child_at_instant is not None:
                    self.add_child(child_name, child_at_instant)
            return
        for child_name in source:
            # A child whose name the node itself answers to shadows it, as
            # when every child was set here.
            if child_name in _OWN_NAMES or hasattr(cls, child_name):
                child_at_instant = _resolve(state, child_name)
                if child_at_instant is not _MISSING:
                    setattr(self, child_name, child_at_instant)
        if not _is_date(instant_str):
            # Built up front, a node at an instant that is not a date (such as
            # ``parameters(None)``, which reads "None") failed on the first
            # scale below it, and nothing was cached. Building it up front
            # still does that, and otherwise holds what that build held.
            _resolve_remaining(state)

    def add_child(self, child_name: str, child_at_instant: "ParameterNodeAtInstant"):
        if child_name == _STATE:
            raise ValueError(f"{_STATE!r} cannot name a parameter.")
        state = self.__dict__[_STATE]
        with _RESOLUTION_LOCK:
            state.added[child_name] = child_at_instant
            if state.children is not None:
                state.children[child_name] = child_at_instant
        setattr(self, child_name, child_at_instant)

    def __getattr__(self, key: str):
        # Reached only for a name that is not an instance attribute: a child
        # not read yet by this instance, ``_children`` before it is set, or a
        # name this node does not have.
        state = self.__dict__.get(_STATE)
        if state is None or key == "_name":
            # An instance still being rebuilt by copy or pickle.
            raise AttributeError(key)
        if key == "_children":
            return _materialise(self, state)
        child_at_instant = _resolve(state, key)
        if child_at_instant is not _MISSING:
            # Attributes hold what the parameters give; ``add_child`` on
            # another copy changes only that copy's attribute, as before.
            self.__dict__.setdefault(key, child_at_instant)
            return child_at_instant
        param_name = helpers._compose_name(self._name, item_name=key)
        raise ParameterNotFoundError(param_name, self._instant_str)

    def __dir__(self):
        children = self._children
        names = set(super().__dir__()) | set(map(str, children))
        names.discard(_STATE)
        return list(names)

    def __getstate__(self):
        # Reads resolve children into ``__dict__``: see ``_copied_state``.
        return _copied_state(self)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            state, slots = state
            for slot_name, value in slots.items():
                setattr(self, slot_name, value)
        state = state or {}
        self.__dict__.update(state)
        if _STATE not in state:
            # Pickled when every child was built up front, all in ``_children``.
            children = state.get("_children")
            if children is None:
                children = self.__dict__["_children"] = {}
            lazy = _LazyChildren(state.get("_instant_str"), None)
            lazy.children = children
            lazy.resolved = dict(children)
            self.__dict__[_STATE] = lazy

    def __getitem__(
        self, key: str
    ) -> Union["ParameterNodeAtInstant", VectorialParameterNodeAtInstant]:
        # If fancy indexing is used, cast to a vectorial node
        # Convert pandas arrays (e.g., StringArray from pandas 3) to numpy
        # before checking, since StringArray has __array__ but is not hashable
        if hasattr(key, "__array__") and not isinstance(key, numpy.ndarray):
            key = numpy.asarray(key)
        if isinstance(key, numpy.ndarray):
            # Cache the vectorial node to avoid rebuilding the recarray on
            # every call -- build_from_node is expensive (walks the full
            # parameter subtree each time).
            try:
                vectorial = self._vectorial_node
            except AttributeError:
                vectorial = parameters.VectorialParameterNodeAtInstant.build_from_node(
                    self
                )
                self._vectorial_node = vectorial
            return vectorial[key]
        state = self.__dict__[_STATE]
        children = state.children
        if children is not None:
            # Every child is resolved: the items are ``_children``'s.
            return children[key]
        child_at_instant = state.added.get(key, _MISSING)
        if child_at_instant is _MISSING:
            child_at_instant = state.resolved.get(key, _MISSING)
            if child_at_instant is _MISSING:
                child_at_instant = _resolve(state, key)
                if child_at_instant is _MISSING:
                    raise KeyError(key)
        return child_at_instant

    def __iter__(self) -> Iterable:
        return iter(self._children)

    def __repr__(self) -> str:
        result = os.linesep.join(
            [
                os.linesep.join(["{}:", "{}"]).format(name, tools.indent(repr(value)))
                for name, value in self._children.items()
            ]
        )
        return result


def _resolve(state: _LazyChildren, child_name):
    """A child as the parameters give it, resolving it if no read has yet;
    ``_MISSING`` if the node has no such child or it has no value at the
    instant."""
    with _RESOLUTION_LOCK:
        child_at_instant = state.resolved.get(child_name, _MISSING)
        if child_at_instant is not _MISSING or child_name in state.absent:
            return child_at_instant
        source = state.source
        child = None if source is None else source.get(child_name)
        if child is None:
            return _MISSING
        child_at_instant = child._get_at_instant(state.instant)
        # A read made while resolving (the same thread re-entering) may have
        # recorded this child already; the first recorded result stands.
        if child_name in state.absent:
            return _MISSING
        recorded = state.resolved.get(child_name, _MISSING)
        if recorded is not _MISSING:
            child_at_instant = recorded
        elif child_at_instant is None:
            state.absent.add(child_name)
        else:
            state.resolved[child_name] = child_at_instant
        if state.source is not None and len(state.resolved) + len(state.absent) == len(
            state.source
        ):
            _complete(state)
        return _MISSING if child_at_instant is None else child_at_instant


def _complete(state: _LazyChildren):
    """Order every child once all of ``source`` is resolved; let go of the
    parameter nodes. The caller holds ``_RESOLUTION_LOCK``."""
    resolved, added = state.resolved, state.added
    children = {
        name: added.get(name, resolved[name])
        for name in state.source
        if name in resolved
    }
    for name, child_at_instant in added.items():
        # Added children the parameters do not have (or have no value for)
        # follow, in the order they were added.
        children.setdefault(name, child_at_instant)
    # ``resolved`` stays apart from ``children``: attribute reads return what
    # the parameters give, whatever ``add_child`` or a direct write does to
    # the items.
    state.children = children
    state.source = None
    state.absent = set()


def _resolve_remaining(state: _LazyChildren):
    """Resolve every child not read yet, and order every child."""
    with _RESOLUTION_LOCK:
        if state.children is None:
            for child_name in list(state.source):
                if state.children is not None:
                    break
                _resolve(state, child_name)
            if state.children is None:
                _complete(state)


def _materialise(node_at_instant: ParameterNodeAtInstant, state: _LazyChildren):
    """Resolve every child and set ``_children``."""
    with _RESOLUTION_LOCK:
        _resolve_remaining(state)
        children = state.children
    if "_children" not in node_at_instant.__dict__:
        # Unless a child named ``_children`` took the name.
        node_at_instant.__dict__["_children"] = children
    return children
