import os
import sys
import threading
from typing import TYPE_CHECKING, Iterable, Union

import numpy

from policyengine_core import parameters, tools
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
        source = dict(node.children)
        if _STATE in source:
            raise ValueError(f"{_STATE!r} cannot name a parameter.")
        state = _LazyChildren(instant_str, source)
        self.__dict__[_STATE] = state
        cls = type(self)
        if cls.add_child is not ParameterNodeAtInstant.add_child:
            # A subclass's add_child sees every child, as before.
            for child_name, child in source.items():
                child_at_instant = child._get_at_instant(instant_str)
                if child_at_instant is not None:
                    self.add_child(child_name, child_at_instant)
            with _RESOLUTION_LOCK:
                state.source = None
                state.children = dict(state.added)
            return
        for child_name in source:
            # A child whose name the node itself answers to shadows it, as
            # when every child was set here.
            if child_name in _OWN_NAMES or hasattr(cls, child_name):
                child_at_instant = _resolve(state, child_name)
                if child_at_instant is not _MISSING:
                    setattr(self, child_name, child_at_instant)

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

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        if _STATE not in state:
            # Pickled when every child was built up front, all in ``_children``.
            children = state.get("_children")
            if children is None:
                children = self.__dict__["_children"] = {}
            lazy = _LazyChildren(state.get("_instant_str"), None)
            lazy.resolved = lazy.children = children
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
    if not added:
        state.resolved = children
    state.children = children
    state.source = None
    state.absent = set()


def _materialise(node_at_instant: ParameterNodeAtInstant, state: _LazyChildren):
    """Resolve every child and set ``_children``."""
    with _RESOLUTION_LOCK:
        if state.children is None:
            for child_name in list(state.source):
                if state.children is not None:
                    break
                _resolve(state, child_name)
            if state.children is None:
                _complete(state)
        children = state.children
    if "_children" not in node_at_instant.__dict__:
        # Unless a child named ``_children`` took the name.
        node_at_instant.__dict__["_children"] = children
    return children
