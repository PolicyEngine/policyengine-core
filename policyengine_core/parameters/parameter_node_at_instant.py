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


# A node's own state, under name-mangled keys so that no parameter child can
# shadow it.
_NODE = "_ParameterNodeAtInstant__node"
_INSTANT = "_ParameterNodeAtInstant__instant"
_RESOLVED = "_ParameterNodeAtInstant__resolved"
_ABSENT = "_ParameterNodeAtInstant__absent"
_CHILDREN = "_ParameterNodeAtInstant__children"
_MISSING = object()

# Held while a child is resolved for the first time, so that concurrent
# readers never see a half-resolved node or resolve one child twice. Reads of
# a resolved child never take it. Reentrant because resolving a child can
# build that child's own node.
_RESOLUTION_LOCK = threading.RLock()


class ParameterNodeAtInstant:
    """
    Parameter node of the legislation, at a given instant.

    Children resolve on first read and are kept from then on. Building every
    descendant up front made each instant a model is asked about cost a copy of
    the whole parameter tree (16 to 33 MB and several seconds per instant for
    policyengine-us), which every tax-benefit system then kept for its
    lifetime. A child that has not been read yet takes the value its parameter
    has when it is first read; a child read once, including one found to have
    no value at this instant, keeps what that read found.
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
        state = self.__dict__
        # The parameter node children still resolve from; None once every
        # child is resolved.
        state[_NODE] = node
        state[_INSTANT] = instant_str
        # Children resolved so far. Only ever grows, and a shallow copy shares
        # it, so every copy reads the same child objects.
        state[_RESOLVED] = {}
        # Children of the node that have no value at this instant.
        state[_ABSENT] = set()
        # A child whose name the node itself uses (``add_child``, ``_name``,
        # ...) shadows that name, as when every child was set here.
        for child_name in _SHADOWING_NAMES:
            if child_name in node.children:
                _resolve(self, child_name)

    def add_child(self, child_name: str, child_at_instant: "ParameterNodeAtInstant"):
        with _RESOLUTION_LOCK:
            _store(self, child_name, child_at_instant)

    def __getattr__(self, key: str):
        # Reached only for a name that is not an instance attribute: a child
        # not read yet, ``_children`` before every child is resolved, or a name
        # this node does not have.
        if key.startswith("__") and key.endswith("__"):
            # Protocol lookups (copy, pickle, numpy) are never parameters.
            raise AttributeError(key)
        if _RESOLVED not in self.__dict__ or key == "_name":
            # An instance still being rebuilt by copy or pickle.
            raise AttributeError(key)
        if key == "_children":
            return _materialise(self)
        child_at_instant = _resolve(self, key)
        if child_at_instant is not _MISSING:
            return child_at_instant
        param_name = helpers._compose_name(self._name, item_name=key)
        raise ParameterNotFoundError(param_name, self._instant_str)

    def __dir__(self):
        self._children
        return [
            name
            for name in super().__dir__()
            if not name.startswith("_ParameterNodeAtInstant__")
        ]

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
        child_at_instant = self.__dict__[_RESOLVED].get(key, _MISSING)
        if child_at_instant is _MISSING:
            child_at_instant = _resolve(self, key)
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


def _store(node_at_instant, child_name, child_at_instant):
    """Record a child. The caller holds ``_RESOLUTION_LOCK``."""
    state = node_at_instant.__dict__
    state[_RESOLVED][child_name] = child_at_instant
    state[_ABSENT].discard(child_name)
    children = state.get(_CHILDREN)
    if children is not None:
        children[child_name] = child_at_instant
    setattr(node_at_instant, child_name, child_at_instant)


def _resolve(node_at_instant, child_name):
    """Return one child, resolving it if no read has yet; ``_MISSING`` if the
    node has no such child or it has no value at this instant."""
    state = node_at_instant.__dict__
    with _RESOLUTION_LOCK:
        resolved = state[_RESOLVED]
        if child_name in resolved:
            return resolved[child_name]
        if child_name in state[_ABSENT]:
            return _MISSING
        node = state[_NODE]
        child = None if node is None else node.children.get(child_name)
        if child is None:
            return _MISSING
        child_at_instant = child._get_at_instant(state[_INSTANT])
        if child_at_instant is None:
            state[_ABSENT].add(child_name)
        else:
            _store(node_at_instant, child_name, child_at_instant)
        if len(resolved) + len(state[_ABSENT]) >= len(node.children):
            # Every child has been read: drop the parameter node.
            _materialise(node_at_instant)
        return _MISSING if child_at_instant is None else child_at_instant


def _materialise(node_at_instant):
    """Resolve every child; return them in the parameter node's order, with
    any added by ``add_child`` after them."""
    state = node_at_instant.__dict__
    with _RESOLUTION_LOCK:
        children = state.get(_CHILDREN)
        if children is not None:
            return children
        node = state[_NODE]
        children = {}
        for child_name in node.children if node is not None else ():
            child_at_instant = _resolve(node_at_instant, child_name)
            if child_at_instant is not _MISSING:
                children[child_name] = child_at_instant
        if state.get(_CHILDREN) is not None:
            # Resolving the last child completed the node already.
            return state[_CHILDREN]
        for child_name, child_at_instant in state[_RESOLVED].items():
            children.setdefault(child_name, child_at_instant)
        state[_CHILDREN] = children
        if "_children" not in state:
            # Unless a child named ``_children`` took the name.
            state["_children"] = children
        # Every child is resolved: the parameter node is no longer needed.
        state[_NODE] = None
        return children


# Names a ParameterNodeAtInstant itself answers to, which a child of the same
# name shadows.
_SHADOWING_NAMES = frozenset(
    {"_name", "_instant_str", "_children", "_vectorial_node"}
    | {
        name
        for name in dir(ParameterNodeAtInstant)
        if not name.startswith("__") and not name.startswith("_ParameterNodeAtInstant")
    }
)
