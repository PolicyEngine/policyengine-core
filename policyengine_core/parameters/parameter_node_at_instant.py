import os
import sys
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


# Attributes of the node itself. A lookup of one that is not set is an
# ordinary missing attribute, never a parameter.
_TECHNICAL_ATTRIBUTES = frozenset(
    {"_name", "_instant_str", "_node", "_resolved", "_children", "_vectorial_node"}
)


class ParameterNodeAtInstant:
    """
    Parameter node of the legislation, at a given instant.

    Children resolve on first read and are kept from then on. Building every
    descendant up front made each instant a model is asked about cost a copy of
    the whole parameter tree (16 to 33 MB and several seconds per instant for
    policyengine-us), which every tax-benefit system then kept for its
    lifetime. A child that has not been read yet takes the value its parameter
    has when it is read.
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
        # The node whose children are still to be resolved; ``None`` once
        # ``_children`` holds all of them.
        self._node = node
        # Children resolved so far, by name. Becomes ``_children`` itself once
        # every child is resolved.
        self._resolved = {}

    def _resolve(self, child_name: str):
        """Resolve one child, returning ``None`` if this node has no such
        child at its instant."""
        node = self._node
        child = None if node is None else node.children.get(child_name)
        if child is None:
            return None
        child_at_instant = child._get_at_instant(self._instant_str)
        if child_at_instant is not None:
            self.add_child(child_name, child_at_instant)
        return child_at_instant

    def _resolve_all(self) -> dict:
        """Resolve every child, and return them in the parameter node's order."""
        resolved = self._resolved
        children = {}
        for child_name, child in self._node.children.items():
            if child_name in resolved:
                children[child_name] = resolved[child_name]
                continue
            child_at_instant = child._get_at_instant(self._instant_str)
            if child_at_instant is not None:
                children[child_name] = child_at_instant
                if child_name not in _TECHNICAL_ATTRIBUTES:
                    setattr(self, child_name, child_at_instant)
        for child_name, child_at_instant in resolved.items():
            # Children added with ``add_child`` that the node does not have.
            children.setdefault(child_name, child_at_instant)
        self._children = self._resolved = children
        self._node = None
        return children

    def add_child(self, child_name: str, child_at_instant: "ParameterNodeAtInstant"):
        self._resolved[child_name] = child_at_instant
        if child_name not in _TECHNICAL_ATTRIBUTES:
            setattr(self, child_name, child_at_instant)

    def __getattr__(self, key: str):
        # Reached only for a name that is not an instance attribute yet: a
        # child that has not been read, ``_children`` before every child has
        # been resolved, or a name this node does not have.
        if key in _TECHNICAL_ATTRIBUTES:
            if key == "_children" and self.__dict__.get("_node") is not None:
                return self._resolve_all()
            raise AttributeError(key)
        if self.__dict__.get("_node") is not None:
            child_at_instant = self._resolve(key)
            if child_at_instant is not None:
                return child_at_instant
        param_name = helpers._compose_name(self._name, item_name=key)
        raise ParameterNotFoundError(param_name, self._instant_str)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(map(str, self._children)))

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
        resolved = self._resolved
        if key in resolved:
            return resolved[key]
        child_at_instant = self._resolve(key)
        if child_at_instant is None:
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
