from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from threading import RLock
from typing import TYPE_CHECKING, Generic, Iterable, TypeVar, Union

import numpy

from policyengine_core.errors import ParameterNotFoundError

from . import helpers
from .parameter_node_at_instant import ParameterNodeAtInstant
from .parameter_revision import ParameterTreeRevision
from .vectorial_parameter_node_at_instant import VectorialParameterNodeAtInstant

if TYPE_CHECKING:
    from .parameter_node import ParameterNode

SourceT = TypeVar("SourceT")
InstantT = TypeVar("InstantT")
ViewT = TypeVar("ViewT")


class ParameterMaterializer(ABC, Generic[SourceT, InstantT, ViewT]):
    """Construct a dated view from a parameter source under one revision."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the stable public name of this strategy."""

    @abstractmethod
    def materialize(
        self,
        source: SourceT,
        instant: InstantT,
        revision: ParameterTreeRevision,
    ) -> ViewT:
        """Return a dated view bound to ``revision.current``."""


class EagerParameterMaterializer(
    ParameterMaterializer["ParameterNode", str, ParameterNodeAtInstant]
):
    """Construct the complete dated parameter subtree immediately."""

    @property
    def strategy_name(self) -> str:
        return "eager"

    def materialize(
        self,
        source: ParameterNode,
        instant: str,
        revision: ParameterTreeRevision,
    ) -> ParameterNodeAtInstant:
        captured = revision.current
        view = ParameterNodeAtInstant(source.name, source, instant)
        revision.validate(captured)
        return view


class LazyParameterNodeAtInstant(ParameterNodeAtInstant):
    """A dated node that resolves each requested child on first access.

    The view captures the source tree's revision. Every operation that can
    expose parameter data validates that revision before returning data.
    """

    _PROTOCOL_ATTRIBUTES = frozenset(
        {
            "__copy__",
            "__deepcopy__",
            "__getnewargs__",
            "__getnewargs_ex__",
            "__getstate__",
            "__reduce__",
            "__reduce_ex__",
            "__setstate__",
            "__slots__",
        }
    )

    def __init__(
        self,
        source: ParameterNode,
        instant: str,
        revision: ParameterTreeRevision,
    ) -> None:
        self.__source = source
        self.__instant = instant
        self.__revision = revision
        self.__captured_revision = revision.current
        self.__resolved_children: dict[str, object] = {}
        self.__missing_children: set[str] = set()
        self.__vectorial_node: VectorialParameterNodeAtInstant | None = None
        self.__lock = RLock()

    @property
    def is_materialized(self) -> bool:
        """Whether at least one immediate child has been resolved."""

        self._validate_revision()
        with self.__lock:
            return bool(self.__resolved_children or self.__missing_children)

    @property
    def materialized_child_count(self) -> int:
        self._validate_revision()
        with self.__lock:
            return len(self.__resolved_children)

    @property
    def _name(self) -> str:
        self._validate_revision()
        return self.__source.name

    @property
    def _instant_str(self) -> str:
        self._validate_revision()
        return self.__instant

    @property
    def _children(self) -> dict[str, object]:
        self._materialize_all_children()
        return self.__resolved_children

    def _validate_revision(self) -> None:
        self.__revision.validate(self.__captured_revision)

    def _missing_parameter(self, key: str) -> ParameterNotFoundError:
        name = helpers._compose_name(self.__source.name, item_name=key)
        return ParameterNotFoundError(name, self.__instant)

    def _materialize_child(self, key: str) -> object:
        self._validate_revision()
        with self.__lock:
            if key in self.__resolved_children:
                return self.__resolved_children[key]
            if key in self.__missing_children:
                raise self._missing_parameter(key)
            source_child = self.__source.children.get(key)
            if source_child is None:
                self.__missing_children.add(key)
                raise self._missing_parameter(key)

            from .parameter_node import ParameterNode

            if isinstance(source_child, ParameterNode):
                child = source_child.get_plain_at_instant(self.__instant)
            else:
                child = source_child._get_at_instant(self.__instant)
            self._validate_revision()
            if child is None:
                self.__missing_children.add(key)
                raise self._missing_parameter(key)
            self.__resolved_children[key] = child
            return child

    def _materialize_all_children(self) -> None:
        self._validate_revision()
        for key in tuple(self.__source.children):
            try:
                self._materialize_child(key)
            except ParameterNotFoundError:
                pass
        self._validate_revision()

    def __getattr__(self, key: str) -> object:
        if key in self._PROTOCOL_ATTRIBUTES:
            raise AttributeError(key)
        return self._materialize_child(key)

    def __getitem__(
        self, key: str
    ) -> Union[ParameterNodeAtInstant, VectorialParameterNodeAtInstant, object]:
        self._validate_revision()
        if hasattr(key, "__array__") and not isinstance(key, numpy.ndarray):
            key = numpy.asarray(key)
        if isinstance(key, numpy.ndarray):
            with self.__lock:
                if self.__vectorial_node is None:
                    self.__vectorial_node = (
                        VectorialParameterNodeAtInstant.build_from_node(self)
                    )
                vectorial = self.__vectorial_node
            self._validate_revision()
            return vectorial[key]
        return self._materialize_child(key)

    def __iter__(self) -> Iterable[str]:
        self._materialize_all_children()
        return iter(tuple(self.__resolved_children))

    def __repr__(self) -> str:
        self._validate_revision()
        self._materialize_all_children()
        return ParameterNodeAtInstant.__repr__(self)

    def __copy__(self) -> LazyParameterNodeAtInstant:
        self._validate_revision()
        duplicate = type(self)(self.__source, self.__instant, self.__revision)
        with self.__lock:
            duplicate.__resolved_children = self.__resolved_children.copy()
            duplicate.__missing_children = self.__missing_children.copy()
            duplicate.__vectorial_node = self.__vectorial_node
        return duplicate

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> LazyParameterNodeAtInstant:
        self._validate_revision()
        source = copy.deepcopy(self.__source, memo)
        revision = copy.deepcopy(self.__revision, memo)
        duplicate = type(self)(source, self.__instant, revision)
        memo[id(self)] = duplicate
        with self.__lock:
            duplicate.__resolved_children = copy.deepcopy(
                self.__resolved_children, memo
            )
            duplicate.__missing_children = self.__missing_children.copy()
            duplicate.__vectorial_node = copy.deepcopy(self.__vectorial_node, memo)
        return duplicate

    def __getstate__(self) -> dict[str, object]:
        self._validate_revision()
        with self.__lock:
            return {
                "source": self.__source,
                "instant": self.__instant,
                "revision": self.__revision,
                "resolved_children": self.__resolved_children,
                "missing_children": self.__missing_children,
                "vectorial_node": self.__vectorial_node,
            }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__init__(
            state["source"],
            state["instant"],
            state["revision"],
        )
        self.__resolved_children = state["resolved_children"]
        self.__missing_children = state["missing_children"]
        self.__vectorial_node = state["vectorial_node"]


class LazyParameterMaterializer(
    ParameterMaterializer["ParameterNode", str, LazyParameterNodeAtInstant]
):
    """Construct revision-bound nodes that resolve children on demand."""

    @property
    def strategy_name(self) -> str:
        return "lazy"

    def materialize(
        self,
        source: ParameterNode,
        instant: str,
        revision: ParameterTreeRevision,
    ) -> LazyParameterNodeAtInstant:
        return LazyParameterNodeAtInstant(source, instant, revision)
