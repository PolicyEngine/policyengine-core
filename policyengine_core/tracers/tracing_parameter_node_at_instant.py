from __future__ import annotations

import typing
from collections.abc import Iterator, Mapping
from typing import Union

import numpy

from policyengine_core import parameters
from policyengine_core.taxscales import TaxScaleLike

from .. import tracers

if typing.TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from policyengine_core.parameters import (
        ParameterNodeAtInstant,
        VectorialParameterNodeAtInstant,
    )

    ParameterNode = Union[ParameterNodeAtInstant, VectorialParameterNodeAtInstant]

    Child = Union[ParameterNode, ArrayLike]


class TracingParameterNodeAtInstant:
    def __init__(
        self,
        parameter_node_at_instant: ParameterNode,
        tracer: tracers.FullTracer,
        branch_name: str,
    ) -> None:
        self.parameter_node_at_instant = parameter_node_at_instant
        # Captured once: a wrapper belongs to the formula that obtained it.
        # The parameter tree's own tracer and branch name move while a
        # nested branch calculation runs; reading them at access time would
        # label this formula's later reads with that branch (#543 review).
        self.tracer = tracer
        self.branch_name = branch_name

    def __getattr__(
        self,
        key: str,
    ) -> Union[TracingParameterNodeAtInstant, Child]:
        if key == "_children":
            # A formula reading the node's structure (iterating child names,
            # picking a child by computed key). Keep it traced rather than
            # handing out the raw dict of unwrapped nodes.
            return TracingChildren(self)
        child = getattr(self.parameter_node_at_instant, key)
        return self.get_traced_child(child, key)

    def _child_names(self) -> list[str]:
        return list(self.parameter_node_at_instant._children)

    def _record_structure_read(self) -> None:
        """A read of which children a node has: recorded at the node's own
        path, with the child names as its value."""
        self.tracer.record_parameter_access(
            self.parameter_node_at_instant._name,
            self.parameter_node_at_instant._instant_str,
            self.branch_name,
            self._child_names(),
        )

    def __iter__(self) -> Iterator[str]:
        self._record_structure_read()
        return iter(self._child_names())

    def __contains__(self, key: object) -> bool:
        self._record_structure_read()
        return key in self.parameter_node_at_instant._children

    def __getitem__(
        self,
        key: str,
    ) -> Union[TracingParameterNodeAtInstant, Child]:
        child = self.parameter_node_at_instant[key]
        return self.get_traced_child(child, key)

    def get_traced_child(
        self,
        child: Child,
        key: Union[str, ArrayLike],
    ) -> Union[TracingParameterNodeAtInstant, Child]:
        period: str = self.parameter_node_at_instant._instant_str

        if isinstance(
            child,
            (
                parameters.ParameterNodeAtInstant,
                parameters.VectorialParameterNodeAtInstant,
            ),
        ):
            return TracingParameterNodeAtInstant(child, self.tracer, self.branch_name)

        if not isinstance(key, str) or isinstance(
            self.parameter_node_at_instant,
            parameters.VectorialParameterNodeAtInstant,
        ):
            # In case of vectorization, we keep the parent node name as, for
            # instance, rate[status].zone1 is best described as the value of
            # "rate".
            name = self.parameter_node_at_instant._name

        else:
            name = ".".join([self.parameter_node_at_instant._name, key])

        if isinstance(child, (numpy.ndarray,) + parameters.ALLOWED_PARAM_TYPES):
            self.tracer.record_parameter_access(name, period, self.branch_name, child)
        elif isinstance(child, TaxScaleLike):
            # A scale (read through .calc() or [index]) is a parameter read
            # too; record its brackets, JSON-safe, so a serialized trace
            # keeps the schedule and not just its path.
            self.tracer.record_parameter_access(
                name, period, self.branch_name, describe_tax_scale(child)
            )
        else:
            self.tracer.record_parameter_access(name, period, self.branch_name, None)

        return child


def describe_tax_scale(scale: TaxScaleLike) -> dict:
    """The schedule as plain lists: its class, thresholds, and the rates
    or amounts it applies. Values are converted so the flat trace can be
    serialized without a NumPy-aware encoder."""
    description: dict = {"type": type(scale).__name__}
    for attribute in ("thresholds", "rates", "amounts"):
        values = getattr(scale, attribute, None)
        if values is not None:
            description[attribute] = [_plain(v) for v in values]
    return description


def _plain(value):
    if isinstance(value, numpy.generic):
        return value.item()
    return value


class TracingChildren(Mapping):
    """The child mapping of a traced node.

    Iterating, counting, or testing membership records a read of the
    node's structure at the node's canonical path (never a path ending in
    ``._children``, which nothing can resolve). Looking a child up returns
    it through the parent's tracing, so nested nodes stay wrapped and
    their later reads are recorded.
    """

    def __init__(self, parent: TracingParameterNodeAtInstant) -> None:
        self._parent = parent

    def __getitem__(self, key: str):
        child = self._parent.parameter_node_at_instant._children[key]
        return self._parent.get_traced_child(child, key)

    def __iter__(self) -> Iterator[str]:
        self._parent._record_structure_read()
        return iter(self._parent._child_names())

    def __len__(self) -> int:
        self._parent._record_structure_read()
        return len(self._parent._child_names())

    def __contains__(self, key: object) -> bool:
        self._parent._record_structure_read()
        return key in self._parent.parameter_node_at_instant._children
