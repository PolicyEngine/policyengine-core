from __future__ import annotations

import typing
from typing import Union

import numpy

from policyengine_core import parameters

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
        tracing_root=None,
    ) -> None:
        self.parameter_node_at_instant = parameter_node_at_instant
        self._tracer = tracer
        self._branch_name = branch_name
        # A cached wrapper outlives the simulation that built it: branch
        # simulations share the parameter tree and swap tracers and branch
        # names per formula. Reading them from the tree's root at access
        # time keeps the cache valid across those swaps.
        self._tracing_root = tracing_root

    @property
    def tracer(self) -> tracers.FullTracer:
        if self._tracing_root is not None:
            return self._tracing_root.tracer
        return self._tracer

    @property
    def branch_name(self) -> str:
        if self._tracing_root is not None:
            return self._tracing_root.branch_name
        return self._branch_name

    def __getattr__(
        self,
        key: str,
    ) -> Union[TracingParameterNodeAtInstant, Child]:
        child = getattr(self.parameter_node_at_instant, key)
        return self.get_traced_child(child, key)

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
            return TracingParameterNodeAtInstant(
                child, self._tracer, self._branch_name, self._tracing_root
            )

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
        else:
            # A scale or bracket (read through .calc() or [index]) is a
            # parameter read too; record it at its node with no scalar value.
            self.tracer.record_parameter_access(name, period, self.branch_name, None)

        return child
