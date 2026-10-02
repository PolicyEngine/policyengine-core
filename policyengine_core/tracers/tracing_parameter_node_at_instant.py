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


def _wrapped(wrapper: object, attribute: str, key: str) -> object:
    """Return the object ``wrapper`` delegates the lookup of ``key`` to.

    ``__getattr__`` only runs when normal lookup fails. Two such lookups must
    not reach the wrapped object:

    - special names, which ``copy``, ``deepcopy`` and ``pickle`` probe on an
      instance (``__deepcopy__``, ``__setstate__``, ...). The wrapped
      object's answer would act on the wrapped object, not on the wrapper;
    - any name on an instance those protocols have created with ``__new__``
      and not filled in yet. It has no ``attribute``, so reading it here
      would call ``__getattr__`` again, without end.
    """
    if key.startswith("__") and key.endswith("__"):
        raise AttributeError(key)
    try:
        return wrapper.__dict__[attribute]
    except KeyError:
        raise AttributeError(key) from None


class TracingParameterNode:
    """The parameter tree as the formulas of one traced simulation see it.

    Calling it at an instant returns a :class:`TracingParameterNodeAtInstant`
    that records every parameter a formula reads in ``tracer``, under
    ``branch_name``. Any other attribute is read from the wrapped node.

    The wrapped node is never modified, so tracing one simulation does not
    trace the tax-benefit system it shares with other simulations, branches
    and clones.
    """

    def __init__(
        self,
        parameter_node: parameters.ParameterNode,
        tracer: tracers.FullTracer,
        branch_name: str,
    ) -> None:
        self.parameter_node = parameter_node
        self.tracer = tracer
        self.branch_name = branch_name

    def __call__(self, instant) -> TracingParameterNodeAtInstant:
        return self.get_at_instant(instant)

    def get_at_instant(self, instant) -> TracingParameterNodeAtInstant:
        node_at_instant = self.parameter_node.get_at_instant(instant)
        if isinstance(node_at_instant, TracingParameterNodeAtInstant):
            # The node traces by itself (its ``trace`` flag is set): record
            # in this simulation's tracer, not in the one the node holds.
            node_at_instant = node_at_instant.parameter_node_at_instant
        return TracingParameterNodeAtInstant(
            node_at_instant, self.tracer, self.branch_name
        )

    def __getattr__(self, key: str):
        return getattr(_wrapped(self, "parameter_node", key), key)


class TracingParameterNodeAtInstant:
    def __init__(
        self,
        parameter_node_at_instant: ParameterNode,
        tracer: tracers.FullTracer,
        branch_name: str,
    ) -> None:
        self.parameter_node_at_instant = parameter_node_at_instant
        self.tracer = tracer
        self.branch_name = branch_name

    def __getattr__(
        self,
        key: str,
    ) -> Union[TracingParameterNodeAtInstant, Child]:
        child = getattr(_wrapped(self, "parameter_node_at_instant", key), key)
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

        return child
