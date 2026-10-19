from __future__ import annotations

import typing
from typing import Iterator, Union

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


# Copying and pickling look these names up on an instance. Answered by the
# wrapped object, they would copy or restore the wrapped object, not the
# wrapper. ``__slots__`` describes the wrapper's own layout: pickle protocols
# 0 and 1 refuse an instance that reports slots but no ``__getstate__``.
_COPY_PROTOCOL = frozenset(
    {
        "__slots__",
        "__copy__",
        "__deepcopy__",
        "__getstate__",
        "__setstate__",
        "__reduce__",
        "__reduce_ex__",
        "__getnewargs__",
        "__getnewargs_ex__",
    }
)


def _wrapped(wrapper: object, attribute: str, key: str) -> object:
    """Return the object ``wrapper`` delegates the lookup of ``key`` to.

    ``__getattr__`` only runs when normal lookup fails. Two such lookups must
    not reach the wrapped object:

    - the copy and pickle protocol (``__deepcopy__``, ``__setstate__``,
      ``__slots__``, ...);
    - any name on an instance ``copy`` or ``pickle`` has created with
      ``__new__`` and not filled in yet. It has no ``attribute``, so reading
      it here would call ``__getattr__`` again, without end.

    Every other name is delegated, special names included: NumPy reads its
    array protocol (``__array_interface__``, ...) from a vectorial node this
    way.
    """
    if key in _COPY_PROTOCOL:
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
    and clones. Like :class:`TracingParameterNodeAtInstant`, it is not an
    instance of the class it wraps.
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

    def __repr__(self) -> str:
        return repr(self.parameter_node)


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

    def __iter__(self) -> Iterator:
        # Without it, ``iter`` and ``in`` fall back to ``__getitem__(0)``.
        return iter(self.parameter_node_at_instant)

    def __repr__(self) -> str:
        return repr(self.parameter_node_at_instant)

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
