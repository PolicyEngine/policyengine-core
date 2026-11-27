from __future__ import annotations

import typing
from collections.abc import Iterator, Mapping
from typing import Optional, Union

import numpy

from policyengine_core import parameters, periods
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


# Attributes of an undated parameter that describe it and hold no values.
_UNDATED_LABELS = frozenset({"name", "description", "documentation", "file_path"})


def describe_tax_scale(scale: TaxScaleLike) -> dict[str, object]:
    """Describe a tax scale in plain values a trace can serialize as JSON.

    The description holds the scale's class name and its thresholds, and its
    rates or amounts when it has them.
    """
    description: dict[str, object] = {"type": type(scale).__name__}
    for attribute in ("thresholds", "rates", "amounts"):
        values = getattr(scale, attribute, None)
        if values is not None:
            description[attribute] = [_plain_number(value) for value in values]
    return description


def _plain_number(value: object) -> object:
    return value.item() if isinstance(value, numpy.generic) else value


def _record_value_read(
    tracer: tracers.FullTracer,
    name: str,
    instant_str: str,
    branch_name: str,
    value: object,
) -> None:
    """Record a read of a parameter value or a tax scale.

    Any other object, such as a method or a node's label, is not a parameter
    value and is not recorded.
    """
    if isinstance(value, TaxScaleLike):
        value = describe_tax_scale(value)
    elif not isinstance(value, (numpy.ndarray,) + parameters.ALLOWED_PARAM_TYPES):
        return
    tracer.record_parameter_access(name, instant_str, branch_name, value)


class TracingParameterNode:
    """The parameter tree as the formulas of one traced simulation see it.

    Calling it at an instant records the read in ``tracer``, under
    ``branch_name``. A parameter or a tax scale returns its value; a node
    returns a :class:`TracingParameterNodeAtInstant`, which records the reads
    made through it.

    Children read before an instant is chosen, as in
    ``parameters.gov.rate(period)``, are returned as tracing views too. Any
    other attribute, such as ``values_list``, ``children`` or ``clone``, is
    read from the wrapped node. What the formula does with it cannot be
    followed, so the read is recorded at this node's path for all instants
    (period ``eternity``, no value); it stands for everything below the node.

    The wrapped node is never modified, so tracing one simulation does not
    trace the tax-benefit system it shares with other simulations, branches
    and clones. Like :class:`TracingParameterNodeAtInstant`, it is not an
    instance of the class it wraps.
    """

    def __init__(
        self,
        parameter_node: parameters.AtInstantLike,
        tracer: tracers.FullTracer,
        branch_name: str,
    ) -> None:
        self.parameter_node = parameter_node
        self.tracer = tracer
        self.branch_name = branch_name

    def __call__(self, instant) -> object:
        return self.get_at_instant(instant)

    def get_at_instant(self, instant) -> object:
        value_at_instant = self.parameter_node.get_at_instant(instant)
        if isinstance(value_at_instant, TracingParameterNodeAtInstant):
            # The node traces by itself (its ``trace`` flag is set): record
            # in this simulation's tracer, not in the one the node holds.
            value_at_instant = value_at_instant.parameter_node_at_instant
        if isinstance(value_at_instant, _node_at_instant_types()):
            return TracingParameterNodeAtInstant(
                value_at_instant, self.tracer, self.branch_name
            )
        _record_value_read(
            self.tracer,
            self.parameter_node.name,
            str(periods.instant(instant)),
            self.branch_name,
            value_at_instant,
        )
        return value_at_instant

    def _traced_child(self, child: object) -> object:
        if isinstance(child, parameters.AtInstantLike):
            return TracingParameterNode(child, self.tracer, self.branch_name)
        return child

    def _record_read_for_all_instants(self, key: str) -> None:
        if key.startswith("__") or key in _UNDATED_LABELS:
            return
        self.tracer.record_parameter_access(
            self.parameter_node.name, periods.ETERNITY, self.branch_name, None
        )

    def __getattr__(self, key: str):
        parameter_node = _wrapped(self, "parameter_node", key)
        try:
            attribute = getattr(parameter_node, key)
        except AttributeError:
            self._record_read_for_all_instants(key)
            raise
        if isinstance(attribute, parameters.AtInstantLike):
            return self._traced_child(attribute)
        self._record_read_for_all_instants(key)
        return attribute

    def __getitem__(self, key: object) -> object:
        # A tax scale gives its brackets by position.
        return self._traced_child(self.parameter_node[key])

    def get_child(self, path: str) -> object:
        return self._traced_child(self.parameter_node.get_child(path))

    def __repr__(self) -> str:
        return repr(self.parameter_node)


def _node_at_instant_types() -> tuple[type, ...]:
    # Read when called: ``parameters`` imports this module while it loads.
    return (
        parameters.ParameterNodeAtInstant,
        parameters.VectorialParameterNodeAtInstant,
    )


def _child_path(parent_path: str, key: str) -> str:
    """The path of the child ``key`` of the node at ``parent_path``.

    The root of the tree has the empty path, so its children's paths are their
    keys. Joining uses the key's string value, which ``str()`` of a ``str``
    based enum member is not.
    """
    return ".".join([parent_path, key]) if parent_path else str.__str__(key)


class TracingParameterNodeAtInstant:
    """A parameter node at one instant, as the formulas of a traced simulation
    see it.

    Each parameter value or tax scale read through it is recorded in
    ``tracer``, under ``branch_name``, at its own path. A child node is
    returned as another tracing view.

    A look-up by name that finds no child, by attribute (``node.rate``,
    ``getattr``, ``hasattr``) or by key (``node["rate"]``), is recorded too,
    at the missing child's path with no value, because a formula can fall
    back on a default and a value added there would change its result.

    These are not recorded when they find nothing: a look-up by a dotted
    name, which never names a child, a look-up by name on a vectorial node, a
    look-up by an array that names a missing child, and,
    on the undated tree, ``get_child`` and a bracket position past the end of
    a scale.
    """

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
        node_at_instant = _wrapped(self, "parameter_node_at_instant", key)
        if key == "_children" and isinstance(
            node_at_instant, parameters.ParameterNodeAtInstant
        ):
            return TracingChildren(self)
        try:
            child = getattr(node_at_instant, key)
        except AttributeError:
            self._record_absent_child_read(key)
            raise
        return self.get_traced_child(child, key)

    def __getitem__(
        self,
        key: str,
    ) -> Union[TracingParameterNodeAtInstant, Child]:
        try:
            child = self.parameter_node_at_instant[key]
        except KeyError:
            self._record_missing_key_read(key)
            raise
        return self.get_traced_child(child, key)

    def __iter__(self) -> Iterator:
        # Without it, ``iter`` and ``in`` fall back to ``__getitem__(0)``.
        if isinstance(
            self.parameter_node_at_instant, parameters.ParameterNodeAtInstant
        ):
            return iter(self._record_children_read())
        return iter(self.parameter_node_at_instant)

    def __repr__(self) -> str:
        return repr(self.parameter_node_at_instant)

    def _record_absent_child_read(self, key: str) -> None:
        """Record an attribute look-up that found no child, at that child's
        path, as a key look-up is recorded.

        Special names such as ``__array__`` are not children and are not
        recorded. The node's other children are not listed: a node at an
        instant builds each child when it is first read, and listing them
        would build every one, so a child that fails to build would stop a
        traced formula that never reads it.
        """
        if not key.startswith("__"):
            self._record_missing_key_read(key)

    def _record_missing_key_read(self, key: object) -> None:
        """Record a look-up by name that found no child, at that child's path.

        The read holds no value. A key that is not a string, such as an array
        for a vectorial read, names no single child and is not recorded.
        Nor is a dotted name: the look-up does not descend into it, and
        ``ParameterNode.add_child`` asks that a child's name hold no dot, so
        it finds nothing whatever the parameters hold. Its path can also be
        a real parameter's, whose recorded value it would replace.
        """
        node_at_instant = self.parameter_node_at_instant
        if (
            isinstance(key, str)
            and "." not in key
            and isinstance(node_at_instant, parameters.ParameterNodeAtInstant)
        ):
            self.tracer.record_parameter_access(
                _child_path(node_at_instant._name, key),
                node_at_instant._instant_str,
                self.branch_name,
                None,
            )

    def _record_children_read(self) -> list[str]:
        """Record that the formula read which children this node has.

        The read is recorded at the node's own path, with the child names as
        its value, and the names are returned.
        """
        node_at_instant = self.parameter_node_at_instant
        child_names = list(node_at_instant._children)
        self.tracer.record_parameter_access(
            node_at_instant._name,
            node_at_instant._instant_str,
            self.branch_name,
            child_names,
        )
        return child_names

    def get_traced_child(
        self,
        child: Child,
        key: Union[str, ArrayLike],
    ) -> Union[TracingParameterNodeAtInstant, Child]:
        period: str = self.parameter_node_at_instant._instant_str

        if isinstance(child, _node_at_instant_types()):
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

        _record_value_read(self.tracer, name, period, self.branch_name, child)

        return child


class TracingChildren(Mapping):
    """The ``_children`` mapping of a traced node.

    Reading which children exist (iterating, counting, testing membership or
    asking for a missing child) records one read at the node's own path, for
    this mapping. Taking a child returns it through the node's tracing, so
    the child's value is recorded at its own path and a child node stays
    traced.
    """

    def __init__(self, parent: TracingParameterNodeAtInstant) -> None:
        self._parent = parent
        self._child_names: Optional[list[str]] = None

    def _names_read(self) -> list[str]:
        if self._child_names is None:
            self._child_names = self._parent._record_children_read()
        return self._child_names

    def __getitem__(self, key: str) -> object:
        children = self._parent.parameter_node_at_instant._children
        if key not in children:
            self._names_read()
            raise KeyError(key)
        return self._parent.get_traced_child(children[key], key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names_read())

    def __len__(self) -> int:
        return len(self._names_read())

    def __contains__(self, key: object) -> bool:
        return key in self._names_read()
