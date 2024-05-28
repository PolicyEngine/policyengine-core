from __future__ import annotations

import typing
from typing import Optional, Union

import numpy

from policyengine_core.enums import EnumArray

from .. import tracers

if typing.TYPE_CHECKING:
    from numpy.typing import ArrayLike

    Array = Union[EnumArray, ArrayLike]


class VariableGraph:
    _full_tracer: tracers.FullTracer

    def __init__(self, full_tracer: tracers.FullTracer) -> None:
        self._full_tracer = full_tracer

    def tree(
        self,
        aggregate: bool = False,
        max_depth: Optional[int] = None,
    ) -> VisualizeNode:
        depth = 1

        node_by_tree = [
            self._get_node(node, depth, aggregate, max_depth)
            for node in self._full_tracer.trees
        ]

        return node_by_tree

    def visualize(self, aggregate=False, max_depth: Optional[int] = None) -> None:
        """
        Visualize the computation log of a simulation as a relationship graph in the web browser.

        If ``aggregate`` is ``False`` (default), visualize the value of each
        computed vector.

        If ``aggregate`` is ``True``, only the minimum, maximum, and
        average value will be used of each computed vector.

        This mode is more suited for simulations on a large population.

        If ``max_depth`` is ``None`` (default), visualize the entire computation.

        If ``max_depth`` is set, for example to ``3``, only visualize computed
        vectors up to a depth of ``max_depth``.
        """
        for tree in self.tree(aggregate, max_depth):
            print(tree.value)

    def _get_node(
        self,
        node: tracers.TraceNode,
        depth: int,
        aggregate: bool,
        max_depth: Optional[int],
    ) -> VisualizeNode:
        if max_depth is not None and depth > max_depth:
            return []

        children = [
            self._get_node(child, depth + 1, aggregate, max_depth)
            for child in node.children
        ]

        is_leaf = len(node.children) == 0
        visualization_node = VisualizeNode(
            node, children, is_leaf=is_leaf, aggregate=aggregate
        )

        return visualization_node


class VisualizeNode:
    def __init__(
        self,
        node: tracers.TraceNode,
        children: list[VisualizeNode],
        is_leaf=False,
        aggregate=False,
    ):
        self.node = node
        self.children = children
        self.is_leaf = is_leaf
        self.value = self._value(aggregate)

    def _display(
        self,
        value: Optional[Array],
    ) -> str:
        if isinstance(value, EnumArray):
            value = value.decode_to_str()

        return numpy.array2string(value, max_line_width=float("inf"))

    def _value(self, aggregate: bool) -> str:
        value = self.node.value

        if value is None:
            formatted_value = "{'avg': '?', 'max': '?', 'min': '?'}"

        elif aggregate:
            try:
                formatted_value = str(
                    {
                        "avg": numpy.mean(value),
                        "max": numpy.max(value),
                        "min": numpy.min(value),
                    }
                )

            except TypeError:
                formatted_value = "{'avg': '?', 'max': '?', 'min': '?'}"

        else:
            formatted_value = self._display(value)

        return f"{self.node.name}<{self.node.period}, ({self.node.branch_name})> = {formatted_value}"
