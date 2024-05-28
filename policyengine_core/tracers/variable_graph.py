from __future__ import annotations

import os
import sys
import typing
from typing import Optional, Union

import numpy

from policyengine_core.enums import EnumArray

from .. import tracers

from pyvis.network import Network

if typing.TYPE_CHECKING:
    from numpy.typing import ArrayLike

    Array = Union[EnumArray, ArrayLike]


class VariableGraph:
    _full_tracer: tracers.FullTracer

    NETWORK_OPTIONS = """
        const options = {
            "physics": {
                "repulsion": {
                    "theta": 1,
                    "centralGravity": 0,
                    "springLength": 255,
                    "springConstant": 0.06,
                    "damping": 1,
                    "avoidOverlap": 1
                },
                "minVelocity": 0.75,
                "solver": "repulsion"
            }
        }
    """

    def __init__(self, full_tracer: tracers.FullTracer) -> None:
        self._full_tracer = full_tracer

    def tree(
        self,
        aggregate: bool = False,
        max_depth: Optional[int] = None,
        output_vars: list[str] = [],
    ) -> VisualizeNode:
        depth = 1
        is_root = True

        node_by_tree = [
            self._get_node(node, depth, aggregate, max_depth, output_vars)
            for node in self._full_tracer.trees
        ]

        return node_by_tree

    def visualize(
        self,
        name: str,
        output_vars: list[str] = [],
        aggregate=False,
        max_depth: Optional[int] = None,
    ) -> None:
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

        i = 0
        for root_node in self.tree(aggregate, max_depth, output_vars):
            net = self._network()
            self._add_nodes_and_edges(net, root_node)

            file_name = f"{self._to_snake_case(name)}.{i}.html"
            i += 1

            # redirect stdout to prevent net.show from printing the file name
            old_stdout = sys.stdout  # backup current stdout
            sys.stdout = open(os.devnull, "w")

            net.show(file_name, notebook=False)

            sys.stdout = old_stdout

    def _network(self) -> Network:
        net = Network(
            height="100vh", directed=True, select_menu=True, neighborhood_highlight=True
        )
        Network.set_options(net, self.NETWORK_OPTIONS)

        return net

    def _add_nodes_and_edges(self, net: Network, root_node: VisualizeNode):
        stack = [root_node]
        edges: set[tuple[str, str]] = set()

        while len(stack) > 0:
            node = stack.pop()

            net.add_node(node.name, color=node.color(), title=node.value)

            for child in node.children:
                edge = (node.name, child.name)
                edges.add(edge)
                stack.append(child)

        for parent, child in edges:
            net.add_edge(child, parent)

    def _get_node(
        self,
        node: tracers.TraceNode,
        depth: int,
        aggregate: bool,
        max_depth: Optional[int],
        output_vars: list[str],
    ) -> VisualizeNode:
        if max_depth is not None and depth > max_depth:
            return []

        children = [
            self._get_node(child, depth + 1, aggregate, max_depth, output_vars)
            for child in node.children
        ]

        is_leaf = len(node.children) == 0
        visualization_node = VisualizeNode(
            node,
            children,
            is_leaf=is_leaf,
            aggregate=aggregate,
            output_vars=output_vars,
        )

        return visualization_node

    def _to_snake_case(self, string: str):
        return string.replace(" ", "_").lower()


class VisualizeNode:
    DEFAULT_COLOR = "#BFD0DF"
    LEAF_COLOR = "#0099FF"
    ROOT_COLOR = "#7B61FF"

    def __init__(
        self,
        node: tracers.TraceNode,
        children: list[VisualizeNode],
        is_leaf=False,
        aggregate=False,
        output_vars: list[str] = [],
    ):
        self.node = node
        self.name = node.name
        self.children = children
        self.is_leaf = is_leaf
        self.is_root = self.name in output_vars
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

    def color(self) -> str:
        if self.is_root:
            return self.ROOT_COLOR
        if self.is_leaf:
            return self.LEAF_COLOR

        return self.DEFAULT_COLOR
