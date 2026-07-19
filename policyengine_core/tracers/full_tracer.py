from __future__ import annotations

import re
import time
import typing
from typing import Any, Dict, Iterator, List, Optional, Union

from .. import tracers

if typing.TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from policyengine_core.periods import Period

    Stack = List[Dict[str, Union[str, Period]]]


TRACE_FORMAT_V1 = "policyengine.trace.v1"
# Matches FlatTrace.key(): ``name<period, (branch)>``
_TRACE_KEY_RE = re.compile(
    r"^(?P<name>.+)<(?P<period>[^<>]+), \((?P<branch>[^()]*)\)>$"
)


def parse_trace_key(key: str) -> Dict[str, str]:
    """Parse a serialized flat-trace node key into name/period/branch."""
    match = _TRACE_KEY_RE.match(key)
    if not match:
        raise ValueError(f"Invalid PolicyEngine trace key: {key!r}")
    return {
        "name": match.group("name"),
        "period": match.group("period").strip(),
        "branch": match.group("branch").strip(),
    }


class FullTracer:
    _simple_tracer: tracers.SimpleTracer
    _trees: list
    _current_node: Optional[tracers.TraceNode]

    def __init__(self) -> None:
        self._simple_tracer = tracers.SimpleTracer()
        self._trees = []
        self._current_node = None

    def record_calculation_start(
        self,
        variable: str,
        period: str,
        branch_name: str = "default",
    ) -> None:
        self._simple_tracer.record_calculation_start(variable, period, branch_name)
        self._enter_calculation(variable, period, branch_name)
        self._record_start_time()

    def _enter_calculation(
        self,
        variable: str,
        period: str,
        branch_name: str = "default",
    ) -> None:
        new_node = tracers.TraceNode(
            name=variable,
            period=period,
            parent=self._current_node,
            branch_name=branch_name,
        )

        if self._current_node is None:
            self._trees.append(new_node)

        else:
            self._current_node.append_child(new_node)

        self._current_node = new_node

    def record_parameter_access(
        self,
        parameter: str,
        period: str,
        branch_name: str,
        value: ArrayLike,
    ) -> None:
        if self._current_node is not None:
            self._current_node.parameters.append(
                tracers.TraceNode(
                    name=parameter,
                    period=period,
                    branch_name=branch_name,
                    value=value,
                ),
            )

    def _record_start_time(
        self,
        time_in_s: Optional[float] = None,
    ) -> None:
        if time_in_s is None:
            time_in_s = self._get_time_in_sec()

        if self._current_node is not None:
            self._current_node.start = time_in_s

    def record_calculation_result(self, value: ArrayLike) -> None:
        if self._current_node is not None:
            self._current_node.value = value

    def record_calculation_end(self) -> None:
        self._simple_tracer.record_calculation_end()
        self._record_end_time()
        self._exit_calculation()

    def _record_end_time(
        self,
        time_in_s: Optional[float] = None,
    ) -> None:
        if time_in_s is None:
            time_in_s = self._get_time_in_sec()

        if self._current_node is not None:
            self._current_node.end = time_in_s

    def _exit_calculation(self) -> None:
        if self._current_node is not None:
            self._current_node = self._current_node.parent

    @property
    def stack(self) -> Stack:
        return self._simple_tracer.stack

    @property
    def trees(self) -> List[tracers.TraceNode]:
        return self._trees

    @property
    def computation_log(self) -> tracers.ComputationLog:
        return tracers.ComputationLog(self)

    @property
    def performance_log(self) -> tracers.PerformanceLog:
        return tracers.PerformanceLog(self)

    @property
    def variable_graph(self) -> tracers.VariableGraph:
        return tracers.VariableGraph(self)

    @property
    def flat_trace(self) -> tracers.FlatTrace:
        return tracers.FlatTrace(self)

    def _get_time_in_sec(self) -> float:
        return time.time_ns() / (10**9)

    def print_computation_log(self, aggregate=False, max_depth=None):
        self.computation_log.print_log(aggregate, max_depth)

    def generate_performance_graph(self, dir_path: str) -> None:
        self.performance_log.generate_graph(dir_path)

    def generate_performance_tables(self, dir_path: str) -> None:
        self.performance_log.generate_performance_tables(dir_path)

    def generate_variable_graph(self, name: str, output_vars: list[str]) -> None:
        self.variable_graph.visualize(
            name, aggregate=False, max_depth=None, output_vars=output_vars
        )

    def _get_nb_requests(self, tree: tracers.TraceNode, variable: str) -> int:
        tree_call = tree.name == variable
        children_calls = sum(
            self._get_nb_requests(child, variable) for child in tree.children
        )

        return tree_call + children_calls

    def get_nb_requests(self, variable: str) -> int:
        return sum(self._get_nb_requests(tree, variable) for tree in self.trees)

    def get_flat_trace(self) -> dict:
        return self.flat_trace.get_trace()

    def get_serialized_flat_trace(self) -> dict:
        return self.flat_trace.get_serialized_trace()

    def browse_trace(self) -> Iterator[tracers.TraceNode]:
        def _browse_node(node):
            yield node

            for child in node.children:
                yield from _browse_node(child)

        for node in self._trees:
            yield from _browse_node(node)

    def to_trace(
        self,
        format: str = TRACE_FORMAT_V1,
        *,
        model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export a versioned, JSON-compatible computation trace.

        Node identity fields (``variable`` / ``period`` / ``branch``) are taken
        from structured :class:`~policyengine_core.tracers.TraceNode` attributes
        via :meth:`browse_trace`, not by regex-parsing serialized flat-trace
        keys. :func:`parse_trace_key` remains available for callers that only
        have a flat key string (for example parameter maps without a node).

        **v1 intentionally omits** entity/count on ``calculation`` roots and
        source/version on ``parameters``. Those were listed as optional
        "minimum useful fields" in the provenance issue and can land in a later
        format version without breaking ``policyengine.trace.v1`` consumers.

        This document is *per-computation* provenance. It is complementary to
        PolicyEngine's TRACE Transparent Research Object (TRO) surface in
        ``policyengine`` (JSON-LD under ``https://policyengine.org/trace/0.1#``),
        which pins release/composition artifacts. A future TRO may embed or
        reference a ``policyengine.trace.v1`` payload as its runtime trace; the
        format id and TRO namespace are intentionally distinct.

        Values are full serialized vectors (household-audit friendly). Optional
        summarization for population-scale traces is deferred.

        :param format: Trace document format identifier. Currently only
            ``policyengine.trace.v1`` is supported.
        :param model: Optional country-package / model metadata
            (for example ``{"package": "policyengine-us", "version": "…"}``).
        :returns: A plain ``dict`` ready for ``json.dumps``.
        :raises ValueError: If ``format`` is not a supported version string.
        """
        if format != TRACE_FORMAT_V1:
            raise ValueError(
                f"Unsupported trace format {format!r}. "
                f"Supported formats: {TRACE_FORMAT_V1!r}"
            )

        engine = self._engine_metadata()
        flat_trace = self.flat_trace
        nodes: List[Dict[str, Any]] = []
        parameters: Dict[str, Any] = {}
        seen_node_ids: set[str] = set()

        for tree_node in self.browse_trace():
            node_id = flat_trace.key(tree_node)
            # Cache reads can revisit the same calculation; keep the first
            # structured record (same non-overwriting rule as FlatTrace).
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)

            parameter_map = {
                flat_trace.key(parameter): flat_trace.serialize(parameter.value)
                for parameter in tree_node.parameters
            }
            node: Dict[str, Any] = {
                "id": node_id,
                "variable": tree_node.name,
                "period": str(tree_node.period),
                "branch": tree_node.branch_name,
                "dependencies": [flat_trace.key(child) for child in tree_node.children],
                "parameters": parameter_map,
                "value": flat_trace.serialize(tree_node.value),
                "calculation_time": tree_node.calculation_time(),
                "formula_time": tree_node.formula_time(),
            }
            nodes.append(node)

            for parameter in tree_node.parameters:
                parameter_id = flat_trace.key(parameter)
                if parameter_id in parameters:
                    continue
                parameters[parameter_id] = {
                    "id": parameter_id,
                    "name": parameter.name,
                    "instant": str(parameter.period),
                    "branch": parameter.branch_name,
                    "value": flat_trace.serialize(parameter.value),
                }

        document: Dict[str, Any] = {
            "format": format,
            "engine": engine,
            "calculation": {
                "roots": [
                    {
                        "id": flat_trace.key(tree),
                        "variable": tree.name,
                        "period": str(tree.period),
                        "branch": tree.branch_name,
                    }
                    for tree in self._trees
                ],
            },
            "nodes": nodes,
            "parameters": parameters,
        }
        if model is not None:
            document["model"] = model
        return document

    @staticmethod
    def _engine_metadata() -> Dict[str, Any]:
        try:
            from policyengine_core.build_metadata import get_runtime_metadata

            metadata = get_runtime_metadata()
            engine: Dict[str, Any] = {
                "package": metadata.get("name", "policyengine-core"),
                "version": metadata.get("version", "unknown"),
            }
            if metadata.get("git_sha"):
                engine["git_sha"] = metadata["git_sha"]
            return engine
        except Exception:
            return {
                "package": "policyengine-core",
                "version": "unknown",
            }
