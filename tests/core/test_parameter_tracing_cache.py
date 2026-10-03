"""A traced parameter tree wraps its cached nodes on each read and caches only
the plain nodes, so no tracer outlives its simulation and no cached node goes
untraced."""

from policyengine_core.parameters import ParameterNode, ParameterNodeAtInstant
from policyengine_core.tracers import FullTracer, TracingParameterNodeAtInstant


def tree():
    return ParameterNode("", data={"x": {"values": {"2010-01-01": 1}}})


def trace(node, tracer):
    node.trace, node.tracer, node.branch_name = True, tracer, "default"


def test_a_node_cached_before_tracing_is_traced_once_tracing_starts():
    parameters = tree()
    plain = parameters.get_at_instant("2017-01-01")
    trace(parameters, FullTracer())
    traced = parameters.get_at_instant("2017-01-01")
    assert isinstance(traced, TracingParameterNodeAtInstant)
    assert traced.parameter_node_at_instant is plain


def test_the_cache_never_holds_a_tracer():
    parameters = tree()
    for tracer in (FullTracer(), FullTracer()):
        trace(parameters, tracer)
        assert parameters.get_at_instant("2017-01-01").tracer is tracer
        assert parameters.get_at_instant("2018-01-01").tracer is tracer
    assert all(
        type(node) is ParameterNodeAtInstant
        for node in parameters._at_instant_cache.values()
    )
    parameters.trace, parameters.tracer = False, None
    assert type(parameters.get_at_instant("2017-01-01")) is ParameterNodeAtInstant
