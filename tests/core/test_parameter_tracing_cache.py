"""A traced parameter tree wraps its cached nodes on each read and caches only
the plain nodes, so no tracer outlives its simulation and no cached node goes
untraced. The tax-benefit system's own cache of the tree at each instant
(``TaxBenefitSystem.get_parameters_at_instant``) follows the same rule."""

import gc
import weakref

from policyengine_core.country_template.entities import entities
from policyengine_core.parameters import ParameterNode, ParameterNodeAtInstant
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.tracers import FullTracer, TracingParameterNodeAtInstant

DATES = [f"{year}-01-01" for year in range(2011, 2027)]


def tree():
    return ParameterNode("", data={"x": {"values": {"2010-01-01": 1}}})


def system():
    tax_benefit_system = TaxBenefitSystem(entities)
    tax_benefit_system.parameters = tree()
    return tax_benefit_system


def trace(node, tracer, branch_name="default"):
    node.trace, node.tracer, node.branch_name = True, tracer, branch_name


def untrace(node):
    node.trace, node.tracer, node.branch_name = False, None, None


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


def test_the_system_cache_never_holds_a_tracer():
    """Reads at distinct dates, each under a tracer of its own, as traced
    cases on one cached system make them: the system used to cache one
    tracing wrapper per date, each keeping its tracer alive."""
    tax_benefit_system = system()
    refs = []
    for date in DATES:
        tracer = FullTracer()
        refs.append(weakref.ref(tracer))
        trace(tax_benefit_system.parameters, tracer)
        assert tax_benefit_system.get_parameters_at_instant(date).tracer is tracer
        del tracer
    untrace(tax_benefit_system.parameters)
    gc.collect()

    cached = tax_benefit_system._parameters_at_instant_cache
    assert len(cached) == len(DATES)
    assert all(type(node) is ParameterNodeAtInstant for node in cached.values())
    assert [ref for ref in refs if ref() is not None] == []


def test_a_system_node_cached_before_tracing_is_traced_once_tracing_starts():
    tax_benefit_system = system()
    plain = tax_benefit_system.get_parameters_at_instant("2017-01-01")
    tracer = FullTracer()
    trace(tax_benefit_system.parameters, tracer, "branch")
    traced = tax_benefit_system.get_parameters_at_instant("2017-01-01")
    assert isinstance(traced, TracingParameterNodeAtInstant)
    assert traced.parameter_node_at_instant is plain
    assert (traced.tracer, traced.branch_name) == (tracer, "branch")
    untrace(tax_benefit_system.parameters)
    assert tax_benefit_system.get_parameters_at_instant("2017-01-01") is plain


def test_each_system_read_records_in_the_tracer_current_at_the_read():
    """A read through the system records in the tree's tracer at the time of
    the read, and nowhere once tracing stops: it used to record every later
    read in the tracer current when the instant was first read."""
    tax_benefit_system = system()
    first, second = FullTracer(), FullTracer()
    for tracer in (first, second):
        tracer.record_calculation_start("y", "2017")

    trace(tax_benefit_system.parameters, first)
    assert tax_benefit_system.get_parameters_at_instant("2017-01-01").x == 1
    trace(tax_benefit_system.parameters, second)
    assert tax_benefit_system.get_parameters_at_instant("2017-01-01").x == 1
    untrace(tax_benefit_system.parameters)
    assert tax_benefit_system.get_parameters_at_instant("2017-01-01").x == 1

    assert len(first.trees[0].parameters) == 1
    assert len(second.trees[0].parameters) == 1
