"""Flat traces retain accepted calculations across spiral retries."""

import numpy as np
import pytest

from policyengine_core.tracers import FullTracer
from tests.fixtures.spirals import build, make_recurrence


@pytest.mark.parametrize("storage", ["store", "drop", "blacklist"])
def test_flat_trace_matches_returned_result_after_a_cut_retry(storage):
    # Reduced from a counterexample executed by the prior review: a cut
    # child recorded 0 before the accepted request returned 1.
    variable = make_recurrence("r", [("r", 1), ("r", -1)], end=2020)
    plain = build([variable], max_spiral_loops=1)
    traced = build([variable], max_spiral_loops=1)
    traced.trace = True
    for simulation in (plain, traced):
        if storage == "drop":
            simulation.get_holder("r")._do_not_store = True
        elif storage == "blacklist":
            simulation.opt_out_cache = True
            simulation.tax_benefit_system.cache_blacklist = {"r"}

    expected = plain.calculate("r", "2000").tolist()
    result = traced.calculate("r", "2000").tolist()
    assert result == expected == [1.0]
    trace = traced.tracer.get_serialized_flat_trace()
    assert trace["r<2000, (default)>"]["value"] == result
    assert traced.tracer.stack == []


def test_flat_trace_keeps_first_accepted_calculation_over_cache_reads():
    tracer = FullTracer()
    tracer.record_calculation_start("r", "2000")
    tracer.record_calculation_end()  # An abandoned retry has no result.

    tracer.record_calculation_start("r", "2000")
    tracer.record_calculation_result(np.array([0.0]))
    tracer._current_node._spiral_provisional = True
    tracer.record_calculation_end()

    tracer.record_calculation_start("r", "2000")
    tracer.record_calculation_start("dependency", "2000")
    tracer.record_calculation_result(np.array([1.0]))
    tracer.record_calculation_end()
    tracer.record_calculation_result(np.array([1.0]))
    tracer.record_calculation_end()

    tracer.record_calculation_start("r", "2000")
    tracer.record_calculation_result(np.array([1.0]))
    tracer.record_calculation_end()  # A cache read has no dependencies.

    entry = tracer.get_serialized_flat_trace()["r<2000, (default)>"]
    assert entry["value"] == [1.0]
    assert entry["dependencies"] == ["dependency<2000, (default)>"]


def test_stable_child_metadata_survives_an_abandoned_parent_retry():
    tracer = FullTracer()
    tracer.record_calculation_start("retry", "2000")
    tracer._current_node._spiral_provisional = True
    tracer.record_calculation_start("stable", "2000")
    tracer.record_parameter_access("rate", "2000-01-01", "default", 2.0)
    tracer.record_calculation_start("dependency", "2000")
    tracer.record_calculation_result(np.array([3.0]))
    tracer.record_calculation_end()
    tracer.record_calculation_result(np.array([6.0]))
    tracer.record_calculation_end()
    tracer.record_calculation_end()  # The parent's deeper request unwinds.

    tracer.record_calculation_start("stable", "2000")
    tracer.record_calculation_result(np.array([6.0]))
    tracer.record_calculation_end()

    entry = tracer.get_serialized_flat_trace()["stable<2000, (default)>"]
    assert entry["value"] == [6.0]
    assert entry["dependencies"] == ["dependency<2000, (default)>"]
    assert entry["parameters"] == {"rate<2000-01-01, (default)>": 2.0}


def test_provisional_dependencies_remain_in_the_flat_trace():
    tracer = FullTracer()
    tracer.record_calculation_start("r", "2000")
    tracer.record_calculation_start("cut", "1999")
    tracer.record_calculation_result(np.array([0.0]))
    tracer._current_node._spiral_provisional = True
    tracer.record_calculation_end()
    tracer.record_calculation_result(np.array([1.0]))
    tracer.record_calculation_end()

    trace = tracer.get_serialized_flat_trace()
    dependency = trace["r<2000, (default)>"]["dependencies"][0]
    assert trace[dependency]["value"] == [0.0]
