"""The parameter-instant caches under tracing, for any sequence of tracing
switches and reads at any dates.

Two paths read the parameter tree at an instant: the tree's own
(``ParameterNode.get_at_instant``, cached in ``_at_instant_cache``) and the
tax-benefit system's (``TaxBenefitSystem.get_parameters_at_instant``, cached
in ``_parameters_at_instant_cache``). Invariants checked after every step:

- both caches hold plain ``ParameterNodeAtInstant`` objects only, so neither
  keeps a tracer;
- differential: both paths return the same plain node for an instant, traced
  exactly when the tree is traced at the time of the read, with the tree's
  tracer and branch name at that time;
- a traced read records in that tracer and in no other, and tracing never
  changes a value read;
- once tracing stops and the caller drops its tracers, none is alive.

``test_parameter_tracing_cache.py`` pins the same behaviour with examples.
"""

from __future__ import annotations

import gc
import weakref

import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.country_template.entities import entities
from policyengine_core.parameters import ParameterNode, ParameterNodeAtInstant
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.tracers import FullTracer, TracingParameterNodeAtInstant

# The parameter's value from each start date on: the reference the reads are
# checked against.
X_VALUES = {"2010-01-01": 1, "2015-01-01": 2, "2020-01-01": 3}
DATES = ["2012-06-01", "2015-01-01", "2017-01-01", "2020-01-01", "2024-12-31"]
TRACERS = 3


def expected_x(date: str) -> int:
    return X_VALUES[max(start for start in X_VALUES if start <= date)]


def system() -> TaxBenefitSystem:
    tax_benefit_system = TaxBenefitSystem(entities)
    tax_benefit_system.parameters = ParameterNode(
        "", data={"x": {"values": dict(X_VALUES)}}
    )
    return tax_benefit_system


steps = st.lists(
    st.one_of(
        st.tuples(
            st.just("trace"),
            st.integers(min_value=0, max_value=TRACERS - 1),
            st.sampled_from(["default", "branch"]),
        ),
        st.tuples(st.just("untrace")),
        st.tuples(st.sampled_from(["system", "tree"]), st.sampled_from(DATES)),
    ),
    max_size=30,
)


def check_caches(tax_benefit_system: TaxBenefitSystem) -> None:
    for cache in (
        tax_benefit_system._parameters_at_instant_cache,
        tax_benefit_system.parameters._at_instant_cache,
    ):
        assert all(type(node) is ParameterNodeAtInstant for node in cache.values())


def plain(node):
    if isinstance(node, TracingParameterNodeAtInstant):
        return node.parameter_node_at_instant
    return node


@hypothesis.given(sequence=steps)
# ``too_slow`` times input generation by the wall clock, so it fails on a busy
# host, not on anything this property checks.
@hypothesis.settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_caches_stay_tracer_free_and_both_paths_agree(sequence):
    tax_benefit_system = system()
    parameters = tax_benefit_system.parameters
    tracers = [FullTracer() for _ in range(TRACERS)]
    for tracer in tracers:
        tracer.record_calculation_start("y", "2017")
    refs = [weakref.ref(tracer) for tracer in tracers]
    current = None  # (tracer index, branch name) while the tree is traced
    reads = [0] * TRACERS

    for step in sequence:
        if step[0] == "trace":
            _, index, branch_name = step
            parameters.trace = True
            parameters.tracer = tracers[index]
            parameters.branch_name = branch_name
            current = (index, branch_name)
        elif step[0] == "untrace":
            parameters.trace, parameters.tracer, parameters.branch_name = (
                False,
                None,
                None,
            )
            current = None
        else:
            path, date = step
            read = (
                tax_benefit_system.get_parameters_at_instant(date)
                if path == "system"
                else parameters.get_at_instant(date)
            )
            other = (
                parameters.get_at_instant(date)
                if path == "system"
                else tax_benefit_system.get_parameters_at_instant(date)
            )
            for node in (read, other):
                if current is None:
                    assert type(node) is ParameterNodeAtInstant
                else:
                    index, branch_name = current
                    assert isinstance(node, TracingParameterNodeAtInstant)
                    assert node.tracer is tracers[index]
                    assert node.branch_name == branch_name
            assert plain(read) is plain(other)
            assert read.x == expected_x(date)
            if current is not None:
                reads[current[0]] += 1
            del read, other, node
        check_caches(tax_benefit_system)
        assert [len(tracer.trees[0].parameters) for tracer in tracers] == reads

    parameters.trace, parameters.tracer, parameters.branch_name = False, None, None
    del tracers, tracer
    gc.collect()
    assert [ref for ref in refs if ref() is not None] == []
