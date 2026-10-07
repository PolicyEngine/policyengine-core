"""What a traced simulation records must not depend on what ran before it.

Runs random sequences of calculations on one tax-benefit system: traced and
untraced simulations, branches of traced simulations, and traced simulations
on clones of the system. After every step:

- the parameter tree of the system, and of every clone made of it, is as
  untraced as a fresh system's;
- a traced simulation, or a branch of one, records exactly the parameter
  reads the same calculation records when traced on a fresh system, a
  branch's all under its own name;
- every value equals the same calculation on a system nothing has traced.

``test_tracing_parameter_isolation.py`` has the example tests.
"""

import functools

import numpy as np
import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from tests.fixtures.tracing import (
    assert_untraced,
    build_simulation,
    parameter_reads,
)

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

VARIABLES = sorted(
    name
    for name, variable in CountryTaxBenefitSystem().variables.items()
    if variable.formulas
)
# Both dated formulas of ``basic_income`` (from 2015-12 and from 2016-12).
# Earlier months fail traced or not: parenting_allowance has no parameters yet.
MONTHS = ["2015-12", "2016-06", "2016-12", "2017-01", "2017-02"]
KINDS = ["traced", "untraced", "branch", "clone"]

_REFERENCE = CountryTaxBenefitSystem()


def _period(variable_name, month):
    variable = _REFERENCE.get_variable(variable_name)
    return month[:4] if variable.definition_period == "year" else month


@functools.lru_cache(maxsize=None)
def _expected(variable_name, period):
    """Value and parameter reads of the calculation on fresh systems."""
    value = build_simulation(CountryTaxBenefitSystem()).calculate(variable_name, period)
    traced = build_simulation(CountryTaxBenefitSystem(), trace=True)
    traced.calculate(variable_name, period)
    return value, parameter_reads(traced.tracer.trees)


@hypothesis.settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(
    steps=st.lists(
        st.tuples(
            st.sampled_from(KINDS),
            st.sampled_from(VARIABLES),
            st.sampled_from(MONTHS),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_traced_record_does_not_depend_on_history(steps):
    system = CountryTaxBenefitSystem()
    systems = [system]

    for kind, variable_name, month in steps:
        period = _period(variable_name, month)
        expected_value, expected_reads = _expected(variable_name, period)

        if kind == "untraced":
            value = build_simulation(system).calculate(variable_name, period)
        elif kind == "traced":
            simulation = build_simulation(system, trace=True)
            value = simulation.calculate(variable_name, period)
            assert parameter_reads(simulation.tracer.trees) == expected_reads
        elif kind == "branch":
            simulation = build_simulation(system, trace=True)
            # The parent reads parameters at the same instant first. An input
            # has no formula, so the branch reuses nothing that reads them.
            simulation.calculate("salary", month)
            trees_before = len(simulation.tracer.trees)
            value = simulation.get_branch("policy").calculate(variable_name, period)
            reads = parameter_reads(simulation.tracer.trees[trees_before:])
            assert reads == [(name, "policy") for name, _ in expected_reads]
        else:
            clone = system.clone()
            systems.append(clone)
            simulation = build_simulation(clone, trace=True)
            value = simulation.calculate(variable_name, period)
            assert parameter_reads(simulation.tracer.trees) == expected_reads

        np.testing.assert_array_equal(value, expected_value)
        for each_system in systems:
            assert_untraced(each_system.parameters)
