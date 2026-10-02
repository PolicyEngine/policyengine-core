"""Tracing a simulation must not change the tax-benefit system it runs on.

``Simulation._run_formula`` used to switch tracing on in the system's
parameter tree (setting ``trace``, ``tracer`` and ``branch_name`` on the root
``ParameterNode``) the first time a traced simulation ran a formula, and never
switched it off. From then on:

- the root node caches one node per instant. ``Simulation._calculate`` reads
  ``parameters(period)`` (the abolition check) before any formula runs, so at
  an instant first read before tracing came on the cached node was untraced,
  and the traced simulation recorded no parameter reads there. At an instant
  first read afterwards it was a ``TracingParameterNodeAtInstant`` holding the
  tracer and branch name of that moment, and every later simulation on the
  system, traced or not, read through it: their own tracers missed the reads
  and a branch's reads were filed under another branch's name;
- ``ParameterNode.clone`` copies ``trace`` and ``tracer``, so every clone of
  the system traced as well;
- copying a ``TracingParameterNodeAtInstant`` (``copy``, ``deepcopy`` or
  ``pickle``, as ``Reform.modify_parameters`` does when it deep-copies the
  tree) recursed without end, because ``__getattr__`` read
  ``parameter_node_at_instant`` from an instance the copy protocol had not
  filled in yet.

Formulas of a traced simulation now read parameters through a
``TracingParameterNode`` that belongs to the call, and both wrappers answer
lookups they cannot delegate with ``AttributeError``.

Invariants checked here: tracing a simulation leaves the system's parameter
tree exactly as untraced (flags, cached nodes, clones, deep copies); each
traced simulation and branch records its own parameter reads, under its own
branch name; and tracing never changes a computed value.
"""

import copy
import pickle

import numpy as np
import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.parameters import ParameterNodeAtInstant
from policyengine_core.periods import MONTH
from policyengine_core.reforms import Reform
from policyengine_core.tracers import (
    FullTracer,
    TracingParameterNode,
    TracingParameterNodeAtInstant,
)
from policyengine_core.variables import Variable
from tests.fixtures.tracing import (
    assert_untraced,
    build_simulation,
    parameter_reads,
)

JANUARY = "2017-01"
INSTANT = "2017-01-01"


# ----- The system is left as it was ----- #


def test_traced_calculation_leaves_the_system_untraced(isolated_tax_benefit_system):
    system = isolated_tax_benefit_system
    simulation = build_simulation(system, trace=True)
    simulation.calculate("income_tax", JANUARY)
    # A second month: its instant is first read while the simulation traces.
    simulation.calculate("income_tax", "2017-02")

    reads = parameter_reads(simulation.tracer.trees)
    assert reads.count(("taxes.income_tax_rate", "default")) == 2
    assert_untraced(system.parameters)
    assert type(system.get_parameters_at_instant(INSTANT)) is ParameterNodeAtInstant
    assert_untraced(system.clone().parameters)


def test_reform_after_a_traced_calculation_can_copy_the_parameters(
    isolated_tax_benefit_system,
):
    """``Reform.modify_parameters`` deep-copies the baseline's parameter tree."""
    system = isolated_tax_benefit_system
    simulation = build_simulation(system, trace=True)
    simulation.calculate("income_tax", JANUARY)
    # A second month: its instant is first read while the simulation traces.
    simulation.calculate("income_tax", "2017-02")

    class double_income_tax_rate(Reform):
        def apply(self):
            def modify(parameters):
                parameters.taxes.income_tax_rate.update(
                    period=f"year:{INSTANT}:1",
                    value=2 * parameters.taxes.income_tax_rate(INSTANT),
                )
                return parameters

            self.modify_parameters(modify)

    reform = double_income_tax_rate(system)

    rate = system.parameters(INSTANT).taxes.income_tax_rate
    assert reform.get_parameters_at_instant(INSTANT).taxes.income_tax_rate == (
        pytest.approx(2 * rate)
    )
    assert system.parameters(INSTANT).taxes.income_tax_rate == rate


# ----- Each simulation and branch records its own reads ----- #


def test_each_traced_simulation_records_its_own_parameter_reads(
    isolated_tax_benefit_system,
):
    system = isolated_tax_benefit_system
    first = build_simulation(system, trace=True)
    first.calculate("income_tax", JANUARY)
    first_reads = parameter_reads(first.tracer.trees)

    second = build_simulation(system, trace=True)
    second.calculate("income_tax", JANUARY)

    assert ("taxes.income_tax_rate", "default") in parameter_reads(second.tracer.trees)
    assert parameter_reads(first.tracer.trees) == first_reads


def test_branch_records_parameter_reads_under_its_own_name(
    isolated_tax_benefit_system,
):
    simulation = build_simulation(isolated_tax_benefit_system, trace=True)
    # The parent reads parameters at the same instant first.
    simulation.calculate("income_tax", JANUARY)

    branch = simulation.get_branch("policy")
    branch.calculate("basic_income", JANUARY)

    reads = parameter_reads(simulation.tracer.trees)
    assert ("benefits.basic_income", "policy") in reads
    assert ("benefits.basic_income", "default") not in reads


def test_formulas_of_an_untraced_simulation_get_untraced_parameters(
    isolated_tax_benefit_system,
):
    system = isolated_tax_benefit_system
    seen = []

    class parameters_seen_by_formula(Variable):
        value_type = float
        entity = Person
        definition_period = MONTH
        label = "Parameters seen by the formula"

        def formula(person, period, parameters):
            seen.append(parameters(period))
            return person.filled_array(0)

    system.add_variable(parameters_seen_by_formula)

    build_simulation(system, trace=True).calculate(
        "parameters_seen_by_formula", JANUARY
    )
    build_simulation(system).calculate("parameters_seen_by_formula", JANUARY)

    traced, untraced = seen
    assert type(traced) is TracingParameterNodeAtInstant
    assert type(untraced) is ParameterNodeAtInstant


def test_traced_formula_can_read_parameters_by_attribute(isolated_tax_benefit_system):
    system = isolated_tax_benefit_system

    class income_tax_rate_by_attribute(Variable):
        value_type = float
        entity = Person
        definition_period = MONTH
        label = "Income tax rate read by attribute"

        def formula(person, period, parameters):
            return person.filled_array(parameters.taxes.income_tax_rate(period))

    system.add_variable(income_tax_rate_by_attribute)

    traced = build_simulation(system, trace=True).calculate(
        "income_tax_rate_by_attribute", JANUARY
    )
    untraced = build_simulation(system).calculate(
        "income_tax_rate_by_attribute", JANUARY
    )

    np.testing.assert_array_equal(traced, untraced)
    assert traced[0] == system.parameters(INSTANT).taxes.income_tax_rate


# ----- Tracing never changes a value ----- #


COMPUTED_VARIABLES = sorted(
    name
    for name, variable in CountryTaxBenefitSystem().variables.items()
    if variable.formulas
)


@pytest.mark.parametrize("variable_name", COMPUTED_VARIABLES)
def test_traced_and_untraced_simulations_agree(tax_benefit_system, variable_name):
    """Differential check over every formula of the country template."""
    variable = tax_benefit_system.get_variable(variable_name)
    period = "2017" if variable.definition_period == "year" else JANUARY

    untraced = build_simulation(tax_benefit_system).calculate(variable_name, period)
    traced_simulation = build_simulation(tax_benefit_system, trace=True)
    traced = traced_simulation.calculate(variable_name, period)

    np.testing.assert_array_equal(traced, untraced)
    assert_untraced(tax_benefit_system.parameters)


# ----- The tracing wrappers ----- #


def test_tracing_parameter_node_reads_through_to_the_node(isolated_tax_benefit_system):
    parameters = isolated_tax_benefit_system.parameters
    tracer = FullTracer()
    view = TracingParameterNode(parameters, tracer, "policy")

    assert view.taxes is parameters.taxes
    assert view.children is parameters.children

    at_instant = view(INSTANT)
    assert type(at_instant) is TracingParameterNodeAtInstant
    assert at_instant.parameter_node_at_instant is parameters(INSTANT)
    assert at_instant.tracer is tracer
    assert at_instant.branch_name == "policy"
    assert type(view.get_at_instant(INSTANT)) is TracingParameterNodeAtInstant
    assert_untraced(parameters)


def test_tracing_parameter_node_records_in_its_own_tracer_when_the_node_traces(
    isolated_tax_benefit_system,
):
    parameters = isolated_tax_benefit_system.parameters
    parameters.trace = True
    parameters.tracer = FullTracer()
    parameters.branch_name = "elsewhere"
    tracer = FullTracer()

    at_instant = TracingParameterNode(parameters, tracer, "policy")(INSTANT)

    assert at_instant.tracer is tracer
    assert at_instant.branch_name == "policy"
    assert type(at_instant.parameter_node_at_instant) is ParameterNodeAtInstant


def _round_trip_by_pickle(value):
    return pickle.loads(pickle.dumps(value))


DUPLICATORS = pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, _round_trip_by_pickle],
    ids=["copy", "deepcopy", "pickle"],
)


@DUPLICATORS
def test_tracing_parameter_node_at_instant_can_be_copied(
    isolated_tax_benefit_system, duplicate
):
    tracer = FullTracer()
    tracer.record_calculation_start("income_tax", JANUARY)
    traced = TracingParameterNodeAtInstant(
        isolated_tax_benefit_system.parameters(INSTANT), tracer, "default"
    )

    duplicated = duplicate(traced)

    assert type(duplicated) is TracingParameterNodeAtInstant
    assert duplicated.branch_name == "default"
    assert duplicated.taxes.income_tax_rate == traced.taxes.income_tax_rate
    assert parameter_reads(duplicated.tracer.trees)[-1] == (
        "taxes.income_tax_rate",
        "default",
    )


@DUPLICATORS
def test_tracing_parameter_node_can_be_copied(isolated_tax_benefit_system, duplicate):
    view = TracingParameterNode(
        isolated_tax_benefit_system.parameters, FullTracer(), "default"
    )

    duplicated = duplicate(view)

    assert type(duplicated) is TracingParameterNode
    assert duplicated.branch_name == "default"
    assert duplicated(INSTANT).taxes.income_tax_rate == (
        view(INSTANT).taxes.income_tax_rate
    )


def test_parameters_holding_traced_nodes_can_be_deep_copied(
    isolated_tax_benefit_system,
):
    """A tree whose ``trace`` flag is set caches traced nodes; copying it works."""
    parameters = isolated_tax_benefit_system.parameters
    parameters.trace = True
    parameters.tracer = FullTracer()
    parameters(INSTANT)
    assert type(parameters._at_instant_cache[INSTANT]) is (
        TracingParameterNodeAtInstant
    )

    duplicated = copy.deepcopy(parameters)

    assert duplicated(INSTANT).taxes.income_tax_rate == (
        parameters(INSTANT).taxes.income_tax_rate
    )


@pytest.mark.parametrize(
    "wrapper_class", [TracingParameterNode, TracingParameterNodeAtInstant]
)
def test_lookups_on_an_unfilled_wrapper_raise_attribute_error(wrapper_class):
    """``copy`` and ``pickle`` probe instances they have only created with ``__new__``."""
    unfilled = wrapper_class.__new__(wrapper_class)

    with pytest.raises(AttributeError):
        unfilled.taxes
    assert getattr(unfilled, "__setstate__", None) is None
    assert not hasattr(unfilled, "parameter_node_at_instant")
    assert not hasattr(unfilled, "parameter_node")
