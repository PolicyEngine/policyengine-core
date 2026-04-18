"""Regression test: ``apply_reform`` must not wipe user-provided inputs.

The cache-invalidation added to fix bug H3 cleared
``holder._memory_storage._arrays`` for every variable, which also wiped
values populated via ``set_input`` that pre-dated the reform. Those values
are source data, not stale formula output, and must survive
``apply_reform`` so that downstream country packages (e.g. policyengine-uk,
policyengine-us) can load a dataset, apply a structural reform, and then
calculate against the loaded data.

See:
- https://github.com/PolicyEngine/policyengine.py/issues/1628 (symptom
  surfaced here — UK household-impact tests returning 0 after reform apply)
- https://github.com/PolicyEngine/policyengine-us/issues/8058 (symptom
  surfaced here — US ``tax_unit_itemizes`` integration test and many
  others crashing with ``TypeError: int() argument must be a string,
  a bytes-like object or a real number, not 'NoneType'`` because
  ``state_fips`` got wiped and the downstream
  ``state_name``/``state_code`` chain returned ``None``)
- bug H3 in the existing ``test_apply_reform_invalidates_cache.py``
  (the cache invalidation that over-reached)
"""

from __future__ import annotations

import numpy as np

from policyengine_core.model_api import Reform
from policyengine_core.country_template import situation_examples
from policyengine_core.simulations import Simulation, SimulationBuilder


def test_apply_reform_preserves_set_input_values(tax_benefit_system):
    """Values set via ``set_input`` before ``apply_reform`` must survive it.

    ``set_input`` is the data-load path: it populates the holder with source
    values, not cached formula output. Wiping it across ``apply_reform``
    would mean every country-package dataset is silently discarded whenever
    a structural reform is applied during initialisation.
    """
    sim = SimulationBuilder().build_from_entities(
        tax_benefit_system, situation_examples.single
    )
    period = "2017-01"
    expected_salary = np.array([5_000.0])

    sim.set_input("salary", period, expected_salary)

    assert sim.get_holder("salary").get_known_periods(), (
        "precondition failure: set_input did not register the period"
    )

    class NoOpReform(Reform):
        """Reform that touches nothing; should not invalidate inputs."""

        def apply(self):
            pass

    sim.apply_reform(NoOpReform)

    assert sim.get_holder("salary").get_known_periods(), (
        "apply_reform wiped salary holder — set_input values must be preserved"
    )

    result = sim.calculate("salary", period=period)
    assert np.allclose(result, expected_salary), (
        f"apply_reform lost the user-provided salary input; got {result} "
        f"instead of {expected_salary}."
    )


def test_apply_reform_preserves_inputs_across_multiple_variables(tax_benefit_system):
    """Every variable set via ``set_input`` must survive, not just one."""
    sim = SimulationBuilder().build_from_entities(
        tax_benefit_system, situation_examples.single
    )
    period = "2017-01"

    sim.set_input("salary", period, np.array([1_234.0]))
    sim.set_input("age", period, np.array([27]))

    class NoOpReform(Reform):
        def apply(self):
            pass

    sim.apply_reform(NoOpReform)

    assert np.allclose(sim.calculate("salary", period=period), [1_234.0])
    assert sim.calculate("age", period=period)[0] == 27


def test_apply_reform_preserves_situation_dict_inputs(tax_benefit_system):
    """Situation-dict inputs must survive ``apply_reform`` too.

    ``Simulation(situation=...)`` routes inputs through
    ``SimulationBuilder.finalize_variables_init``, which calls
    ``holder.set_input`` directly — bypassing ``Simulation.set_input``.
    The preservation tracking must cover that path too, otherwise
    country-package subclasses that build from a situation dict and then
    apply a structural reform during construction (the
    ``policyengine-us`` pattern) silently lose every household input.
    Surfaced in ``PolicyEngine/policyengine-us#8058``.
    """
    situation = {
        "persons": {
            "Alicia": {
                "salary": {"2017-01": 3_000.0},
                "age": {"2017-01": 42},
            }
        },
        "households": {
            "_": {"parents": ["Alicia"]},
        },
    }
    sim = Simulation(
        tax_benefit_system=tax_benefit_system,
        situation=situation,
    )

    assert sim.get_holder("salary").get_known_periods(), (
        "precondition failure: situation dict did not register salary"
    )
    assert sim.get_holder("age").get_known_periods(), (
        "precondition failure: situation dict did not register age"
    )

    class NoOpReform(Reform):
        def apply(self):
            pass

    sim.apply_reform(NoOpReform)

    # Both inputs were set through ``holder.set_input`` via the builder,
    # not through ``Simulation.set_input``. They must still survive.
    assert np.allclose(sim.calculate("salary", period="2017-01"), [3_000.0]), (
        "apply_reform wiped the situation-dict salary input"
    )
    assert sim.calculate("age", period="2017-01")[0] == 42, (
        "apply_reform wiped the situation-dict age input"
    )


def test_apply_reform_still_invalidates_formula_caches(tax_benefit_system):
    """The H3 fix must still hold — formula output caches must be cleared.

    This is a belt-and-braces test: preserving set_input values is orthogonal
    to invalidating formula outputs. A reform that neutralizes a variable
    must still cause subsequent ``calculate`` calls to return the new value,
    not the cached pre-reform output.
    """
    sim = SimulationBuilder().build_from_entities(
        tax_benefit_system, situation_examples.single
    )
    period = "2017-01"

    # Compute once to populate the formula-output cache.
    before_reform = sim.calculate("basic_income", period=period)
    assert before_reform[0] > 0

    class NeutraliseBasicIncome(Reform):
        def apply(self):
            self.neutralize_variable("basic_income")

    sim.apply_reform(NeutraliseBasicIncome)

    after_reform = sim.calculate("basic_income", period=period)
    assert after_reform[0] == 0, (
        f"apply_reform did not invalidate formula cache for basic_income; "
        f"got {after_reform[0]} instead of 0."
    )
