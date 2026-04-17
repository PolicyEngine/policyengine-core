"""Regression test: Reform must not mutate baseline parameters (C4).

The ``Reform`` base class previously stored a reference to the baseline's
``parameters`` tree and shared its ``_parameters_at_instant_cache``. Any
direct parameter mutation inside ``apply()`` therefore mutated the baseline
too, and stale cached ``ParameterNodeAtInstant`` objects kept serving the
baseline's pre-reform view.
"""

from __future__ import annotations

from policyengine_core.model_api import Reform


def test_reform_direct_parameter_mutation_does_not_leak_to_baseline(
    tax_benefit_system,
):
    # Record the baseline's basic income value at a known instant.
    baseline_param = tax_benefit_system.parameters.benefits.basic_income
    baseline_value_before = baseline_param("2017-01-01")

    class BasicIncomeRaised(Reform):
        def apply(self):
            # Directly mutate self.parameters, the exact pattern users follow
            # when writing reforms without the ``modify_parameters`` helper.
            self.parameters.benefits.basic_income.update(
                period="year:2000:100",
                value=baseline_value_before * 2,
            )

    reform = BasicIncomeRaised(tax_benefit_system)

    # The reform tree must see the new value.
    assert reform.parameters.benefits.basic_income("2017-01-01") == (
        baseline_value_before * 2
    )
    # The baseline must be untouched.
    assert (
        tax_benefit_system.parameters.benefits.basic_income("2017-01-01")
        == baseline_value_before
    )


def test_reform_has_independent_parameters_at_instant_cache(tax_benefit_system):
    # Prime the baseline cache.
    tax_benefit_system.get_parameters_at_instant("2017-01-01")
    cached_before = tax_benefit_system._parameters_at_instant_cache

    class NoOp(Reform):
        def apply(self):
            pass

    reform = NoOp(tax_benefit_system)
    # The reform must NOT share the cache dict with the baseline;
    # otherwise parameter mutations under the reform would still be served
    # from cached baseline snapshots.
    assert reform._parameters_at_instant_cache is not cached_before
