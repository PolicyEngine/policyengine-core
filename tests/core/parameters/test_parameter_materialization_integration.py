from __future__ import annotations

import numpy as np
import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.parameters import (
    LazyParameterMaterializer,
    LazyParameterNodeAtInstant,
    StaleParameterViewError,
)
from policyengine_core.reforms import Reform
from tests.core.parameters_fancy_indexing.test_fancy_indexing import (
    parameters as fancy_indexing_parameters,
)
from tests.fixtures.tracing import build_simulation, parameter_reads


@pytest.mark.parametrize(
    ("variable_name", "period"),
    [
        pytest.param("income_tax", "2017-01", id="income-tax"),
        pytest.param(
            "social_security_contribution",
            "2017-01",
            id="social-security-contribution",
        ),
        pytest.param("basic_income", "2017-01", id="basic-income"),
        pytest.param("housing_allowance", "2017-01", id="housing-allowance"),
        pytest.param("disposable_income", "2017-01", id="disposable-income"),
        pytest.param("housing_tax", "2017", id="housing-tax"),
    ],
)
def test_eager_and_lazy_strategies_produce_equal_formula_results(
    isolated_tax_benefit_system,
    variable_name: str,
    period: str,
) -> None:
    eager_system = isolated_tax_benefit_system
    lazy_system = eager_system.clone()
    lazy_system.set_parameter_materializer(LazyParameterMaterializer())

    eager = build_simulation(eager_system).calculate(variable_name, period)
    lazy = build_simulation(lazy_system).calculate(variable_name, period)

    np.testing.assert_array_equal(lazy, eager)


def test_lazy_strategy_preserves_parameter_tracing(
    isolated_tax_benefit_system,
) -> None:
    isolated_tax_benefit_system.set_parameter_materializer(LazyParameterMaterializer())
    simulation = build_simulation(isolated_tax_benefit_system, trace=True)

    simulation.calculate("income_tax", "2017-01")

    assert ("taxes.income_tax_rate", "default") in parameter_reads(
        simulation.tracer.trees
    )
    assert all(
        type(value) is LazyParameterNodeAtInstant
        for value in isolated_tax_benefit_system._parameters_at_instant_cache.values()
    )


def test_lazy_strategy_preserves_vectorial_parameter_indexing() -> None:
    eager = fancy_indexing_parameters.clone()
    lazy = fancy_indexing_parameters.clone()
    lazy.set_parameter_materializer(LazyParameterMaterializer())
    families = np.array(["single", "couple", "single", "couple"])
    housing = np.array(["owner", "owner", "tenant", "tenant"])

    eager_values = eager("2015-01-01").rate[families][housing].z2
    lazy_values = lazy("2015-01-01").rate[families][housing].z2

    np.testing.assert_array_equal(lazy_values, eager_values)


def test_lazy_baseline_and_reform_keep_independent_revisions() -> None:
    baseline = CountryTaxBenefitSystem()
    baseline.set_parameter_materializer(LazyParameterMaterializer())
    baseline_view = baseline.get_parameters_at_instant("2026-01-01")
    baseline_rate = baseline_view.taxes.income_tax_rate

    reform = Reform.from_dict(
        {"taxes.income_tax_rate": {"2026-01-01": baseline_rate + 0.1}}
    )(baseline)

    assert reform.get_parameters_at_instant(
        "2026-01-01"
    ).taxes.income_tax_rate == pytest.approx(baseline_rate + 0.1)
    assert baseline_view.taxes.income_tax_rate == baseline_rate
    assert reform.parameters.parameter_revision is not (
        baseline.parameters.parameter_revision
    )


def test_shared_lazy_policy_state_invalidates_both_system_lookups() -> None:
    first = CountryTaxBenefitSystem()
    first.set_parameter_materializer(LazyParameterMaterializer())
    second = CountryTaxBenefitSystem()
    second.share_parameters_from(first)
    old_view = first.get_parameters_at_instant("2026-01-01")
    assert second.get_parameters_at_instant("2026-01-01") is old_view

    first.parameters.taxes.income_tax_rate.update(
        value=0.5,
        start="2026-01-01",
    )

    with pytest.raises(StaleParameterViewError):
        old_view.taxes
    assert first.get_parameters_at_instant(
        "2026-01-01"
    ).taxes.income_tax_rate == pytest.approx(0.5)
    assert second.get_parameters_at_instant(
        "2026-01-01"
    ).taxes.income_tax_rate == pytest.approx(0.5)
