"""A case's dotted parameter inputs run on a system that copies only the
parameters they set, on top of the system the case's reforms select."""

import pytest
import yaml

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.reforms import Reform, set_parameter
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.tools.test_runner import (
    OpenFiscaPlugin,
    _ParameterInputReform,
    _get_tax_benefit_system,
    _tax_benefit_system_cache,
)
from tests.fixtures.yaml_runner_memory import RUNNER_ARGV, write_cases

SCALE_REFORM = (
    "policyengine_core.country_template.reforms."
    "modify_social_security_taxation.modify_social_security_taxation"
)
PARAMETER_INPUTS = [
    ("taxes.income_tax_rate", 0.3),
    ("taxes.housing_tax.minimal_amount", 350),
    ("taxes.social_security_contribution[1].rate", 0.5),
]


def _full_clone_system(baseline, reforms, parameter_inputs):
    """The system the runner built before: a full clone, the named reforms,
    then a reform that sets the inputs on a full copy of the tree."""
    system = baseline.clone()
    for reform in reforms:
        system = system.apply_reform(reform)

    class inline_reform_class(Reform):
        def apply(self):
            for key, value in parameter_inputs:
                self.parameters = set_parameter(
                    key, value, return_modifier=True, period="year:2000:40"
                )(self.parameters)

    return inline_reform_class(system)


def _run(path, baseline):
    return pytest.main(
        [*RUNNER_ARGV, "-q", "-p", "no:cacheprovider", str(path)],
        plugins=[OpenFiscaPlugin(baseline, {})],
    )


def test_the_system_shares_every_parameter_it_does_not_set():
    baseline = CountryTaxBenefitSystem()
    reform_free = _get_tax_benefit_system(baseline, [], [])
    system = _ParameterInputReform(reform_free, [("taxes.income_tax_rate", 0.3)])

    assert system.baseline is reform_free
    assert system.parameters.taxes.income_tax_rate("2017-01-01") == 0.3
    assert reform_free.parameters.taxes.income_tax_rate("2017-01-01") == 0.15
    assert system.parameters.benefits is reform_free.parameters.benefits
    assert system.parameters.taxes.housing_tax is (
        reform_free.parameters.taxes.housing_tax
    )


@pytest.mark.parametrize("reforms", [[], [SCALE_REFORM]])
@pytest.mark.parametrize("instant", ["2015-06-01", "2017-01-01", "2030-01-01"])
def test_it_reads_what_a_full_clone_with_the_same_inputs_reads(reforms, instant):
    """Including a bracket index, which counts the brackets the named reform
    left: the inputs apply after the reforms, as they did on a full clone."""
    baseline = CountryTaxBenefitSystem()
    structural = _get_tax_benefit_system(baseline, reforms, [])
    system = _ParameterInputReform(structural, PARAMETER_INPUTS)
    reference = _full_clone_system(baseline, reforms, PARAMETER_INPUTS)

    values = system.get_parameters_at_instant(instant)
    expected = reference.get_parameters_at_instant(instant)
    assert values.taxes.income_tax_rate == expected.taxes.income_tax_rate
    assert (
        values.taxes.housing_tax.minimal_amount
        == expected.taxes.housing_tax.minimal_amount
    )
    scale = values.taxes.social_security_contribution
    expected_scale = expected.taxes.social_security_contribution
    assert list(scale.rates) == list(expected_scale.rates)
    assert list(scale.thresholds) == list(expected_scale.thresholds)


def test_cases_give_the_same_results_as_on_a_full_clone(tmp_path):
    baseline = CountryTaxBenefitSystem()
    situation = {"salary": 50_000}
    inputs = {**situation, **{key: value for key, value in PARAMETER_INPUTS}}
    variables = ["income_tax", "social_security_contribution"]
    cases = []
    for reforms in [[], [SCALE_REFORM]]:
        reference = _full_clone_system(baseline, reforms, PARAMETER_INPUTS)
        builder = SimulationBuilder()
        builder.set_default_period("2017-01")
        simulation = builder.build_from_dict(reference, dict(situation))
        expected = {
            variable: float(simulation.calculate(variable, "2017-01")[0])
            for variable in variables
        }
        assert expected["income_tax"] == pytest.approx(15_000)
        for variable in variables:
            cases.append(
                {
                    "name": f"{variable} with reforms {reforms}",
                    "period": "2017-01",
                    **({"reforms": reforms} if reforms else {}),
                    "input": inputs,
                    "output": {variable: expected[variable]},
                    "absolute_error_margin": 0.01,
                }
            )
    # The reform changes the scale, so the two contributions must differ for
    # this to show the inputs were applied on top of it.
    assert cases[1]["output"] != cases[3]["output"]
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump(cases, sort_keys=False))
    assert _run(path, baseline) == 0


def test_cases_with_parameter_inputs_add_nothing_to_the_reform_cache(tmp_path):
    baseline = CountryTaxBenefitSystem()
    path = write_cases(tmp_path / "cases.yaml", 30, reform_every=1)
    assert _run(path, baseline) == 0
    cached = _tax_benefit_system_cache[baseline]
    assert cached.reform_free is not None
    assert len(cached.reforms) == 0


def test_an_unknown_parameter_input_fails_its_case(tmp_path, capsys):
    path = tmp_path / "unknown.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "unknown",
                    "period": "2017-01",
                    "input": {"salary": 1000, "taxes.no_such_tax": 1},
                    "output": {"income_tax": 150},
                }
            ]
        )
    )
    assert _run(path, CountryTaxBenefitSystem()) == 1
    assert "Could not find the parameter taxes.no_such_tax" in capsys.readouterr().out
