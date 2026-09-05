import json
from pathlib import Path

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.tools.dependency_map import (
    build_dependency_map,
    iter_yaml_tests,
    merge_edges,
    model_fingerprint,
    trace_yaml_tests,
    write_dependency_map,
)

import policyengine_core.country_template as country_template

TESTS_ROOT = Path(country_template.__file__).resolve().parent / "tests"


def test_trace_yaml_tests_records_parameter_and_variable_edges():
    (readers, consumers), stats = trace_yaml_tests(
        CountryTaxBenefitSystem(), [TESTS_ROOT / "income_tax.yaml"]
    )

    assert stats == {"tests": 1, "failed": 0}
    assert readers["taxes.income_tax_rate"] == {"income_tax"}
    assert consumers["salary"] == {"income_tax"}


def test_trace_yaml_tests_records_scale_reads():
    (readers, _), _ = trace_yaml_tests(
        CountryTaxBenefitSystem(), [TESTS_ROOT / "social_security_contribution.yaml"]
    )

    assert readers["taxes.social_security_contribution"] == {
        "social_security_contribution"
    }


def test_iter_yaml_tests_keeps_one_test_per_new_output(tmp_path: Path):
    (tmp_path / "cases.yaml").write_text(
        """
- name: plain
  period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
- name: with reform
  period: 2017-01
  reforms: policyengine_core.country_template.reforms.some_reform
  input: {salary: 1000}
  output: {income_tax: 150}
- name: inline parameter change
  period: 2017-01
  input: {salary: 1000, taxes.income_tax_rate: 0.5}
  output: {income_tax: 500}
- name: no output
  period: 2017-01
  input: {salary: 1000}
- name: same output again
  period: 2017-01
  input: {salary: 2000}
  output: {income_tax: 300}
- name: new output
  period: 2017-01
  input: {salary: 2000}
  output: {income_tax: 300, social_security_contribution: 40}
"""
    )

    names = [test["name"] for _, test in iter_yaml_tests([tmp_path])]
    every = [test["name"] for _, test in iter_yaml_tests([tmp_path], every_test=True)]

    assert names == ["plain", "new output"]
    assert every == ["plain", "same output again", "new output"]


def test_merge_edges_unions_both_sides():
    readers, consumers = merge_edges(
        ({"gov.a": {"x"}}, {"x": {"y"}}),
        ({"gov.a": {"z"}, "gov.b": {"w"}}, {"x": {"q"}}),
    )

    assert readers == {"gov.a": {"x", "z"}, "gov.b": {"w"}}
    assert consumers == {"x": {"y", "q"}}


def test_model_fingerprint_changes_with_the_model_surface(tmp_path: Path):
    (tmp_path / "parameters").mkdir()
    (tmp_path / "parameters" / "rate.yaml").write_text("values: {2017-01-01: 0.1}")
    before = model_fingerprint(tmp_path)

    (tmp_path / "parameters" / "rate.yaml").write_text("values: {2017-01-01: 0.2}")

    assert before.startswith("sha256:")
    assert model_fingerprint(tmp_path) != before


def test_build_and_write_dependency_map_for_the_country_template(tmp_path: Path):
    payload = build_dependency_map(
        "policyengine_core.country_template", tests_root=TESTS_ROOT
    )
    output = write_dependency_map(payload, tmp_path / "map.json")

    written = json.loads(output.read_text())
    assert written["model"]["package"] == "policyengine_core.country_template"
    assert written["model"]["fingerprint"].startswith("sha256:")
    assert written["model"]["coreVersion"]
    assert written["populations"]["tests"]["tests"] > 0
    assert "income_tax" in written["readers"]["taxes.income_tax_rate"]
    assert "income_tax" in written["consumers"]["salary"]
