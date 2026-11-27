import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import policyengine_core.country_template as country_template
from policyengine_core.country_template import (
    CountryTaxBenefitSystem,
    Microsimulation,
)
from policyengine_core.country_template.entities import Person
from policyengine_core.parameters import ParameterNode
from policyengine_core.periods import MONTH
from policyengine_core.scripts import policyengine_command
from policyengine_core.tools.dependency_map import (
    build_dependency_map,
    collect_edges,
    iter_yaml_tests,
    merge_edges,
    model_fingerprint,
    trace_microdata,
    trace_yaml_tests,
    write_dependency_map,
)
from policyengine_core.tracers import FullTracer
from policyengine_core.variables import Variable

TESTS_ROOT = Path(country_template.__file__).resolve().parent / "tests"
TEMPLATE_PACKAGE = "policyengine_core.country_template"


class surtax_eligible(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Whether the surtax applies"


class surtax(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "A tax that only runs for eligible people"
    defined_for = "surtax_eligible"

    def formula(person, period, parameters):
        return person("salary", period) * parameters(period).taxes.income_tax_rate


class counts_taxes(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Reads which children the taxes node has"

    def formula(person, period, parameters):
        return person.filled_array(len(parameters(period).taxes._children))


class undated_rate(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Reads a parameter before choosing an instant"

    def formula(person, period, parameters):
        return person.filled_array(parameters.taxes.income_tax_rate(period))


class raises_without_salary(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Raises before reading its parameter when nobody has a salary"

    def formula(person, period, parameters):
        salary = person("salary", period)
        if not salary.any():
            raise ValueError("no salary")
        return salary * parameters(period).taxes.income_tax_rate


@pytest.fixture
def system(isolated_tax_benefit_system):
    for variable in (
        surtax_eligible,
        surtax,
        counts_taxes,
        undated_rate,
        raises_without_salary,
    ):
        isolated_tax_benefit_system.add_variable(variable)
    return isolated_tax_benefit_system


def write_tests(directory: Path, content: str, name: str = "cases.yaml") -> Path:
    file = directory / name
    file.write_text(content)
    return file


INELIGIBLE_THEN_ELIGIBLE = """
- name: not yet eligible
  period: 2017-01
  input: {salary: 1000}
  output: {surtax: 0}
- name: eligible
  period: 2017-01
  input: {salary: 1000, surtax_eligible: 1}
  output: {surtax: 150}
- name: eligible again
  period: 2017-01
  input: {salary: 2000, surtax_eligible: 1}
  output: {surtax: 300}
"""


# ----- Edges ----- #


def test_trace_yaml_tests_records_parameter_and_variable_edges():
    (readers, consumers), stats = trace_yaml_tests(
        CountryTaxBenefitSystem(), [TESTS_ROOT / "income_tax.yaml"]
    )

    assert stats == {
        "tests": 1,
        "skipped": 0,
        "excluded": 0,
        "failed": 0,
        "outputs": 1,
        "outputErrors": 0,
    }
    assert readers["taxes.income_tax_rate"] == {"income_tax"}
    assert consumers["salary"] == {"income_tax"}


def test_trace_yaml_tests_records_scale_reads():
    (readers, _), _ = trace_yaml_tests(
        CountryTaxBenefitSystem(), [TESTS_ROOT / "social_security_contribution.yaml"]
    )

    assert readers["taxes.social_security_contribution"] == {
        "social_security_contribution"
    }


def test_map_holds_only_paths_that_exist(system, tmp_path: Path):
    write_tests(
        tmp_path,
        """
- period: 2017-01
  input: {salary: 1000}
  output: {counts_taxes: 3, undated_rate: 0.15}
""",
    )

    (readers, _), _ = trace_yaml_tests(system, [tmp_path])

    assert readers["taxes"] == {"counts_taxes"}
    assert readers["taxes.income_tax_rate"] == {"undated_rate"}
    assert not [path for path in readers if path.endswith("_children")]


class replacement_payment(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Paid only while the basic income is abolished"

    def formula(person, period, parameters):
        basic_income_abolished = parameters(period).gov.abolitions.basic_income
        return person.filled_array(100.0 * basic_income_abolished)


def test_formula_reading_another_variables_abolition_switch_keeps_its_edge(
    system, make_simulation
):
    """A formula that reads another variable's switch changes with it. Core's
    check of each variable's own switch is not the formula's read, and is not
    recorded."""
    system.parameters.add_child(
        "gov",
        ParameterNode(
            "gov",
            data={"abolitions": {"basic_income": {"2015-01-01": {"value": False}}}},
        ),
    )
    system.add_variable(replacement_payment)
    simulation = make_simulation(system, {"salary": 1000}, "2017-01")
    simulation.trace = True
    # A country package can trace its own parameter root, which sends core's
    # abolition check for every variable through this simulation's tracer.
    system.parameters.trace = True
    system.parameters.tracer = simulation.tracer
    system.parameters.branch_name = "default"

    simulation.calculate("replacement_payment", "2017-01")
    simulation.calculate("basic_income", "2017-01")
    readers, _ = collect_edges(simulation)

    assert readers["gov.abolitions.basic_income"] == {"replacement_payment"}
    assert "gov.abolitions.replacement_payment" not in readers


def test_trace_yaml_tests_follows_entity_scoped_and_period_keyed_outputs(
    tmp_path: Path,
):
    write_tests(
        tmp_path,
        """
- name: plural entity outputs
  period: 2017-01
  input:
    persons:
      alice: {salary: 1000}
    households:
      home: {parents: [alice]}
  output:
    persons:
      alice: {income_tax: 150}
- name: singular entity output keyed by period
  period: 2017-01
  input: {salary: 1000}
  output:
    person:
      social_security_contribution: {2017-01: 20}
- name: same variable again under the plural key
  period: 2017-01
  input:
    persons:
      bob: {salary: 3000}
    households:
      home: {parents: [bob]}
  output:
    persons:
      bob: {income_tax: 450}
""",
    )

    (readers, consumers), stats = trace_yaml_tests(
        CountryTaxBenefitSystem(), [tmp_path]
    )

    assert stats == {
        "tests": 2,
        "skipped": 1,
        "excluded": 0,
        "failed": 0,
        "outputs": 2,
        "outputErrors": 0,
    }
    assert readers["taxes.income_tax_rate"] == {"income_tax"}
    assert readers["taxes.social_security_contribution"] == {
        "social_security_contribution"
    }
    assert consumers["salary"] == {"income_tax", "social_security_contribution"}


def test_trace_yaml_tests_counts_outputs_that_fail_to_calculate(tmp_path: Path):
    write_tests(
        tmp_path,
        """
- name: an output the model does not define
  period: 2017-01
  input: {salary: 1000}
  output:
    person:
      no_such_variable: 1
      social_security_contribution: 20
""",
    )

    (readers, _), stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert stats["outputs"] == 1
    assert stats["outputErrors"] == 1
    assert readers["taxes.social_security_contribution"] == {
        "social_security_contribution"
    }


# ----- Which tests are traced ----- #


def test_iter_yaml_tests_keeps_tests_of_the_baseline_system(tmp_path: Path):
    write_tests(
        tmp_path,
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
""",
    )

    names = [test["name"] for _, test in iter_yaml_tests([tmp_path])]

    assert names == ["plain", "same output again"]


def test_a_file_holding_one_mapping_is_one_test(tmp_path: Path):
    write_tests(
        tmp_path,
        """
name: lone test
period: 2017-01
input: {salary: 1000}
output: {income_tax: 150}
""",
    )

    (readers, _), stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert stats["tests"] == 1
    assert readers["taxes.income_tax_rate"] == {"income_tax"}


def test_yml_files_are_traced_and_empty_files_ignored(tmp_path: Path):
    write_tests(
        tmp_path,
        """
- period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
""",
        "cases.yml",
    )
    write_tests(tmp_path, "", "empty.yaml")

    (readers, _), stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert stats["tests"] == 1
    assert readers["taxes.income_tax_rate"] == {"income_tax"}


def test_repeated_output_is_skipped_and_keeps_every_edge(tmp_path: Path):
    write_tests(
        tmp_path,
        """
- period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
- period: 2017-01
  input: {salary: 2000}
  output: {income_tax: 300}
- period: 2017-01
  input: {salary: 2000}
  output: {income_tax: 300, social_security_contribution: 40}
""",
    )
    system = CountryTaxBenefitSystem()

    selected_edges, selected_stats = trace_yaml_tests(system, [tmp_path])
    every_edges, every_stats = trace_yaml_tests(system, [tmp_path], every_test=True)

    assert (selected_stats["tests"], selected_stats["skipped"]) == (2, 1)
    assert (every_stats["tests"], every_stats["skipped"]) == (3, 0)
    assert selected_edges == every_edges


def test_output_masked_by_defined_for_stays_uncovered(system, tmp_path: Path):
    write_tests(tmp_path, INELIGIBLE_THEN_ELIGIBLE)

    selected_edges, stats = trace_yaml_tests(system, [tmp_path])
    every_edges, _ = trace_yaml_tests(system, [tmp_path], every_test=True)

    assert selected_edges[0]["taxes.income_tax_rate"] == {"surtax"}
    assert selected_edges[1]["salary"] == {"surtax"}
    assert selected_edges == every_edges
    assert (stats["tests"], stats["skipped"]) == (2, 1)


def test_output_given_as_an_input_stays_uncovered(tmp_path: Path):
    write_tests(
        tmp_path,
        """
- period: 2017-01
  input: {income_tax: 5}
  output: {income_tax: 5}
- period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
""",
    )

    (readers, _), stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert readers["taxes.income_tax_rate"] == {"income_tax"}
    assert stats["tests"] == 2


def test_output_that_raised_stays_uncovered(system, tmp_path: Path):
    write_tests(
        tmp_path,
        """
- name: the formula starts, then raises before its later reads
  period: 2017-01
  input: {salary: 0}
  output: {raises_without_salary: 0}
- name: the formula finishes
  period: 2017-01
  input: {salary: 1000}
  output: {raises_without_salary: 150}
""",
    )

    (readers, _), stats = trace_yaml_tests(system, [tmp_path])

    assert stats["outputErrors"] == 1
    assert stats["tests"] == 2
    assert readers["taxes.income_tax_rate"] == {"raises_without_salary"}


def test_each_dated_formula_of_an_output_is_traced(tmp_path: Path):
    write_tests(
        tmp_path,
        """
- name: formula from December 2016, which does not read salary
  period: 2017-01
  input: {age: 30, salary: 0}
  output: {basic_income: 600}
- name: formula from December 2015, which reads salary
  period: 2016-01
  input: {age: 30, salary: 0}
  output: {basic_income: 600}
""",
    )

    (_, consumers), stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert consumers["salary"] == {"basic_income"}
    assert stats["tests"] == 2


def test_coverage_does_not_carry_between_files(tmp_path: Path):
    test_text = """
- period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
"""
    write_tests(tmp_path, test_text, "first.yaml")
    write_tests(tmp_path, test_text, "second.yaml")

    _, stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert (stats["tests"], stats["skipped"]) == (2, 0)


def test_tests_outside_the_baseline_system_are_counted(tmp_path: Path):
    write_tests(
        tmp_path,
        """
- name: with reform
  period: 2017-01
  reforms: policyengine_core.country_template.reforms.some_reform
  output: {income_tax: 150}
- name: with extension
  period: 2017-01
  extensions: some_extension
  output: {income_tax: 150}
- name: inline parameter change
  period: 2017-01
  input: {salary: 1000, taxes.income_tax_rate: 0.5}
  output: {income_tax: 500}
- name: plain
  period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
""",
    )

    _, stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert (stats["excluded"], stats["tests"]) == (3, 1)


def test_unreadable_file_is_counted_and_later_files_are_traced(tmp_path: Path):
    write_tests(tmp_path, "- {unclosed", "first.yaml")
    write_tests(
        tmp_path,
        """
- period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
""",
        "second.yaml",
    )

    (readers, _), stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert (stats["failed"], stats["tests"]) == (1, 1)
    assert readers["taxes.income_tax_rate"] == {"income_tax"}


def test_test_with_invalid_input_is_counted_and_later_tests_are_traced(tmp_path: Path):
    write_tests(
        tmp_path,
        """
- period: 2017-01
  input: 3
  output: {income_tax: 0}
- period: 2017-01
  input: {1: 2}
  output: {income_tax: 0}
- period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
""",
    )

    (readers, _), stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert (stats["failed"], stats["tests"]) == (2, 1)
    assert readers["taxes.income_tax_rate"] == {"income_tax"}


def test_nested_period_mappings_are_followed_to_each_period(tmp_path: Path):
    write_tests(
        tmp_path,
        """
- period: 2017-01
  input: {salary: 1000}
  output:
    income_tax:
      2017:
        2017-01: 150
        2017-02: 150
""",
    )

    _, stats = trace_yaml_tests(CountryTaxBenefitSystem(), [tmp_path])

    assert (stats["outputs"], stats["outputErrors"]) == (2, 0)


# ----- Microdata ----- #


class rate_read_through_the_root(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Reads a parameter through the system's root, not its argument"

    def formula(person, period, parameters):
        root = person.simulation.tax_benefit_system.parameters
        return person.filled_array(root(period).taxes.income_tax_rate)


class RootTracingMicrosimulation(Microsimulation):
    """Hands its tracer to its own parameter root when tracing is switched on,
    as policyengine-us's simulations do."""

    @property
    def trace(self) -> bool:
        return Microsimulation.trace.fget(self)

    @trace.setter
    def trace(self, trace: bool) -> None:
        Microsimulation.trace.fset(self, trace)
        root = self.tax_benefit_system.parameters
        root.trace = bool(trace)
        root.tracer = self.tracer
        root.branch_name = self.branch_name
        root.clear_parent_cache()


def build_root_tracing_microsimulation() -> RootTracingMicrosimulation:
    system = CountryTaxBenefitSystem()
    system.add_variable(rate_read_through_the_root)
    return RootTracingMicrosimulation(tax_benefit_system=system)


def test_trace_microdata_records_reads_through_a_root_the_package_traces():
    (readers, _), _ = trace_microdata(
        build_root_tracing_microsimulation, households=2, year=2022
    )

    assert "rate_read_through_the_root" in readers["taxes.income_tax_rate"]


def test_trace_microdata_records_edges_of_every_calculated_variable():
    (readers, consumers), stats = trace_microdata(
        Microsimulation, households=2, year=2022
    )

    assert stats["variables"] > 0
    assert stats["failed"] == 0
    assert "income_tax" in readers["taxes.income_tax_rate"]
    assert "income_tax" in consumers["salary"]


# ----- Merging and fingerprints ----- #


def test_merge_edges_unions_both_sides():
    readers, consumers = merge_edges(
        ({"gov.a": {"x"}}, {"x": {"y"}}),
        ({"gov.a": {"z"}, "gov.b": {"w"}}, {"x": {"q"}}),
    )

    assert readers == {"gov.a": {"x", "z"}, "gov.b": {"w"}}
    assert consumers == {"x": {"y", "q"}}


@pytest.fixture
def package_root(tmp_path: Path) -> Path:
    root = tmp_path / "country"
    for relative_path, content in {
        "parameters/rate.yaml": "values: {2017-01-01: 0.1}",
        "variables/tax.py": "RATE = 1",
        "model_api.py": "HELPER = 1",
        "tools/helpers.py": "HELPER = 1",
        "tests/tax.yaml": "[]",
    }.items():
        file = root / relative_path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
    return root


@pytest.mark.parametrize(
    "changed_file",
    ["parameters/rate.yaml", "variables/tax.py", "model_api.py", "tools/helpers.py"],
)
def test_fingerprint_changes_with_any_model_file(package_root: Path, changed_file: str):
    before = model_fingerprint(package_root)

    (package_root / changed_file).write_text("CHANGED = 2")

    assert before.startswith("sha256:")
    assert model_fingerprint(package_root) != before


def test_fingerprint_changes_with_the_traced_tests(package_root: Path):
    tests_root = package_root / "tests"
    before = model_fingerprint(package_root, tests_root)

    (tests_root / "tax.yaml").write_text("- {}")

    assert model_fingerprint(package_root, tests_root) != before


def test_fingerprint_ignores_tests_that_were_not_traced(package_root: Path):
    before = model_fingerprint(package_root)

    (package_root / "tests" / "tax.yaml").write_text("- {}")

    assert model_fingerprint(package_root) == before


def test_fingerprint_ignores_compiled_files(package_root: Path):
    before = model_fingerprint(package_root)
    cache_directory = package_root / "variables" / "__pycache__"
    cache_directory.mkdir()

    (cache_directory / "tax.cpython-314.pyc").write_bytes(b"compiled")

    assert model_fingerprint(package_root) == before


def test_fingerprint_ignores_hidden_files(package_root: Path):
    before = model_fingerprint(package_root)

    (package_root / "variables" / ".DS_Store").write_bytes(b"folder settings")

    assert model_fingerprint(package_root) == before


# ----- The map and the command ----- #


def test_build_and_write_dependency_map_for_the_country_template(tmp_path: Path):
    payload = build_dependency_map(TEMPLATE_PACKAGE, tests_root=TESTS_ROOT)
    output = write_dependency_map(payload, tmp_path / "map.json")

    written = json.loads(output.read_text())
    assert written["model"]["package"] == TEMPLATE_PACKAGE
    assert written["model"]["fingerprint"].startswith("sha256:")
    assert written["model"]["coreVersion"]
    assert written["populations"]["tests"]["tests"] > 0
    assert "income_tax" in written["readers"]["taxes.income_tax_rate"]
    assert "income_tax" in written["consumers"]["salary"]


def test_command_writes_the_map(tmp_path: Path, monkeypatch, capsys):
    tests_directory = tmp_path / "tests"
    tests_directory.mkdir()
    write_tests(
        tests_directory,
        """
- period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
- period: 2017-01
  input: {salary: 2000}
  output: {income_tax: 300}
""",
    )
    output = tmp_path / "nested" / "map.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "policyengine-core",
            "dependency-map",
            "--country-package",
            TEMPLATE_PACKAGE,
            "--tests-root",
            str(tests_directory),
            "--every-test",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as command_exit:
        policyengine_command.main()

    assert command_exit.value.code == 0
    written = json.loads(output.read_text())
    assert written["populations"]["tests"]["everyTest"] is True
    assert written["populations"]["tests"]["tests"] == 2
    assert written["readers"]["taxes.income_tax_rate"] == ["income_tax"]
    assert "tests: 2" in capsys.readouterr().err


def run_command(monkeypatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["policyengine-core", *arguments])
    with pytest.raises(SystemExit) as command_exit:
        policyengine_command.main()
    return command_exit.value.code


def test_command_says_when_tests_were_skipped(tmp_path: Path, monkeypatch, capsys):
    tests_directory = tmp_path / "tests"
    tests_directory.mkdir()
    write_tests(
        tests_directory,
        """
- period: 2017-01
  input: {salary: 1000}
  output: {income_tax: 150}
- period: 2017-01
  input: {salary: 2000}
  output: {income_tax: 300}
""",
    )

    exit_code = run_command(
        monkeypatch,
        "dependency-map",
        "--country-package",
        TEMPLATE_PACKAGE,
        "--tests-root",
        str(tests_directory),
        "--output",
        str(tmp_path / "map.json"),
    )

    assert exit_code == 0
    assert "--every-test" in capsys.readouterr().err


def test_command_rejects_a_missing_tests_root(tmp_path: Path, monkeypatch, capsys):
    exit_code = run_command(
        monkeypatch,
        "dependency-map",
        "--country-package",
        TEMPLATE_PACKAGE,
        "--tests-root",
        str(tmp_path / "missing"),
        "--output",
        str(tmp_path / "map.json"),
    )

    assert exit_code == 2
    assert "missing" in capsys.readouterr().err
    assert not (tmp_path / "map.json").exists()
