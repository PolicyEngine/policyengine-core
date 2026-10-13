"""With ``read_only_arrays``, a formula that writes into an array it read from
another variable fails at that line instead of changing that variable."""

import numpy as np
import pytest
import yaml

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.periods import MONTH, period
from policyengine_core.tools.test_runner import OpenFiscaPlugin, run_tests
from policyengine_core.variables import Variable
from tests.fixtures.yaml_runner_memory import RUNNER_ARGV

JANUARY = period("2017-01")


class salary_plus_one(Variable):
    """Writes into the salary array it read: changes ``salary`` itself."""

    value_type = float
    entity = Person
    label = "Salary plus one, written into the salary array"
    definition_period = MONTH

    def formula(person, period, parameters):
        salary = person("salary", period)
        salary += 1
        return salary


@pytest.fixture
def read_only(monkeypatch):
    monkeypatch.setattr(InMemoryStorage, "read_only_arrays", True)


def test_stored_arrays_stay_writeable_by_default():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0, 2.0]), JANUARY)
    stored = storage.get(JANUARY)
    stored += 1
    assert list(storage.get(JANUARY)) == [2.0, 3.0]


def test_stored_arrays_are_read_only_and_the_callers_array_is_not(read_only):
    storage = InMemoryStorage(is_eternal=False)
    given = np.array([1.0, 2.0])
    storage.put(given, JANUARY)
    stored = storage.get(JANUARY)

    with pytest.raises(ValueError, match="read-only"):
        stored += 1
    assert given.flags.writeable
    assert list(storage.get(JANUARY)) == [1.0, 2.0]


def test_an_array_a_clone_copies_on_first_read_is_read_only(read_only):
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0, 2.0]), JANUARY)
    clone = storage.clone(share_arrays=True)

    copied = clone.get(JANUARY)
    with pytest.raises(ValueError, match="read-only"):
        copied[0] = 5
    assert list(storage.get(JANUARY)) == [1.0, 2.0]


def _system_with_a_writing_formula():
    system = CountryTaxBenefitSystem()
    system.add_variable(salary_plus_one)
    return system


def _cases(tmp_path):
    path = tmp_path / "writes.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "writes into salary",
                    "period": "2017-01",
                    "input": {"salary": 1000},
                    "output": {"salary_plus_one": 1001},
                }
            ]
        )
    )
    return path


def test_a_formula_writing_into_a_cached_array_passes_without_the_option(tmp_path):
    exit_code = pytest.main(
        [*RUNNER_ARGV, "-q", "-p", "no:cacheprovider", str(_cases(tmp_path))],
        plugins=[OpenFiscaPlugin(_system_with_a_writing_formula(), {})],
    )
    assert exit_code == 0


def test_the_option_fails_a_formula_writing_into_a_cached_array(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        "policyengine_core.tools.test_runner.pytest.main",
        _main_without_plugins_from_outside(),
    )
    exit_code = run_tests(
        _system_with_a_writing_formula(),
        str(_cases(tmp_path)),
        {"read_only_arrays": True},
    )
    assert exit_code == 1
    assert "read-only" in capsys.readouterr().out
    # The run restores the setting it found.
    assert InMemoryStorage.read_only_arrays is False


def _main_without_plugins_from_outside():
    """``pytest.main`` with the arguments the memory tests use, so the nested
    session does not load this session's rerun plugin."""
    main = pytest.main

    def nested_main(args, plugins):
        return main(
            [*args, "-p", "no:rerunfailures", "-p", "no:cacheprovider"], plugins=plugins
        )

    return nested_main


def test_command_line_passes_read_only_arrays(monkeypatch):
    from policyengine_core.scripts import policyengine_command, run_test

    seen = {}

    def fake_run_tests(tax_benefit_system, paths, options):
        seen.update(options)
        return 0

    monkeypatch.setattr(run_test, "run_tests", fake_run_tests)
    monkeypatch.setattr(run_test, "build_tax_benefit_system", lambda *args: object())
    for arguments, expected in [([], False), (["--read-only-arrays"], True)]:
        monkeypatch.setattr(
            "sys.argv", ["policyengine-core", "test", "cases.yaml", *arguments]
        )
        with pytest.raises(SystemExit):
            run_test.main(policyengine_command.get_parser())
        assert seen["read_only_arrays"] is expected
