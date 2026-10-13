"""Batches run in children forked from the process that built the system."""

import os
import time

import pytest
import yaml

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.tools import forked_test_runner
from policyengine_core.tools.forked_test_runner import (
    PYTEST_INTERNAL_ERROR,
    run_batches_in_forks,
)
from policyengine_core.tools.test_runner import _tax_benefit_system_cache
from tests.fixtures.yaml_runner_memory import RUNNER_ARGV

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork"),
    # The test process has native threads it did not start (the BLAS
    # library's pool, which Python counts), so every fork here warns. The
    # children only run pytest on the country template.
    pytest.mark.filterwarnings(
        "ignore:This process .* is multi-threaded:DeprecationWarning"
    ),
]

OPTIONS = {}


@pytest.fixture(autouse=True)
def nested_pytest_arguments(monkeypatch):
    """Run each child's session with the memory tests' arguments."""
    main = forked_test_runner.run_tests.__globals__["pytest"].main

    def nested_main(args, plugins):
        return main(
            [*args, "-q", "-p", "no:rerunfailures", "-p", "no:cacheprovider"],
            plugins=plugins,
        )

    monkeypatch.setattr(
        forked_test_runner.run_tests.__globals__["pytest"], "main", nested_main
    )


def _write_case(path, expected_income_tax):
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": path.stem,
                    "period": "2017-01",
                    "input": {"salary": 1000},
                    "output": {"income_tax": expected_income_tax},
                }
            ]
        )
    )
    return str(path)


def test_each_batch_reports_its_own_exit_code_and_output(tmp_path):
    passing = _write_case(tmp_path / "passing.yaml", 150)
    failing = _write_case(tmp_path / "failing.yaml", 999)
    finished = []

    results = run_batches_in_forks(
        CountryTaxBenefitSystem(),
        [[passing], [failing], [passing, failing]],
        OPTIONS,
        workers=2,
        on_finish=lambda index, result: finished.append(index),
    )

    assert [result.exit_code for result in results] == [0, 1, 1]
    assert "1 passed" in results[0].output
    assert "1 failed" in results[1].output
    assert "1 failed, 1 passed" in results[2].output
    assert sorted(finished) == [0, 1, 2]
    assert all(result.elapsed > 0 for result in results)
    assert not any(result.timed_out for result in results)


def test_the_reform_free_system_is_built_once_before_forking(tmp_path):
    system = CountryTaxBenefitSystem()
    run_batches_in_forks(system, [[_write_case(tmp_path / "case.yaml", 150)]])
    # Built in the parent, where every child shares it.
    assert _tax_benefit_system_cache[system].reform_free is not None


def test_no_more_than_workers_batches_run_at_once(tmp_path, monkeypatch):
    def record_while_running(tax_benefit_system, paths, options):
        (marker,) = paths
        with open(marker, "a") as log:
            log.write(f"start {time.monotonic()}\n")
        time.sleep(0.5)
        with open(marker, "a") as log:
            log.write(f"end {time.monotonic()}\n")
        return 0

    monkeypatch.setattr(forked_test_runner, "run_tests", record_while_running)
    markers = [str(tmp_path / f"batch-{index}.log") for index in range(5)]
    results = run_batches_in_forks(
        CountryTaxBenefitSystem(), [[marker] for marker in markers], workers=2
    )

    assert [result.exit_code for result in results] == [0] * 5
    events = []
    for marker in markers:
        with open(marker) as log:
            for line in log:
                kind, moment = line.split()
                events.append((float(moment), 1 if kind == "start" else -1))
    running = most = 0
    for _, change in sorted(events):
        running += change
        most = max(most, running)
    assert most == 2


def test_a_batch_over_its_timeout_is_stopped(monkeypatch):
    monkeypatch.setattr(
        forked_test_runner, "run_tests", lambda *args: time.sleep(60) or 0
    )
    started = time.monotonic()
    (result,) = run_batches_in_forks(
        CountryTaxBenefitSystem(), [["anything"]], timeout=0.5
    )
    assert result.timed_out
    assert result.exit_code < 0
    assert time.monotonic() - started < 30


def test_an_error_outside_pytest_is_reported_as_an_internal_error(monkeypatch):
    def fail(*args):
        raise RuntimeError("the batch could not start")

    monkeypatch.setattr(forked_test_runner, "run_tests", fail)
    (result,) = run_batches_in_forks(CountryTaxBenefitSystem(), [["anything"]])
    assert result.exit_code == PYTEST_INTERNAL_ERROR
    assert "the batch could not start" in result.output


def test_workers_must_be_positive():
    with pytest.raises(ValueError, match="workers must be 1 or more"):
        run_batches_in_forks(CountryTaxBenefitSystem(), [], workers=0)
