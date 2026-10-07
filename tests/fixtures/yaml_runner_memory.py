"""Bounded helpers for YAML-runner process-lifetime memory tests."""

import gc
import sys
import tracemalloc
import weakref
from pathlib import Path

import pytest
import yaml

from policyengine_core.simulations import Simulation
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.tools.test_runner import OpenFiscaPlugin

RUNNER_ARGV = ["--capture", "no", "--maxfail", "0", "--tb", "short"]


def write_cases(path: Path, count: int, reform_every: int = 3) -> Path:
    """Write country-template income-tax cases with bounded policy variants."""

    cases = []
    for index in range(count):
        salary = 1_000 + index
        case = {
            "name": f"case {index}",
            "period": "2017-01",
            "input": {"salary": salary},
            "absolute_error_margin": 0.01,
        }
        if reform_every and index % reform_every == 0:
            rate = round(0.10 + index / 10_000, 4)
            case["input"]["taxes.income_tax_rate"] = rate
            case["output"] = {"income_tax": salary * rate}
        else:
            case["output"] = {"income_tax": salary * 0.15}
        cases.append(case)
    path.write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
    return path


def live(cls) -> weakref.WeakSet:
    """Return live instances of ``cls`` after a complete collection."""

    gc.collect()
    return weakref.WeakSet(item for item in gc.get_objects() if isinstance(item, cls))


class MemoryProbe:
    """Record Python allocations and objects retained across one pytest run."""

    def __init__(self, sample_every: int = 10):
        self.sample_every = sample_every
        self.traced = []
        self.outcomes = []
        self.live_simulations = None
        self.live_systems = None

    def pytest_sessionstart(self, session):
        self.simulations_before = live(Simulation)
        self.systems_before = live(TaxBenefitSystem)
        tracemalloc.start()

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.outcomes.append(report.outcome)

    def pytest_runtest_logfinish(self, nodeid, location):
        if len(self.outcomes) % self.sample_every == 0:
            gc.collect()
            self.traced.append(tracemalloc.get_traced_memory()[0])

    def pytest_sessionfinish(self, session, exitstatus):
        tracemalloc.stop()
        for name in ("last_type", "last_value", "last_traceback", "last_exc"):
            if hasattr(sys, name):
                delattr(sys, name)
        self.live_simulations = len(live(Simulation) - self.simulations_before)
        self.live_systems = len(live(TaxBenefitSystem) - self.systems_before)


def run_with_probe(tax_benefit_system, path: Path, options=None) -> MemoryProbe:
    """Run YAML cases with the runner-local cache and return memory samples."""

    probe = MemoryProbe()
    plugin = OpenFiscaPlugin(tax_benefit_system, options or {})
    try:
        exit_code = pytest.main(
            [*RUNNER_ARGV, "-q", "-p", "no:cacheprovider", str(path)],
            plugins=[plugin, probe],
        )
    finally:
        plugin.close()
    assert exit_code == 0
    assert probe.outcomes and set(probe.outcomes) == {"passed"}
    return probe
