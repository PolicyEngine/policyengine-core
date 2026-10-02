"""Helpers for the YAML runner memory tests: generated case files and a pytest
plugin that measures what a run keeps alive."""

import gc
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
    """Write ``count`` country-template income tax cases to ``path``.

    Every ``reform_every``-th case sets the income tax rate as a dotted
    parameter input with a value no other case uses, so it needs a reform
    system of its own; the others run on the reform-free system. ``0``
    writes no reform cases.
    """
    cases = []
    for i in range(count):
        salary = 1000 + i
        case = {
            "name": f"case {i}",
            "period": "2017-01",
            "input": {"salary": salary},
            "absolute_error_margin": 0.01,
        }
        if reform_every and i % reform_every == 0:
            rate = round(0.10 + i / 10_000, 4)
            case["input"]["taxes.income_tax_rate"] = rate
            case["output"] = {"income_tax": salary * rate}
        else:
            case["output"] = {"income_tax": salary * 0.15}
        cases.append(case)
    path.write_text(yaml.safe_dump(cases, sort_keys=False))
    return path


def live(cls) -> weakref.WeakSet:
    """The live instances of ``cls`` after a full collection."""
    gc.collect()
    return weakref.WeakSet(o for o in gc.get_objects() if isinstance(o, cls))


class MemoryProbe:
    """pytest plugin recording, after every ``sample_every``-th case, the bytes
    Python has allocated (``tracemalloc``, after a full collection) and, at the
    end of the session, how many simulations and tax-benefit systems are
    alive that the run created (not ones other tests left behind)."""

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
        self.live_simulations = len(live(Simulation) - self.simulations_before)
        self.live_systems = len(live(TaxBenefitSystem) - self.systems_before)


def run_with_probe(tax_benefit_system, path: Path, options=None) -> MemoryProbe:
    """Run ``path`` the way ``run_tests`` does, with a ``MemoryProbe``."""
    probe = MemoryProbe()
    exit_code = pytest.main(
        [*RUNNER_ARGV, "-q", "-p", "no:cacheprovider", str(path)],
        plugins=[OpenFiscaPlugin(tax_benefit_system, options or {}), probe],
    )
    assert exit_code == 0, f"YAML run failed with exit code {exit_code}"
    assert probe.outcomes and set(probe.outcomes) == {"passed"}
    return probe
