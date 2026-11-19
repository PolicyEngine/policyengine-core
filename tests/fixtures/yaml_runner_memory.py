"""Helpers for the YAML runner memory tests: generated case files and a pytest
plugin that measures what a run keeps alive."""

import gc
import sys
import tracemalloc
import weakref
from pathlib import Path

import numpy as np
import pytest
import yaml

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.variables.taxes import (
    income_tax as template_income_tax,
)
from policyengine_core.simulations import Simulation
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.tracers import FullTracer
from policyengine_core.tools.test_runner import OpenFiscaPlugin

# pytest-rerunfailures 16.7 stores failed ExceptionInfo objects on collected
# items even when no reruns are requested, retaining the builder's simulation
# through traceback locals. Measure the runner's ownership without this dev
# plugin; the outer test session can still rerun these tests normally.
RUNNER_ARGV = [
    "--capture",
    "no",
    "--maxfail",
    "0",
    "--tb",
    "short",
    "-p",
    "no:rerunfailures",
]


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


def write_cases_at_distinct_dates(path: Path, count: int, people: int = 256) -> Path:
    """Write ``count`` income tax cases for ``people`` people each, every one
    in a month no other case uses, so each reads the parameters at an instant
    of its own."""
    cases = [
        {
            "name": f"case {i}",
            "period": f"{2017 + i // 12}-{1 + i % 12:02d}",
            "input": {"salary": [1000] * people},
            "output": {"income_tax": [150] * people},
        }
        for i in range(count)
    ]
    path.write_text(yaml.safe_dump(cases))
    return path


def system_reading_its_parameters() -> CountryTaxBenefitSystem:
    """A country-template system whose income tax formula reads the rate
    through the public ``TaxBenefitSystem.get_parameters_at_instant``, not
    through the formula's ``parameters`` argument."""
    system = CountryTaxBenefitSystem()

    class income_tax(template_income_tax):
        def formula(person, period, parameters):
            system_parameters = (
                person.simulation.tax_benefit_system.get_parameters_at_instant(
                    period.start
                )
            )
            return person("salary", period) * system_parameters.taxes.income_tax_rate

    system.update_variable(income_tax)
    return system


def mark_parameter_trees_traced(monkeypatch) -> None:
    """Have every simulation that switches tracing on mark its system's
    parameter tree traced with its own tracer and branch name.

    Core's simulations trace per call and leave the tree untraced; code
    outside core still marks it (policyengine-us's
    ``isolate_parameter_tracing`` marks the root of a tree no one else reads
    where it stands).
    """
    original = Simulation.trace

    def set_trace(simulation, trace):
        original.fset(simulation, trace)
        if trace:
            parameters = simulation.tax_benefit_system.parameters
            parameters.trace = True
            parameters.tracer = simulation.tracer
            parameters.branch_name = simulation.branch_name

    monkeypatch.setattr(Simulation, "trace", property(original.fget, set_trace))


def record_results(monkeypatch) -> list:
    """Weak references to every array a ``FullTracer`` records as a
    calculation result from now on."""
    refs = []
    original = FullTracer.record_calculation_result

    def record(tracer, value):
        if isinstance(value, np.ndarray):
            refs.append(weakref.ref(value))
        original(tracer, value)

    monkeypatch.setattr(FullTracer, "record_calculation_result", record)
    return refs


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
        self.items = []
        self.traced = []
        self.outcomes = []
        self.live_simulations = None
        self.live_systems = None
        self.live_tracers = None

    def pytest_sessionstart(self, session):
        self.simulations_before = live(Simulation)
        self.systems_before = live(TaxBenefitSystem)
        self.tracers_before = live(FullTracer)
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
        # Keep the collected items alive even after pytest.main returns, so
        # collecting the session itself cannot hide a runner-owned reference.
        self.items = list(session.items)
        # pytest keeps the last failure's traceback in ``sys.last_*``; that one
        # root is pytest's, not the runner's.
        for name in ("last_type", "last_value", "last_traceback", "last_exc"):
            if hasattr(sys, name):
                delattr(sys, name)
        self.live_simulations = len(live(Simulation) - self.simulations_before)
        self.live_systems = len(live(TaxBenefitSystem) - self.systems_before)
        self.live_tracers = len(live(FullTracer) - self.tracers_before)


def run_with_probe(
    tax_benefit_system, path: Path, options=None, outcome="passed"
) -> MemoryProbe:
    """Run ``path`` the way ``run_tests`` does, with a ``MemoryProbe``; every
    case must end with ``outcome``."""
    probe = MemoryProbe()
    exit_code = pytest.main(
        [*RUNNER_ARGV, "-q", "-p", "no:cacheprovider", str(path)],
        plugins=[OpenFiscaPlugin(tax_benefit_system, options or {}), probe],
    )
    assert exit_code == (0 if outcome == "passed" else 1), exit_code
    assert probe.outcomes and set(probe.outcomes) == {outcome}
    return probe
