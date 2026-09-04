"""Trace the parameter → variable dependency map of a country package.

Downstream tools need to know which variables a parameter path moves and
which variables feed which: validation matching against calibration targets
and scorecards, model-coverage audits, reform classifiers. Static scans of
formula source miss bracket, scale, and vectorised reads; the only exact
record is what the model reads at run time. This module runs simulations
under ``FullTracer`` and folds the trace trees into two edge sets:

    readers[parameter_path] -> variables whose formula read that parameter
    consumers[variable]     -> variables whose formula read that variable

Parameter paths are recorded at the node the formula indexed, so a bracket
read ``p.base.calc(age)`` is recorded as ``gov.irs.credits.ctc.amount.base``.

Populations
-----------
``tests``      builds a simulation for every YAML test under the package's
               tests directory (skipping tests that apply reforms,
               extensions, or inline parameter changes) and calculates the
               test's outputs. Deterministic, no data download, and country
               test suites deliberately exercise every program.
``microdata``  calculates every variable over a subsample of the package's
               default microdata. Broad, but a formula behind ``defined_for``
               only runs when someone in the sample qualifies.
``both``       the union.

Tests in one file mostly vary inputs for the same outputs and record the
same edges, so by default one test is kept per newly covered output
variable in each file; ``every_test`` traces them all.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

from policyengine_core.periods import ETERNITY
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.tracers import FullTracer

Edges = tuple[dict[str, set[str]], dict[str, set[str]]]
Progress = Callable[[str], None]

DEFAULT_OUTPUT = Path("dependency-map.json")
# Read by core's neutralisation check before a formula runs, not by the
# formula: a switch per variable, not a dependency.
IGNORED_PARAMETER_PREFIXES = ("gov.abolitions.",)
# The model surface the map depends on; hashed into the fingerprint.
MODEL_SURFACE = ("entities.py", "parameters", "system.py", "variables")


class _QuietFullTracer(FullTracer):
    """Trace values are never read back; dropping them keeps memory flat."""

    def record_calculation_result(self, value) -> None:
        pass


def _enable_tracing(simulation) -> None:
    simulation.trace = True
    simulation.tracer = _QuietFullTracer()
    simulation.tax_benefit_system.parameters.set_tracing(
        simulation.tracer, simulation.branch_name
    )


def collect_edges(simulation, readers=None, consumers=None) -> Edges:
    """Fold a traced simulation's trees into the two edge sets."""
    readers = defaultdict(set) if readers is None else readers
    consumers = defaultdict(set) if consumers is None else consumers
    seen: set[int] = set()

    def walk(node) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        for parameter in node.parameters:
            if not parameter.name.startswith(IGNORED_PARAMETER_PREFIXES):
                readers[parameter.name].add(node.name)
        for child in node.children:
            if child.name != node.name:
                consumers[child.name].add(node.name)
            walk(child)

    for tree in simulation.tracer.trees:
        walk(tree)
    return readers, consumers


def iter_yaml_tests(paths: Iterable[Path], every_test: bool = False):
    """Yield (file, test) for the tests worth tracing against the baseline."""
    for path in paths:
        files = sorted(path.rglob("*.yaml")) if path.is_dir() else [path]
        for file in files:
            tests = yaml.safe_load(file.read_text()) or []
            if not isinstance(tests, list):
                continue
            covered: set[str] = set()
            for test in tests:
                if not isinstance(test, dict):
                    continue
                inputs = test.get("input") or {}
                if test.get("reforms") or test.get("extensions"):
                    continue
                if any("." in key for key in inputs):
                    continue  # inline parameter change: not the baseline system
                outputs = test.get("output") or {}
                if not outputs:
                    continue
                if not every_test and covered >= set(outputs):
                    continue
                covered |= set(outputs)
                yield file, test


def trace_yaml_tests(
    system,
    paths: Iterable[Path],
    progress: Optional[Progress] = None,
    every_test: bool = False,
) -> tuple[Edges, dict[str, int]]:
    readers: dict[str, set[str]] = defaultdict(set)
    consumers: dict[str, set[str]] = defaultdict(set)
    stats = {"tests": 0, "failed": 0}
    for index, (file, test) in enumerate(iter_yaml_tests(paths, every_test)):
        period = test.get("period")
        try:
            builder = SimulationBuilder()
            builder.set_default_period(period)
            simulation = builder.build_from_dict(system, test.get("input") or {})
            simulation.default_calculation_period = builder.default_period
            _enable_tracing(simulation)
            for output in test["output"]:
                try:
                    simulation.calculate(output, period)
                except Exception:  # a failing test still traced what ran
                    pass
            collect_edges(simulation, readers, consumers)
            stats["tests"] += 1
        except Exception:  # unbuildable situation: skip it
            stats["failed"] += 1
        if progress and index % 500 == 0:
            progress(f"  {index} tests traced ({file.name})")
    return (readers, consumers), stats


def trace_microdata(
    microsimulation_class,
    households: int = 2000,
    year: int = 2026,
    progress: Optional[Progress] = None,
) -> tuple[Edges, dict[str, int]]:
    simulation = microsimulation_class()
    simulation = simulation.subsample(n=households, seed=0) or simulation
    _enable_tracing(simulation)
    variables = simulation.tax_benefit_system.variables
    stats = {"variables": 0, "failed": 0}
    for index, name in enumerate(sorted(variables)):
        variable = variables[name]
        if variable.definition_period == ETERNITY:
            periods = [ETERNITY]
        elif variable.definition_period == "month":
            periods = [f"{year}-01"]
        else:
            periods = [year, f"{year}-01"]
        for period in periods:
            try:
                simulation.calculate(name, period)
                stats["variables"] += 1
                break
            except Exception:  # any formula failure just skips the variable
                continue
        else:
            stats["failed"] += 1
        if progress and index % 500 == 0:
            progress(f"  {index}/{len(variables)} variables")
    return collect_edges(simulation), stats


def merge_edges(*edge_sets: Edges) -> Edges:
    readers: dict[str, set[str]] = defaultdict(set)
    consumers: dict[str, set[str]] = defaultdict(set)
    for edge_readers, edge_consumers in edge_sets:
        for path, names in edge_readers.items():
            readers[path] |= names
        for name, users in edge_consumers.items():
            consumers[name] |= users
    return readers, consumers


def model_fingerprint(package_root: Path) -> str:
    """sha256 over the files the map depends on, so consumers can tell
    whether a map still matches the model they run against."""
    digest = hashlib.sha256()
    for relative in MODEL_SURFACE:
        path = package_root / relative
        files = (
            [path]
            if path.is_file()
            else sorted(
                child
                for child in path.rglob("*")
                if child.is_file()
                and "__pycache__" not in child.parts
                and child.suffix not in {".pyc", ".pyo"}
            )
            if path.is_dir()
            else []
        )
        for file in files:
            digest.update(file.relative_to(package_root).as_posix().encode())
            digest.update(b"\0")
            digest.update(file.read_bytes())
            digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _package_version(package_name: str) -> Optional[str]:
    for distribution in (package_name, package_name.replace("_", "-")):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


def build_dependency_map(
    country_package_name: str,
    population: str = "tests",
    tests_root: Optional[Path] = None,
    every_test: bool = False,
    households: int = 2000,
    year: int = 2026,
    progress: Optional[Progress] = None,
) -> dict:
    package = importlib.import_module(country_package_name)
    package_root = Path(package.__file__).resolve().parent
    tests_root = tests_root or package_root / "tests"

    edge_sets: list[Edges] = []
    populations: dict[str, dict] = {}
    started = time.time()
    if population in ("tests", "both"):
        edges, stats = trace_yaml_tests(
            package.CountryTaxBenefitSystem(), [tests_root], progress, every_test
        )
        edge_sets.append(edges)
        populations["tests"] = {
            "root": str(tests_root),
            "everyTest": every_test,
            **stats,
        }
    if population in ("microdata", "both"):
        edges, stats = trace_microdata(
            package.Microsimulation, households, year, progress
        )
        edge_sets.append(edges)
        populations["microdata"] = {"households": households, "year": year, **stats}
    if not edge_sets:
        raise ValueError(f"unknown population {population!r}")

    readers, consumers = merge_edges(*edge_sets)
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "package": country_package_name,
            "version": _package_version(country_package_name),
            "fingerprint": model_fingerprint(package_root),
            "coreVersion": metadata.version("policyengine-core"),
        },
        "populations": populations,
        "tracingSeconds": round(time.time() - started),
        "readers": {path: sorted(names) for path, names in sorted(readers.items())},
        "consumers": {name: sorted(users) for name, users in sorted(consumers.items())},
    }


def write_dependency_map(payload: dict, output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return output


def main(parser) -> int:
    args = parser.parse_args()

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    from policyengine_core.scripts import detect_country_package

    country_package_name = args.country_package or detect_country_package()
    payload = build_dependency_map(
        country_package_name,
        population=args.population,
        tests_root=Path(args.tests_root) if args.tests_root else None,
        every_test=args.every_test,
        households=args.households,
        year=args.year,
        progress=progress,
    )
    output = write_dependency_map(payload, Path(args.output))
    progress(
        f"wrote {output}: {len(payload['readers'])} parameter paths, "
        f"{len(payload['consumers'])} consumed variables, "
        f"{payload['tracingSeconds']}s"
    )
    return 0
