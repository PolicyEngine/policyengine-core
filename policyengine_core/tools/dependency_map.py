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
A read of which children a node has, or of a parameter or node outside a dated
call (its value history, for example), is recorded at that node's own path and
stands for everything below it; the empty path is the whole tree. A look-up
by name that finds no child on a node at an instant is recorded at the missing
child's path.

Core's check of each variable's own abolition switch,
``gov.abolitions.<variable>``, is not recorded, so it is not in the map. A
formula's read of a switch is, like any other read.

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

Tests in one file mostly vary inputs for the same outputs and record the same
edges. By default a test is skipped when, earlier in its file, the formula
each of its outputs uses at the test's period has already run. A formula that
did not run, because ``defined_for`` was false for everyone or the value was
an input, does not count, so a later test of that output is still traced.
``every_test`` traces them all; it can still find more edges, because a
skipped test can reach formulas and conditional reads below its outputs that
earlier tests did not.

A test's ``output`` block is read the way the YAML runner reads it: a key is a
variable, an entity's singular key holding variables, or an entity's plural
key holding instances of variables, and a dict value keyed by period checks
the variable at each period.

Output
------
The map is one JSON object::

    generatedAt     ISO-8601 UTC timestamp
    model           package, version, fingerprint, coreVersion. The
                    fingerprint is a sha256 over the package's files and, when
                    tests were traced, the traced test files. Microdata is
                    not hashed.
    populations     per population traced, its selection and counts:
                    tests: root, everyTest, tests (situations built), skipped
                    (tests not traced because their outputs were covered),
                    excluded (tests with reforms, extensions or inline
                    parameter changes), failed (situations that could not be
                    built, and test files or entries that could not be read),
                    outputs
                    (variables calculated), outputErrors (output
                    calculations that raised; their edges are whatever
                    ran before the error); microdata: households, year,
                    variables (calculated), failed (raised at every period)
    tracingSeconds  wall time
    readers         parameter path -> sorted variables that read it
    consumers       variable -> sorted variables whose formula read it

The map records what ran. A test that could not be built contributes no
edges, and an output that raised contributes only the edges recorded before
it raised, so ``failed`` and ``outputErrors`` bound how incomplete a map may
be; the command prints both, and a map with non-zero counts should be read
as a lower bound on the model's dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from policyengine_core.periods import ETERNITY, MONTH
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.tools.test_runner import import_yaml
from policyengine_core.tracers import FullTracer, TraceNode

if TYPE_CHECKING:
    from policyengine_core.taxbenefitsystems import TaxBenefitSystem

yaml, YamlLoader = import_yaml()

Edges = tuple[dict[str, set[str]], dict[str, set[str]]]
Progress = Callable[[str], None]
YamlTest = dict[str, Any]
# A period as a YAML test writes it: a string, a year or a date.
YamlPeriod = Any
# A variable and the formula it uses at some period (None if it has none).
FormulaKey = tuple[str, Optional[Callable]]

DEFAULT_OUTPUT = Path("dependency-map.json")
YAML_SUFFIXES = (".yaml", ".yml")
COMPILED_SUFFIXES = (".pyc", ".pyo")


class _QuietFullTracer(FullTracer):
    """Keeps the trace's structure and drops the values, which nothing reads."""

    def record_calculation_result(self, value: object) -> None:
        pass

    def record_parameter_access(
        self, parameter: str, period: str, branch_name: str, value: object
    ) -> None:
        super().record_parameter_access(parameter, period, branch_name, None)


def _enable_tracing(simulation: Simulation) -> None:
    simulation.trace = True
    # A country package can hand the new tracer to its own parameter root when
    # tracing is switched on. Replacing the tracer would leave the root
    # recording into the old one, so the same tracer drops values instead.
    if type(simulation.tracer) is FullTracer:
        simulation.tracer.__class__ = _QuietFullTracer


def _trace_nodes(simulation: Simulation) -> Iterator[TraceNode]:
    pending_nodes = list(simulation.tracer.trees)
    while pending_nodes:
        node = pending_nodes.pop()
        yield node
        pending_nodes.extend(node.children)


def collect_edges(
    simulation: Simulation,
    readers: Optional[dict[str, set[str]]] = None,
    consumers: Optional[dict[str, set[str]]] = None,
) -> Edges:
    """Fold a traced simulation's trees into the two edge sets."""
    readers = {} if readers is None else readers
    consumers = {} if consumers is None else consumers
    for node in _trace_nodes(simulation):
        for parameter in node.parameters:
            readers.setdefault(parameter.name, set()).add(node.name)
        for child in node.children:
            if child.name != node.name:
                consumers.setdefault(child.name, set()).add(node.name)
    return readers, consumers


def iter_output_variables(
    output: object, system: Optional[TaxBenefitSystem] = None
) -> Iterator[tuple[str, Optional[YamlPeriod]]]:
    """Yield ``(variable, period)`` for each variable a test's output checks.

    Mirrors ``YamlItem.check_output``: a key is a variable, an entity's
    singular key holding variables, or an entity's plural key holding
    instances of variables. A dict value keyed by period checks the variable
    at each of those periods; ``period`` is ``None`` for the test's own
    period. Without a ``system`` every key is taken to be a variable.
    """
    if not isinstance(output, dict):
        return
    singular_keys = set() if system is None else {e.key for e in system.entities}
    plural_keys = set() if system is None else {e.plural for e in system.entities}

    def variable_periods(name: str, expected: object, period: object = None):
        if isinstance(expected, dict):
            for nested_period, nested_expected in expected.items():
                yield from variable_periods(name, nested_expected, nested_period)
        else:
            yield name, period

    for key, value in output.items():
        if system is None or system.get_variable(key) is not None:
            yield from variable_periods(key, value)
        elif key in singular_keys and isinstance(value, dict):
            for name, expected in value.items():
                yield from variable_periods(name, expected)
        elif key in plural_keys and isinstance(value, dict):
            for instance in value.values():
                if isinstance(instance, dict):
                    for name, expected in instance.items():
                        yield from variable_periods(name, expected)
        # Any other key fails the YAML runner as an unknown variable.


def _yaml_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    return sorted(
        file
        for file in path.rglob("*")
        if file.suffix in YAML_SUFFIXES and file.is_file()
    )


def _load_yaml_tests(file: Path) -> list[object]:
    """The tests of a YAML file; like the runner, a lone mapping is one test."""
    tests = yaml.load(file.read_text(), Loader=YamlLoader)
    if tests is None:
        return []
    return tests if isinstance(tests, list) else [tests]


def _runs_on_baseline(test: YamlTest) -> bool:
    if test.get("reforms") or test.get("extensions"):
        return False
    inputs = test.get("input")
    if not isinstance(inputs, dict):
        return True  # building the situation will report an invalid input
    # A dotted key is an inline parameter change.
    return not any(isinstance(key, str) and "." in key for key in inputs)


def iter_yaml_tests(
    paths: Iterable[Path],
    system: Optional[TaxBenefitSystem] = None,
    stats: Optional[dict[str, int]] = None,
) -> Iterator[tuple[Path, YamlTest]]:
    """Yield (file, test) for the tests that run on the baseline system.

    With ``stats``, adds to its ``excluded`` count the tests that change the
    system, and to ``failed`` the files and entries that cannot be read.
    """
    counts = {"excluded": 0, "failed": 0} if stats is None else stats
    for path in paths:
        for file in _yaml_files(path):
            try:
                tests = _load_yaml_tests(file)
            except yaml.YAMLError:
                counts["failed"] += 1
                continue
            for test in tests:
                if not isinstance(test, dict):
                    counts["failed"] += 1
                elif test.get("output") and not _runs_on_baseline(test):
                    counts["excluded"] += 1
                elif any(
                    True for _ in iter_output_variables(test.get("output"), system)
                ):
                    yield file, test


def _formula_key(
    system: TaxBenefitSystem, variable_name: str, period: object
) -> FormulaKey:
    variable = system.get_variable(variable_name)
    if variable is None:
        return variable_name, None
    try:
        return variable_name, variable.get_formula(period)
    except ValueError:  # not a period: calculating this output will report it
        return variable_name, None


def _output_formulas(system: TaxBenefitSystem, test: YamlTest) -> set[FormulaKey]:
    return {
        _formula_key(system, name, output_period or test.get("period"))
        for name, output_period in iter_output_variables(test["output"], system)
    }


def _formulas_run(system: TaxBenefitSystem, simulation: Simulation) -> set[FormulaKey]:
    return {
        _formula_key(system, node.name, node.period)
        for node in _trace_nodes(simulation)
        if node.formula_ran
    }


def trace_yaml_tests(
    system: TaxBenefitSystem,
    paths: Iterable[Path],
    progress: Optional[Progress] = None,
    every_test: bool = False,
) -> tuple[Edges, dict[str, int]]:
    """Trace the baseline YAML tests under ``paths`` into edge sets.

    Unless ``every_test`` is set, a test is skipped once the formulas of all
    its outputs have run in an earlier test of the same file.
    """
    readers: dict[str, set[str]] = {}
    consumers: dict[str, set[str]] = {}
    stats = {
        "tests": 0,
        "skipped": 0,
        "excluded": 0,
        "failed": 0,
        "outputs": 0,
        "outputErrors": 0,
    }
    current_file: Optional[Path] = None
    covered_formulas: set[FormulaKey] = set()
    for index, (file, test) in enumerate(iter_yaml_tests(paths, system, stats)):
        if file != current_file:
            current_file = file
            covered_formulas = set()
        output_formulas = _output_formulas(system, test)
        if not every_test and output_formulas <= covered_formulas:
            stats["skipped"] += 1
            continue
        period = test.get("period")
        try:
            builder = SimulationBuilder()
            builder.set_default_period(period)
            simulation = builder.build_from_dict(system, test.get("input") or {})
            simulation.default_calculation_period = builder.default_period
            _enable_tracing(simulation)
            calculated_formulas: set[FormulaKey] = set()
            for name, output_period in iter_output_variables(test["output"], system):
                calculation_period = output_period or period
                try:
                    simulation.calculate(name, calculation_period)
                    stats["outputs"] += 1
                    calculated_formulas.add(
                        _formula_key(system, name, calculation_period)
                    )
                except Exception:  # what ran before the error is still traced
                    stats["outputErrors"] += 1
            collect_edges(simulation, readers, consumers)
            covered_formulas |= calculated_formulas & _formulas_run(system, simulation)
            stats["tests"] += 1
        except Exception:  # unbuildable situation: skip it
            stats["failed"] += 1
        if progress and index % 500 == 0:
            progress(f"  {index} tests traced ({file.name})")
    return (readers, consumers), stats


def trace_microdata(
    microsimulation_class: Callable[[], Simulation],
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
        elif variable.definition_period == MONTH:
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


def _hashable_files(root: Path) -> list[Path]:
    return sorted(
        file
        for file in root.rglob("*")
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix not in COMPILED_SUFFIXES
        and not any(part.startswith(".") for part in file.relative_to(root).parts)
    )


def _hash_files(digest: Any, root: Path, files: Iterable[Path], label: str) -> None:
    for file in files:
        digest.update(f"{label}/{file.relative_to(root).as_posix()}".encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")


def model_fingerprint(package_root: Path, tests_root: Optional[Path] = None) -> str:
    """sha256 over what the map depends on, so a consumer can tell whether a
    map still matches the model it runs against.

    It covers every file of the package except its ``tests`` directory, since
    formulas import helpers from anywhere in the package, and the files under
    ``tests_root``, which decide what the map traced.
    """
    package_tests = package_root / "tests"
    digest = hashlib.sha256()
    _hash_files(
        digest,
        package_root,
        [
            file
            for file in _hashable_files(package_root)
            if not file.is_relative_to(package_tests)
        ],
        "package",
    )
    if tests_root is not None:
        files = _hashable_files(tests_root) if tests_root.is_dir() else [tests_root]
        root = tests_root if tests_root.is_dir() else tests_root.parent
        _hash_files(digest, root, files, "tests")
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
            "fingerprint": model_fingerprint(
                package_root, tests_root if "tests" in populations else None
            ),
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


def main(parser: argparse.ArgumentParser) -> int:
    args = parser.parse_args()
    if args.tests_root and not Path(args.tests_root).exists():
        parser.error(f"--tests-root does not exist: {args.tests_root}")

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
    for name, counts in payload["populations"].items():
        summary = ", ".join(
            f"{key}: {value}"
            for key, value in counts.items()
            if key not in ("root", "everyTest")
        )
        progress(f"{name}: {summary}")
        if counts.get("skipped"):
            progress(
                f"{counts['skipped']} tests were skipped because their outputs' "
                "formulas had already run in the same file. They can reach other "
                "formulas, so use --every-test for the complete map."
            )
    progress(
        f"wrote {output}: {len(payload['readers'])} parameter paths, "
        f"{len(payload['consumers'])} consumed variables, "
        f"{payload['tracingSeconds']}s"
    )
    return 0
