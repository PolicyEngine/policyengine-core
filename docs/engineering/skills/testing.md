# Testing Skill

Use this skill whenever adding, moving, or reviewing tests.

## Commands

Use `uv run` for Python commands so the repo environment is selected
consistently.

Common checks:

```bash
uv run pytest tests/core/enums/test_enum.py -v
uv run pytest tests/core/test_file.py::test_name -v
make format
make test
make documentation
```

Run the narrowest test that proves the change while working. Before handing off a
broader behavioral change, run the relevant focused tests and formatting check.

## Placement

- Put core package tests under `tests/core/`.
- Put smoke tests under `tests/smoke/`.
- Keep fixtures under `tests/fixtures/` unless pytest fixture discovery requires
  a local `conftest.py`.
- Do not add tests inside `policyengine_core/`.

## Fixture and dependency boundaries

- Keep root `tests/conftest.py` lightweight.
- Avoid network access, cloud credentials, or country package imports in ordinary
  unit tests unless the test is explicitly a smoke/integration check.
- Prefer small synthetic fixtures for regression tests.
- When fixing a bug, add a regression test that fails without the fix and passes
  with it unless the change is documentation-only.

## YAML suites and memory

`policyengine-core test <paths> -c <country_package>` runs every case in one
process. Its memory is bounded by design, and tests must keep it that way:

- A finished case releases its simulation (`YamlItem.teardown`). pytest keeps
  every collected item until the session ends, so anything stored on the item
  lives for the whole run.
- A case that names `reforms` or `extensions` runs on a system built for
  that combination: a full copy of the baseline system. The runner caches the
  `--reform-cache-size` most recently used of them (default 2) plus the
  reform-free system. Do not add a cache to the runner that grows with the
  number of cases or combinations.
- A case that sets a parameter with a dotted input key runs on a system of its
  own (`_ParameterInputReform`) that copies only the nodes on the paths to the
  parameters it sets (`policyengine_core.parameters.path_copy`) and shares the
  rest of the tree, with its cached at-instant values, with the system its
  `reforms` select. Nothing may change a shared node in place.
- Each system still caches its parameter tree at every distinct instant its
  cases read (`ParameterNode.get_at_instant`,
  `TaxBenefitSystem.get_parameters_at_instant`), so memory grows with the
  number of distinct dates a suite reads, not with the number of cases.
  Neither cache may hold a tracer: a read of a traced tree is wrapped for that
  read only.
- `tests/core/tools/test_runner/test_runner_memory.py` holds the regression
  tests: a few hundred cases in one process must not grow traced memory or
  the number of live simulations and systems, and traced cases at distinct
  dates must leave no tracer or recorded result alive.

Run one large suite at a time. A country package's whole YAML tree in one
process still holds the baseline system, the reform-free copy and the cached
reform systems at once; use the country package's own batch runner where it
has one (policyengine-us: `policyengine_us/tests/test_batched.py` and the
`make test-yaml-*` targets).

`policyengine_core.tools.forked_test_runner.run_batches_in_forks` runs batches
in children forked from a process that has already built the system, so a
batch skips the import and build, and the children share the parent's system
in memory until they write to it. `tests/core/tools/test_runner/test_forked_test_runner.py`
covers its exit codes, worker limit and timeout.

`--read-only-arrays` (option `read_only_arrays`) stores every array read-only
for the run, so a formula that writes into an array it read from another
variable fails at that line (`test_read_only_arrays.py`).
