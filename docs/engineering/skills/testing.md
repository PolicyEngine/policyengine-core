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
- A case that names `reforms` or `extensions`, or sets a parameter with a
  dotted input key, runs on a system built for that combination: a full copy
  of the baseline system. The runner caches the `--reform-cache-size` most
  recently used of them (default 2) plus the reform-free system. Do not add a
  cache to the runner that grows with the number of cases or combinations.
- `tests/core/tools/test_runner/test_runner_memory.py` holds the regression
  tests: a few hundred cases in one process must not grow traced memory or
  the number of live simulations and systems.

Run one large suite at a time. A country package's whole YAML tree in one
process still holds the baseline system, the reform-free copy and the cached
reform systems at once; use the country package's own batch runner where it
has one (policyengine-us: `policyengine_us/tests/test_batched.py` and the
`make test-yaml-*` targets).
