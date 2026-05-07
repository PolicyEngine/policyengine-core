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
