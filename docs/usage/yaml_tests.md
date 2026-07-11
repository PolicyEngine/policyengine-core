# YAML unit tests

PolicyEngine country packages express most unit tests as YAML files executed by
`policyengine_core.tools.test_runner.run_tests` (or the
`policyengine-core test` CLI).

## Document shape

A file may contain either a single mapping or a list of mappings. Each mapping
is one test case. Unknown top-level keys are rejected.

Accepted top-level fields:

| Field | Role |
| --- | --- |
| `name` | Optional human-readable test name |
| `description` | Optional longer description |
| `period` | Default simulation period for the case |
| `input` | Input values (scalars or nested entity maps) |
| `output` | Expected values (required at runtime) |
| `absolute_error_margin` | Absolute tolerance passed to `assert_near` |
| `relative_error_margin` | Relative tolerance passed to `assert_near` |
| `reforms` | Reform module path(s) to apply before running |
| `extensions` | Extension module path(s) to load |
| `keywords` | Tags used with name filters |
| `only_variables` / `ignore_variables` | Variable filters for the runner |

## Input and output conventions

- **Scalar inputs**: top-level `input` keys are variable names.
- **Entity inputs**: nest under the entity plural, then instance id, then
  variable name (the same shape `SimulationBuilder.build_from_dict` accepts).
- **Dotted input keys**: treated as **inline parameter reforms** via
  `set_parameter` (see `policyengine_core/tools/test_runner.py`, which imports
  `set_parameter` and assembles `inline_reforms` while parsing `input`), not as
  person/household variables. This differs from OpenFisca Core, which uses an
  explicit `parameters` field.
- **Outputs**: may be direct variable names, entity singular keys, or entity
  plural → instance id → variable paths.

Example:

```yaml
- name: Basic income example
  period: 2023
  absolute_error_margin: 0.01
  input:
    age: 30
    employment_income: 10000
  output:
    income_tax: 0
```

## Portability notes

PolicyEngine's YAML format is the source of truth for in-repo validation. When
exporting or comparing tests with other engines:

1. Prefer a **documented subset** (scalar or simple nested entity maps, absolute
   margins only).
2. **Reject loudly** constructs that are engine-specific or ambiguous in a
   portable intermediate form: `reforms`, `extensions`, relative margins, dotted
   parameter overrides, expression strings, and nulls.
3. Keep intermediate interchange formats **outside** this repository unless a
   maintainer-approved use case needs them in-tree. External converters should
   pin the `policyengine-core` commit they verified against `test_runner.py`.

These rules keep native tests cheap to write while making cross-engine migration
tools deterministic rather than lossy.
