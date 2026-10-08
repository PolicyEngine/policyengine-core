# Documentation Review

Use this skill when reviewing a pull request that changes public APIs,
architecture, documentation, generated artifacts, or developer-facing workflows.

## Review goal

Documentation review is a harness check, not copyediting. Confirm that durable
documentation still describes the code paths a maintainer or AI agent would use
to understand, validate, or modify the system.

Do not put PR-specific confidence, impact, or reviewer notes into durable API
docs. Keep durable facts in source documentation and keep review judgment in the
PR description or review comment.

## Trigger files

Run documentation review when a PR changes any of these paths:

- `policyengine_core/`
- `docs/`
- `.github/`
- `README.md`
- `CONTRIBUTING.md`
- `pyproject.toml`
- generated documentation or package metadata

Also run it when a PR changes public import surfaces, command-line behavior,
test/development workflows, changelog tooling, or release behavior even if the
changed path is not listed above.

## Checks

- Public API changes are reflected in relevant docs or API reference pages.
- Developer workflow changes are reflected in `README.md`, contributing docs, or
  AI-facing skills when needed.
- Generated artifacts are not hand-edited unless the repo expects them to be
  checked in.
- Changelog fragment guidance still matches the towncrier config in
  `pyproject.toml`.
- Review notes distinguish facts proven by code or tests from assumptions.

## Review output

Use a concise PR-facing note. Include:

- Documentation changes observed, or `not required` with a reason.
- Commands run, or commands not run with a reason.
- Impact: `low`, `medium`, or `high`.
- Confidence: `low`, `medium`, or `high`.
- Known gaps that should be handled in this PR or a follow-up.
