# Engineering Skills

This directory is the canonical source for AI-facing engineering rules.

Tool-specific instruction files such as `AGENTS.md`, `CLAUDE.md`, and
`.github/copilot-instructions.md` should point here instead of duplicating
implementation-specific guidance. When a rule changes, update the skill here
first, then keep adapters thin.

Current skills:

- `documentation_review.md`: model-neutral review checklist for public API,
  architecture, documentation, and generated artifact changes.
- `github-prs.md`: canonical PR workflow, required changelog fragments, PR head
  verification, and title conventions.
- `testing.md`: test placement, fixture scope, command, and environment
  expectations.
