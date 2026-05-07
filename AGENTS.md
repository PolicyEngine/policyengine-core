# Agent Instructions

These instructions apply repository-wide.

## Skills system

Canonical AI-facing engineering skills live under `docs/engineering/skills/`.
Use those files as the source of truth across Codex, Claude, Copilot, and other
AI tools.

When adding, moving, or reviewing tests, read
`docs/engineering/skills/testing.md`.

When reviewing changes to public APIs, architecture, documentation, or generated
artifacts, read `docs/engineering/skills/documentation_review.md`.

## GitHub PRs

Read `docs/engineering/skills/github-prs.md` before opening, replacing, or
sharing any pull request.

Before creating or sharing any PR, all developers and agents must:

1. Confirm the target remote is the canonical repository:
   `gh repo view PolicyEngine/policyengine-core --json nameWithOwner`.
2. Add a towncrier changelog fragment in `changelog.d/` using the format
   documented in `docs/engineering/skills/github-prs.md`.
3. Push the branch to `PolicyEngine/policyengine-core`.
4. Create the PR against `master`.
5. Verify the PR head repository before reporting it.

If you cannot push to the canonical repository, stop and ask for access. Do not
open a fork PR as a fallback unless the user explicitly asks for one.
