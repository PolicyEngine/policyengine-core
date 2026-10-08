# GitHub PRs

These rules apply to every developer and AI agent opening pull requests in this
repository.

## Repository and branch

Open PRs against `master` in `PolicyEngine/policyengine-core`.

Before creating or sharing a PR:

1. Confirm the canonical repository is reachable:
   `gh repo view PolicyEngine/policyengine-core --json nameWithOwner`.
2. Add the required towncrier changelog fragment under `changelog.d/`.
3. Push the current branch to `PolicyEngine/policyengine-core`.
4. Create the PR against `master`.
5. Verify the PR head repository before reporting it:
   `gh pr view <PR> --repo PolicyEngine/policyengine-core --json headRepositoryOwner,headRepository`.

If you cannot push to the canonical repository, stop and ask for access. Do not
open a fork PR as a fallback unless the user explicitly asks for one.

## Changelog fragment

A changelog entry is required before opening, replacing, or sharing a PR. When a
user asks an AI agent to open a PR, the agent must check for an appropriate
fragment and add one if it is missing before running `gh pr create`.

Use towncrier fragments in this format:

```text
changelog.d/<short-slug>.<type>.md
```

Allowed `<type>` values are configured in `pyproject.toml`:

- `breaking`
- `added`
- `changed`
- `fixed`
- `removed`

Examples:

```text
changelog.d/fix-enum-utf8-bytes.fixed.md
changelog.d/add-agent-pr-guidance.changed.md
```

Write one concise Markdown sentence in the fragment. Use `fixed` for bug fixes,
`added` for new user-facing capabilities, `changed` for behavior, documentation,
tooling, or refactors, `removed` for removals, and `breaking` only for changes
that intentionally break compatibility. Prefer updating an existing branch
fragment over adding duplicate fragments for the same PR.

Do not run `make changelog` for a PR. The release workflow builds
`CHANGELOG.md` from fragments after merge.

## PR title

Do not add `[codex]`, `[claude]`, `[copilot]`, or other agent labels to PR
titles. Use a plain descriptive title.

## PR body

Keep the description concrete:

- Link the issue the PR fixes.
- Summarize the user-visible behavior change.
- List the tests or checks run.
- Call out any commands that were not run and why.
