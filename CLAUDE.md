# Claude Instructions

These instructions apply repository-wide.

## Canonical guidance

Repository-wide AI-facing engineering guidance lives in `AGENTS.md`.
Canonical skills live under `docs/engineering/skills/`.

Use those files as the source of truth. This file is a Claude adapter and should
stay thin; do not duplicate detailed testing, CI, formatting, or architecture
rules here.

## Required skill lookup

Before opening, replacing, or sharing a PR, read
`docs/engineering/skills/github-prs.md`. Add the required towncrier changelog
fragment before creating the PR.

When adding, moving, or reviewing tests, read
`docs/engineering/skills/testing.md` before editing.

When reviewing changes to public APIs, architecture, documentation, or generated
artifacts, read `docs/engineering/skills/documentation_review.md`.
