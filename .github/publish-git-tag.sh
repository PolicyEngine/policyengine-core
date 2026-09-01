#!/usr/bin/env bash

set -euo pipefail

version=$(python -c 'import tomllib; from pathlib import Path; print(tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"])')

if git rev-parse --verify --quiet "refs/tags/$version" >/dev/null
then
    echo "Tag $version already exists."
    exit 0
fi

git tag "$version"
git push origin "$version"
