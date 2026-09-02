#!/usr/bin/env bash

set -euo pipefail

version=$(python -c 'import tomllib; from pathlib import Path; print(tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"])')

if git rev-parse --verify --quiet "refs/tags/$version" >/dev/null
then
    tag_commit=$(git rev-parse "refs/tags/${version}^{commit}")
    head_commit=$(git rev-parse HEAD)
    if [ "$tag_commit" != "$head_commit" ]
    then
        echo "Tag $version identifies $tag_commit, not the release commit $head_commit." >&2
        exit 1
    fi
    echo "Tag $version already exists."
    exit 0
fi

git tag "$version"
git push origin "$version"
