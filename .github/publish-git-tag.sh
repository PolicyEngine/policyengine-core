#! /usr/bin/env bash

git tag `python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"`
git push --tags || true  # update the repository version
