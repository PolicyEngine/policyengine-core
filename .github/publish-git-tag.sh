#! /usr/bin/env bash

git tag `python .github/fetch_version.py`
git push --tags || true  # update the repository version
