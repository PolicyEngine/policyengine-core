# PolicyEngine Core

[![codecov](https://codecov.io/gh/PolicyEngine/policyengine-core/branch/master/graph/badge.svg?token=BLoCjCf5Qr)](https://codecov.io/gh/PolicyEngine/policyengine-core)
[![PyPI version](https://badge.fury.io/py/policyengine-core.svg)](https://badge.fury.io/py/policyengine-core)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package, a fork of [OpenFisca-Core](https://github.com/OpenFisca/OpenFisca-Core), powers PolicyEngine country models and apps.

# Prerequisites

Python 3.10 or beyond is required.

# Setting Up

## 1. Fork the repo

```
https://github.com/PolicyEngine/policyengine-core/fork
```

## 2. Clone your own fork

## 3. Install dependencies

```
make install
```

If you are using Windows (not recommended), please install `make` first.
You can either directly download from [Make for Windows](https://gnuwin32.sourceforge.net/packages/make.htm); 
or install [Chocolatey](https://chocolatey.org/install), and simply install `make` with

```
choco install make
```

(See **Acknowledgements** for credit)

# Contributing

## Choosing an Issue

All of our code changes are made against a GitHub issue. If you're new to the project, go to **Issues** and search for good first issues `label: "good first issue"`. If you see an open issue that no one's opened a PR against, it's all yours! Feel free to make some edits, then open a PR, as described below.

## Developing

Keep your fork's `master` branch in sync with the original repo by pulling the original repo's code at times; typically (if the original repo is called "upstream" by Git) this means running `git pull upstream master`, then `git push origin master` to sync the code to your local repo. 

Create branches on your fork off of your master or main branch. Periodically, if you're working on something for a while, you might also run `git rebase master` within your feature branch to sync your code with any new changes.

## Testing, Formatting, Changelogging

You've finished your contribution, but now what? Before opening a PR, we ask contributors to do three things.

### Step 1: Testing

To test your changes against our series of automated tests, run

```
make test
```

We also ask that you add tests for any new features or bug-fixes you add, so we can gradually build up the code coverage. Our tests are written in the Python standard, [Pytest](https://docs.pytest.org/en/7.1.x/getting-started.html), and will be run again against the production environment, as well.

### Step 2: Formatting

In addition to the tests, we use [Black](https://github.com/psf/black) to lint our codebase, so before opening a pull request, Step 2 is to lint the code by running

```
make format
```

This will automatically format the code for you; no need to do anything else.

### Step 3: Changelogging

Finally, we ask contributors to make it clear for our users what changes have been made by contributing to a changelog. This changelog is formatted in YAML and describes the changes you've made to the code. This should follow the below format:

```
- bump: {major, minor, patch}
  changes:
    {added, removed, changed, fixed}:
      - <variable or program>
```

For more info on the syntax, check out the [semantic versioning docs](https://www.semver.org) and [keep a changelog](https://www.keepachangelog.com).

Write your changelog info into the empty file called `changelog_entry.yaml`. When you open your PR, this will automatically be added to the overall changelog.

## Opening a Pull Request

Now you've finished your contribution! Please open a pull request (PR) from your branch to the live `master` branch and request review. At times, it may take some time for the team to review your PR, especially for larger contributions, so please be patient--we will be sure to get to it.

In the first line of your PR, please make sure to include the following:

```
Fixes {issue_number}
```

This makes it much easier for us to maintain and prune our issue board.

Please try to be detailed in your PRs about the changes you made and why you made them. You may find yourself looking back at them for reference in the future, or needing insight about someone else's changes. Save yourself a conversation and write it all in the PR!

Here are some [best practices](https://deepsource.io/blog/git-best-practices/) for using Git.

When you're ready for review, switch the PR from `Draft` to `Ready for review`.

# License

Distributed under the AGPL License. See `LICENSE` for more info.

# Acknowledgements

- Thanks to Othneil Drew for his [README template](https://github.com/othneildrew/Best-README-Template).
- [Installing make on Windows](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows)
