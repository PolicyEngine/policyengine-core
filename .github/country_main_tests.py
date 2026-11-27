#!/usr/bin/env python
"""Run policyengine-us main's Core-facing tests against the Core under test.

The country smoke job installs a pinned policyengine-us release and checks
that its system loads, so it cannot see a policyengine-us test that a Core
change breaks. On 2026-10-27, 3.33.13 broke policyengine-us main's
test_social_security_input_state_machine.py within minutes of its release.

The PolicyEngineUSMain job checks out policyengine-us main and runs the files
in .github/requirements/policyengine-us-main-tests.txt twice, each with a
JUnit report: first with the Core release on PyPI, then with this checkout.

- `select LIST REPO` prints the listed files that exist in REPO, one per
  line, and warns about the rest, which policyengine-us renamed or removed.
- `compare RELEASED CANDIDATE` fails on each test that fails with this Core
  but did not fail with the release. A test that fails with both is a
  policyengine-us failure, so it is only a warning: policyengine-us main is
  sometimes red for its own reasons.
"""

import argparse
import os
import sys
from pathlib import Path
from xml.etree import ElementTree


def listed_files(list_path):
    """The paths in a list file, without comments or blank lines."""
    paths = []
    for line in Path(list_path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            paths.append(line)
    return paths


def select(list_path, repo):
    """The listed files that exist in the checkout, and those that do not."""
    present, missing = [], []
    for path in listed_files(list_path):
        (present if (Path(repo) / path).is_file() else missing).append(path)
    return present, missing


def outcomes(report):
    """Each test's outcome in a JUnit report: passed, failed or skipped.

    A test can appear twice, as when its teardown errors after it passed;
    a failure in either entry counts.
    """
    results = {}
    for case in ElementTree.parse(report).getroot().iter("testcase"):
        test = f"{case.get('classname', '')}::{case.get('name', '')}"
        tags = {child.tag for child in case}
        if tags & {"failure", "error"}:
            outcome = "failed"
        elif "skipped" in tags:
            outcome = "skipped"
        else:
            outcome = "passed"
        if results.get(test) != "failed":
            results[test] = outcome
    return results


def compare(released, candidate):
    """Tests this Core breaks, and tests that fail with both Cores."""
    broken = sorted(
        test
        for test, outcome in candidate.items()
        if outcome == "failed" and released.get(test) != "failed"
    )
    both = sorted(
        test
        for test, outcome in candidate.items()
        if outcome == "failed" and released.get(test) == "failed"
    )
    return broken, both


def summary(released, candidate, broken, both):
    lines = [
        "### policyengine-us main against this Core",
        "",
        f"{len(candidate)} tests ran with this Core and {len(released)} with "
        "the Core release.",
        "",
    ]
    if broken:
        lines.append(f"**This Core breaks {len(broken)}:**")
        lines += [f"- `{test}`" for test in broken]
        lines.append("")
    if both:
        lines.append(f"Failing with both Cores (policyengine-us's own): {len(both)}")
        lines += [f"- `{test}`" for test in both]
    if not broken and not both:
        lines.append("Every test passes with this Core.")
    return "\n".join(lines) + "\n"


def run_compare(released_report, candidate_report, summary_path=None):
    if not Path(candidate_report).exists():
        print(
            "::error::This Core's run wrote no JUnit report: pytest could not "
            "start or was killed."
        )
        return 1
    if not Path(released_report).exists():
        print(
            "::warning::The Core release's run wrote no JUnit report, so "
            "nothing can be blamed on this Core."
        )
        return 0
    released = outcomes(released_report)
    candidate = outcomes(candidate_report)
    broken, both = compare(released, candidate)
    for test in both:
        print(f"::warning::Fails with the Core release too: {test}")
    for test in broken:
        print(f"::error::This Core breaks {test}")
    text = summary(released, candidate, broken, both)
    print(text)
    if summary_path:
        with open(summary_path, "a") as output:
            output.write(text)
    return 1 if broken else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    chooser = commands.add_parser("select", help="list the files to run")
    chooser.add_argument("list")
    chooser.add_argument("repo")
    checker = commands.add_parser("compare", help="compare the two runs")
    checker.add_argument("released")
    checker.add_argument("candidate")
    checker.add_argument(
        "--summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="file to append a Markdown summary to",
    )
    args = parser.parse_args(argv)
    if args.command == "select":
        present, missing = select(args.list, args.repo)
        for path in missing:
            print(
                f"::warning::policyengine-us main has no {path}; update "
                ".github/requirements/policyengine-us-main-tests.txt.",
                file=sys.stderr,
            )
        if not present:
            print("::error::None of the listed files exist.", file=sys.stderr)
            return 1
        print("\n".join(present))
        return 0
    return run_compare(args.released, args.candidate, args.summary)


if __name__ == "__main__":
    sys.exit(main())
