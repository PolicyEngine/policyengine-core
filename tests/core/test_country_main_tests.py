"""The PolicyEngineUSMain job blames Core only for the tests Core breaks.

It runs policyengine-us main's listed tests with the Core release and then
with the Core under test (.github/country_main_tests.py). These tests check
the comparison on synthetic JUnit reports, the file selection, the list, and
the order of the job's steps.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "country_main_tests.py"
LIST = ROOT / ".github" / "requirements" / "policyengine-us-main-tests.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "pr.yaml"
SPEC = importlib.util.spec_from_file_location("country_main_tests", SCRIPT)
country = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(country)


def junit(path, cases):
    """Write a JUnit report of (classname, name, outcome) cases."""
    body = ""
    for classname, name, outcome in cases:
        child = {
            "passed": "",
            "failed": '<failure message="assert"/>',
            "error": '<error message="teardown"/>',
            "skipped": '<skipped message="skip"/>',
        }[outcome]
        body += f'<testcase classname="{classname}" name="{name}">{child}</testcase>'
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    return path


def test_outcomes_count_a_failure_in_any_entry_of_a_test(tmp_path):
    report = junit(
        tmp_path / "r.xml",
        [
            ("a.test_x", "test_pass", "passed"),
            ("a.test_x", "test_fail", "failed"),
            ("a.test_x", "test_skip", "skipped"),
            ("a.test_x.TestCase", "runTest", "passed"),
            ("a.test_x.TestCase", "runTest", "error"),
        ],
    )
    assert country.outcomes(report) == {
        "a.test_x::test_pass": "passed",
        "a.test_x::test_fail": "failed",
        "a.test_x::test_skip": "skipped",
        "a.test_x.TestCase::runTest": "failed",
    }


def test_only_tests_failing_with_this_core_alone_are_blamed_on_it():
    released = {"t::a": "passed", "t::b": "failed", "t::c": "skipped"}
    candidate = {
        "t::a": "failed",  # this Core breaks it
        "t::b": "failed",  # policyengine-us's own failure
        "t::c": "failed",  # skipped before, fails now
        "t::d": "failed",  # absent before, such as a new collection error
        "t::e": "passed",
    }
    assert country.compare(released, candidate) == (
        ["t::a", "t::c", "t::d"],
        ["t::b"],
    )


@pytest.mark.parametrize(
    "released_cases, candidate_cases, status",
    [
        ([("t", "a", "passed")], [("t", "a", "failed")], 1),
        ([("t", "a", "failed")], [("t", "a", "failed")], 0),
        ([("t", "a", "passed")], [("t", "a", "passed")], 0),
        (None, [("t", "a", "failed")], 0),
        ([("t", "a", "passed")], None, 1),
    ],
    ids=["broken", "fails-with-both", "passes", "no-release-report", "no-report"],
)
def test_compare_fails_only_on_breaks_or_a_missing_report(
    tmp_path, capsys, released_cases, candidate_cases, status
):
    released = tmp_path / "released.xml"
    candidate = tmp_path / "candidate.xml"
    if released_cases is not None:
        junit(released, released_cases)
    if candidate_cases is not None:
        junit(candidate, candidate_cases)
    summary = tmp_path / "summary.md"
    assert (
        country.main(
            ["compare", str(released), str(candidate), "--summary", str(summary)]
        )
        == status
    )
    output = capsys.readouterr().out
    if status and candidate_cases is not None:
        assert "::error::This Core breaks t::a" in output
        assert "**This Core breaks 1:**" in summary.read_text()
    if released_cases == candidate_cases == [("t", "a", "failed")]:
        assert "::warning::Fails with the Core release too: t::a" in output


def test_select_skips_files_the_checkout_lacks(tmp_path, capsys):
    repo = tmp_path / "policyengine-us"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_kept.py").write_text("")
    listing = tmp_path / "list.txt"
    listing.write_text(
        "# comment\n\ntests/test_kept.py  # trailing comment\ntests/test_gone.py\n"
    )
    assert country.select(listing, repo) == (
        ["tests/test_kept.py"],
        ["tests/test_gone.py"],
    )
    assert country.main(["select", str(listing), str(repo)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "tests/test_kept.py\n"
    assert "::warning::policyengine-us main has no tests/test_gone.py" in captured.err
    listing.write_text("tests/test_gone.py\n")
    assert country.main(["select", str(listing), str(repo)]) == 1


def test_the_list_names_distinct_policyengine_us_test_files():
    paths = country.listed_files(LIST)
    assert paths, "the list is empty"
    assert len(paths) == len(set(paths)), "a file is listed twice"
    for path in paths:
        assert path.startswith("policyengine_us/tests/"), path
        assert Path(path).name.startswith("test_") and path.endswith(".py"), path


def test_the_job_runs_the_release_then_this_core_and_compares_them():
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["PolicyEngineUSMain"]
    names = [step["name"] for step in job["steps"]]
    assert names == [
        "Checkout repo",
        "Checkout policyengine-us main",
        "Set up Python",
        "Install uv",
        "Install policyengine-us main with the Core release",
        "Select the policyengine-us tests",
        "Run them with the Core release",
        "Install this Core",
        "Run them with this Core",
        "Fail on tests this Core breaks",
    ]
    steps = {step["name"]: step for step in job["steps"]}
    us = steps["Checkout policyengine-us main"]["with"]
    assert (us["repository"], us["ref"]) == ("PolicyEngine/policyengine-us", "main")
    released = steps["Run them with the Core release"]["run"]
    candidate = steps["Run them with this Core"]["run"]
    # The same tests and Hypothesis seed, with separate reports.
    assert released.splitlines()[-1].replace("released.xml", "candidate.xml") == (
        candidate
    )
    assert "--hypothesis-seed=0" in candidate
    # The candidate replaces the release even at the same version number,
    # and the job checks that it was installed from this checkout.
    install = steps["Install this Core"]["run"]
    assert "--reinstall-package policyengine-core ../core" in install
    assert "assert url.startswith('file://')" in install
    assert steps["Fail on tests this Core breaks"]["run"].endswith(
        "compare released.xml candidate.xml"
    )
