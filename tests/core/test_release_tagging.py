import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
TAG_SCRIPT = ROOT / ".github" / "publish-git-tag.sh"
PUSH_WORKFLOW = ROOT / ".github" / "workflows" / "push.yaml"


def run_git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def release_repository(tmp_path: Path) -> tuple[Path, Path]:
    if sys.platform == "win32":
        pytest.skip("the Bash release script is not exercised by Windows jobs")
    if shutil.which("bash") is None:
        pytest.skip("bash is required to exercise the release tag script")

    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    run_git("init", "--bare", str(remote), cwd=tmp_path)
    run_git("init", str(repository), cwd=tmp_path)
    run_git("config", "user.name", "Release Test", cwd=repository)
    run_git("config", "user.email", "release-test@example.com", cwd=repository)
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "release-test"\nversion = "1.2.3"\n'
    )
    run_git("add", "pyproject.toml", cwd=repository)
    run_git("commit", "-m", "Prepare release", cwd=repository)
    run_git("remote", "add", "origin", str(remote), cwd=repository)
    return repository, remote


def run_tag_script(
    repository: Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(TAG_SCRIPT)],
        cwd=repository,
        check=check,
        capture_output=True,
        text=True,
    )


def test_release_tag_script_pushes_exact_pyproject_version(
    release_repository: tuple[Path, Path],
):
    repository, remote = release_repository

    run_tag_script(repository)

    head = run_git("rev-parse", "HEAD", cwd=repository).stdout.strip()
    local_tag = run_git("rev-parse", "refs/tags/1.2.3", cwd=repository)
    remote_tag = run_git(
        "--git-dir",
        str(remote),
        "rev-parse",
        "refs/tags/1.2.3",
        cwd=repository,
    )
    assert local_tag.stdout.strip() == head
    assert remote_tag.stdout.strip() == head
    assert run_git("tag", "--list", cwd=repository).stdout.splitlines() == ["1.2.3"]


def test_release_tag_script_is_idempotent(
    release_repository: tuple[Path, Path],
):
    repository, _ = release_repository
    run_tag_script(repository)

    second_run = run_tag_script(repository)

    assert second_run.stdout.strip() == "Tag 1.2.3 already exists."


def test_release_tag_script_rejects_tag_for_different_commit(
    release_repository: tuple[Path, Path],
):
    repository, remote = release_repository
    run_git("tag", "1.2.3", cwd=repository)
    run_git("push", "origin", "1.2.3", cwd=repository)
    (repository / "change.txt").write_text("Different release commit\n")
    run_git("add", "change.txt", cwd=repository)
    run_git("commit", "-m", "Create a different release commit", cwd=repository)

    result = run_tag_script(repository, check=False)

    head = run_git("rev-parse", "HEAD", cwd=repository).stdout.strip()
    tagged_commit = run_git(
        "rev-parse",
        "refs/tags/1.2.3^{commit}",
        cwd=repository,
    ).stdout.strip()
    remote_tag = run_git(
        "--git-dir",
        str(remote),
        "rev-parse",
        "refs/tags/1.2.3^{commit}",
        cwd=repository,
    ).stdout.strip()
    assert result.returncode != 0
    assert result.stdout == ""
    assert f"Tag 1.2.3 identifies {tagged_commit}" in result.stderr
    assert f"not the release commit {head}" in result.stderr
    assert tagged_commit != head
    assert remote_tag == tagged_commit


def test_release_tag_script_propagates_push_failure(tmp_path: Path):
    if sys.platform == "win32":
        pytest.skip("the Bash release script is not exercised by Windows jobs")
    if shutil.which("bash") is None:
        pytest.skip("bash is required to exercise the release tag script")

    repository = tmp_path / "repository"
    run_git("init", str(repository), cwd=tmp_path)
    run_git("config", "user.name", "Release Test", cwd=repository)
    run_git("config", "user.email", "release-test@example.com", cwd=repository)
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "release-test"\nversion = "1.2.3"\n'
    )
    run_git("add", "pyproject.toml", cwd=repository)
    run_git("commit", "-m", "Prepare release", cwd=repository)

    result = run_tag_script(repository, check=False)

    assert result.returncode != 0
    assert "origin" in result.stderr


def only_step(steps: list[dict], matches, description: str) -> int:
    indices = [index for index, step in enumerate(steps) if matches(step)]
    assert len(indices) == 1, f"expected one {description} step, found {indices}"
    return indices[0]


def assert_publish_tags_only_after_pypi_succeeds(publish: dict) -> None:
    steps = publish["steps"]
    checkout = steps[
        only_step(
            steps,
            lambda step: step.get("uses", "").startswith("actions/checkout@"),
            "checkout",
        )
    ]
    pypi_index = only_step(
        steps,
        lambda step: step.get("uses", "").startswith("pypa/gh-action-pypi-publish@"),
        "PyPI publish",
    )
    tag_index = only_step(
        steps,
        lambda step: ".github/publish-git-tag.sh" in step.get("run", ""),
        "git tag",
    )
    pypi_step = steps[pypi_index]
    tag_step = steps[tag_index]

    # The tag script pushes a tag, so Publish itself needs write access.
    # Permissions granted to any other job do not apply to it.
    permissions = publish.get("permissions", {})
    assert permissions == "write-all" or (
        isinstance(permissions, dict) and permissions.get("contents") == "write"
    ), "Publish lacks contents: write"
    # The script looks for an existing tag locally, and a checkout with
    # fetch-depth 0 fetches every tag. Action inputs are strings, so 0 and "0"
    # are the same input.
    assert str(checkout.get("with", {}).get("fetch-depth")) == "0", (
        "checkout is not fetch-depth 0"
    )
    # Steps run in order and a failure skips the rest, so the tag step runs
    # only after a successful upload. A condition on either step, or
    # continue-on-error on the upload, could let it run anyway: a skipped
    # step does not fail the job.
    assert pypi_index < tag_index, "tag step precedes the PyPI step"
    assert "if" not in pypi_step, "PyPI step has a condition"
    assert "if" not in tag_step, "tag step has a condition"
    assert pypi_step.get("continue-on-error", False) is False, (
        "PyPI step continues on error"
    )
    # A failed tag push must fail the job, not leave an untagged release, so
    # the step runs the script alone: nothing such as `|| true` may follow it.
    assert tag_step.get("continue-on-error", False) is False, (
        "tag step continues on error"
    )
    assert tag_step["run"].strip() == "bash .github/publish-git-tag.sh", (
        "tag step runs more than the tag script"
    )


def test_publish_workflow_tags_only_after_pypi_succeeds():
    publish = yaml.safe_load(PUSH_WORKFLOW.read_text())["jobs"]["Publish"]

    assert_publish_tags_only_after_pypi_succeeds(publish)


def drop_publish_permissions_and_add_later_writer(jobs: dict) -> None:
    # A text slice from "  Publish:" to the end of the file read the later
    # job's permissions as Publish's and passed this workflow.
    del jobs["Publish"]["permissions"]
    jobs["Docs"] = {
        "runs-on": "ubuntu-latest",
        "permissions": {"contents": "write"},
        "steps": [{"run": "make documentation"}],
    }


def find_step(jobs: dict, fragment: str) -> dict:
    return next(
        step
        for step in jobs["Publish"]["steps"]
        if fragment in step.get("uses", "") + step.get("run", "")
    )


def shallow_checkout(jobs: dict) -> None:
    del find_step(jobs, "actions/checkout@")["with"]["fetch-depth"]


def tag_before_pypi(jobs: dict) -> None:
    steps = jobs["Publish"]["steps"]
    tag_step = find_step(jobs, "publish-git-tag.sh")
    steps.remove(tag_step)
    steps.insert(steps.index(find_step(jobs, "gh-action-pypi-publish")), tag_step)


def ignore_tag_failure(jobs: dict) -> None:
    find_step(jobs, "publish-git-tag.sh")["run"] += " || true"


def continue_after_tag_failure(jobs: dict) -> None:
    find_step(jobs, "publish-git-tag.sh")["continue-on-error"] = True


def continue_after_pypi_failure(jobs: dict) -> None:
    find_step(jobs, "gh-action-pypi-publish")["continue-on-error"] = True


def tag_even_if_pypi_fails(jobs: dict) -> None:
    find_step(jobs, "publish-git-tag.sh")["if"] = "always()"


def skip_pypi_on_this_branch(jobs: dict) -> None:
    # The default branch is master, so this skips the upload on every release.
    find_step(jobs, "gh-action-pypi-publish")["if"] = "github.ref == 'refs/heads/main'"


@pytest.mark.parametrize(
    ("mutate", "violation"),
    [
        pytest.param(mutate, violation, id=mutate.__name__)
        for mutate, violation in [
            (
                drop_publish_permissions_and_add_later_writer,
                "Publish lacks contents: write",
            ),
            (shallow_checkout, "checkout is not fetch-depth 0"),
            (tag_before_pypi, "tag step precedes the PyPI step"),
            (skip_pypi_on_this_branch, "PyPI step has a condition"),
            (tag_even_if_pypi_fails, "tag step has a condition"),
            (continue_after_pypi_failure, "PyPI step continues on error"),
            (continue_after_tag_failure, "tag step continues on error"),
            (ignore_tag_failure, "tag step runs more than the tag script"),
        ]
    ],
)
def test_publish_workflow_check_rejects_unsafe_release_order(mutate, violation):
    jobs = yaml.safe_load(PUSH_WORKFLOW.read_text())["jobs"]
    mutate(jobs)

    with pytest.raises(AssertionError, match=violation):
        assert_publish_tags_only_after_pypi_succeeds(jobs["Publish"])


def grant_write_all(jobs: dict) -> None:
    jobs["Publish"]["permissions"] = "write-all"


def quote_fetch_depth(jobs: dict) -> None:
    find_step(jobs, "actions/checkout@")["with"]["fetch-depth"] = "0"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(mutate, id=mutate.__name__)
        for mutate in [grant_write_all, quote_fetch_depth]
    ],
)
def test_publish_workflow_check_accepts_equivalent_settings(mutate):
    jobs = yaml.safe_load(PUSH_WORKFLOW.read_text())["jobs"]
    mutate(jobs)

    assert_publish_tags_only_after_pypi_succeeds(jobs["Publish"])


def test_publish_workflow_publishes_only_after_tests_pass():
    # Without ``needs: Test``, Publish ran beside Test, so a release could
    # reach PyPI while its tests were still running or after they failed.
    jobs = yaml.safe_load(PUSH_WORKFLOW.read_text())["jobs"]
    publishing = [
        name
        for name, job in jobs.items()
        if any(
            "gh-action-pypi-publish" in step.get("uses", "")
            for step in job.get("steps", [])
        )
    ]

    assert publishing == ["Publish"]
    needs = jobs["Publish"].get("needs", [])
    assert "Test" in ([needs] if isinstance(needs, str) else needs)
    # A job that needs Test is skipped whenever Test is, so both must run on
    # the same commits for releases to happen at all.
    assert jobs["Publish"]["if"] == jobs["Test"]["if"]
    # A status function such as always() would let Publish run after a
    # failed Test; without one, Publish runs only when Test succeeds.
    # (Expression function names are matched case-insensitively.)
    assert not re.search(
        r"\b(always|failure|cancelled|success)\s*\(",
        jobs["Publish"]["if"],
        re.IGNORECASE,
    )


WORKFLOWS = ROOT / ".github" / "workflows"
ACTIONS = ROOT / ".github" / "actions"
MAKEFILE = ROOT / "Makefile"
PYPI_UPLOAD_COMMAND = re.compile(
    r"\b(twine\s+upload|(uv|poetry|flit|hatch|pdm|maturin)\s+publish)\b"
)
# A rule line such as `test: test-country-template`, but not `X := y`.
MAKE_RULE = re.compile(r"^([^\s:=#][^:=#]*?)\s*:(?![:=])(.*)$")


def uploads_to_pypi(step: dict) -> bool:
    return step.get("uses", "").startswith("pypa/gh-action-pypi-publish@") or bool(
        PYPI_UPLOAD_COMMAND.search(step.get("run", ""))
    )


def load_workflows(workflows: Path) -> list[tuple[str, dict]]:
    return [
        (path.name, yaml.safe_load(path.read_text()))
        for path in sorted([*workflows.glob("*.yaml"), *workflows.glob("*.yml")])
    ]


def pypi_publishing_jobs(workflows: Path) -> list[tuple[str, str]]:
    return [
        (file_name, name)
        for file_name, workflow in load_workflows(workflows)
        for name, job in (workflow.get("jobs") or {}).items()
        if any(uploads_to_pypi(step) for step in job.get("steps", []))
    ]


def remote_reusable_workflow_jobs(workflows: Path) -> list[tuple[str, str]]:
    # A job-level `uses:` that does not start with ./ calls a workflow in
    # another repository, such as owner/repo/.github/workflows/x.yml@ref, whose
    # steps this scan cannot read. A local reusable workflow lives in
    # .github/workflows/ and is scanned as a file of its own.
    return [
        (file_name, name)
        for file_name, workflow in load_workflows(workflows)
        for name, job in (workflow.get("jobs") or {}).items()
        if "uses" in job and not str(job["uses"]).startswith("./")
    ]


def pypi_publishing_actions(actions: Path) -> list[str]:
    if not actions.is_dir():
        return []
    paths = sorted([*actions.rglob("action.yml"), *actions.rglob("action.yaml")])
    return [
        path.relative_to(actions).as_posix()
        for path in paths
        if any(
            uploads_to_pypi(step)
            for step in (
                ((yaml.safe_load(path.read_text()) or {}).get("runs") or {}).get(
                    "steps"
                )
                or []
            )
        )
    ]


def make_rules(makefile: Path) -> dict[str, tuple[list[str], list[str]]]:
    """Each target's prerequisites and recipe lines (without the tab).

    Enough for this Makefile: it has no line continuations, includes or
    pattern rules.
    """
    rules = {}
    recipe = []
    for line in makefile.read_text().splitlines():
        if line.startswith("\t"):
            recipe.append(line[1:])
        elif rule := MAKE_RULE.match(line):
            recipe = []
            for target in rule.group(1).split():
                rules[target] = (rule.group(2).split(), recipe)
        elif line.strip():
            recipe = []
    return rules


def pypi_uploading_recipe_lines(makefile: Path) -> list[str]:
    return [
        f"{target}: {line}"
        for target, (_, recipe) in make_rules(makefile).items()
        for line in recipe
        if PYPI_UPLOAD_COMMAND.search(line)
    ]


def test_only_push_publish_job_uploads_to_pypi():
    # The checks above cover push.yaml's Publish job alone, so an upload in
    # any other job or workflow file would reach PyPI without them.
    # Recognises the pypa publish action, twine upload and
    # uv/poetry/flit/hatch/pdm/maturin publish in workflow steps, local
    # composite action steps and Makefile recipes, and rejects reusable
    # workflows from other repositories; not uploads inside scripts or remote
    # actions.
    assert pypi_publishing_jobs(WORKFLOWS) == [("push.yaml", "Publish")]
    assert remote_reusable_workflow_jobs(WORKFLOWS) == [], (
        "a job calls a reusable workflow from another repository"
    )
    assert pypi_publishing_actions(ACTIONS) == [], "a composite action uploads to PyPI"
    assert pypi_uploading_recipe_lines(MAKEFILE) == [], (
        "a Makefile recipe uploads to PyPI"
    )


def copy_workflows_with_pr_test_job_change(tmp_path: Path, change) -> None:
    shutil.copytree(WORKFLOWS, tmp_path, dirs_exist_ok=True)
    pr_workflow = tmp_path / "pr.yaml"
    workflow = yaml.safe_load(pr_workflow.read_text())
    change(workflow["jobs"])
    pr_workflow.write_text(yaml.safe_dump(workflow))


@pytest.mark.parametrize(
    "upload",
    [
        {"uses": "pypa/gh-action-pypi-publish@release/v1"},
        {"run": "python -m twine upload dist/*"},
        {"run": "make build\nuv publish"},
        {"run": "poetry publish --build"},
        {"run": "maturin publish"},
    ],
    ids=[
        "pypi-publish-action",
        "twine-upload",
        "uv-publish",
        "poetry-publish",
        "maturin-publish",
    ],
)
def test_pypi_upload_check_finds_upload_in_another_workflow(tmp_path, upload):
    copy_workflows_with_pr_test_job_change(
        tmp_path, lambda jobs: jobs["Test"]["steps"].append(upload)
    )

    assert pypi_publishing_jobs(tmp_path) == [
        ("pr.yaml", "Test"),
        ("push.yaml", "Publish"),
    ]


def test_reusable_workflow_check_finds_only_remote_workflows(tmp_path):
    def call_reusable_workflows(jobs: dict) -> None:
        jobs["Remote"] = {"uses": "some-org/some-repo/.github/workflows/publish.yml@v1"}
        jobs["Local"] = {"uses": "./.github/workflows/push.yaml"}

    copy_workflows_with_pr_test_job_change(tmp_path, call_reusable_workflows)

    assert remote_reusable_workflow_jobs(tmp_path) == [("pr.yaml", "Remote")]


@pytest.mark.parametrize("file_name", ["action.yml", "action.yaml"])
def test_pypi_upload_check_finds_upload_in_composite_action(tmp_path, file_name):
    action = tmp_path / "publish" / file_name
    action.parent.mkdir()
    action.write_text(
        yaml.safe_dump(
            {
                "name": "Publish",
                "runs": {
                    "using": "composite",
                    "steps": [{"run": "uv publish", "shell": "bash"}],
                },
            }
        )
    )

    assert pypi_publishing_actions(tmp_path) == [f"publish/{file_name}"]
    assert pypi_publishing_actions(tmp_path / "missing") == []


def test_pypi_upload_check_finds_upload_in_makefile_recipe(tmp_path):
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        MAKEFILE.read_text().replace(
            "build:\n\tpython -m build\n",
            "build:\n\tpython -m build\n\ttwine upload dist/*\n",
        )
    )

    assert pypi_uploading_recipe_lines(makefile) == ["build: twine upload dist/*"]


def assert_test_job_fails_when_tests_fail(workflow: dict) -> None:
    test_job = workflow["jobs"]["Test"]
    steps = test_job["steps"]
    # continue-on-error on Test or one of its steps lets the job succeed
    # after a step fails, and Publish would then run.
    assert not any("continue-on-error" in item for item in [test_job, *steps]), (
        "Test continues on error"
    )
    # Match the command's words, so that a separate
    # `make test-country-template` step is not taken for the test step.
    test_step = steps[
        only_step(
            steps,
            lambda step: step.get("run", "").split()[:2] == ["make", "test"],
            "test",
        )
    ]
    # A skipped step does not fail the job, so a condition on the step that
    # runs the tests could let Test succeed without running them.
    assert "if" not in test_step, "test step has a condition"
    assert test_step["run"].strip() == "make test", (
        "test step runs more than make test: anything after it (|| true, | tee, "
        "|| exit 0) can hide a failure; if the command changed on purpose, update "
        "this check to the new exact command instead of removing it"
    )
    # A shell template inserts the script's file name at {0}, so
    # `bash -c true {0}` runs `true` and ignores the script. A default shell
    # can come from the job's or the workflow's defaults.run as well as the
    # step.
    assert not any(
        "shell" in item
        for item in [
            test_step,
            (test_job.get("defaults") or {}).get("run") or {},
            (workflow.get("defaults") or {}).get("run") or {},
        ]
    ), "test step sets a shell"
    # pytest adds the options in PYTEST_ADDOPTS to its command line, so a
    # value such as --collect-only lets make test pass without running tests.
    assert not any(
        "PYTEST_ADDOPTS" in (item.get("env") or {})
        for item in [workflow, test_job, test_step]
    ), "Test sets PYTEST_ADDOPTS"


def assert_make_test_fails_when_tests_fail(
    rules: dict[str, tuple[list[str], list[str]]],
) -> None:
    prerequisites, _ = rules["test"]
    # make ignores a failing recipe line whose prefix (any mix of @, + and -
    # after the tab) includes -, so make test could pass after tests fail.
    for target in ["test", *prerequisites]:
        for line in rules.get(target, ([], []))[1]:
            prefix = line[: len(line) - len(line.lstrip("@+- \t"))]
            assert "-" not in prefix, f"make {target} ignores a failing recipe line"


def test_test_job_cannot_pass_with_failing_tests():
    workflow = yaml.safe_load(PUSH_WORKFLOW.read_text())

    assert_test_job_fails_when_tests_fail(workflow)
    assert_make_test_fails_when_tests_fail(make_rules(MAKEFILE))


def find_test_step(workflow: dict) -> dict:
    return next(
        step
        for step in workflow["jobs"]["Test"]["steps"]
        if step.get("run") == "make test"
    )


def continue_test_job_on_error(workflow: dict) -> None:
    workflow["jobs"]["Test"]["continue-on-error"] = True


def continue_after_test_failure(workflow: dict) -> None:
    find_test_step(workflow)["continue-on-error"] = True


def ignore_test_failure(workflow: dict) -> None:
    find_test_step(workflow)["run"] += " || true"


def skip_tests(workflow: dict) -> None:
    find_test_step(workflow)["if"] = False


def run_test_step_in_ignoring_shell(workflow: dict) -> None:
    find_test_step(workflow)["shell"] = "bash -c true {0}"


def set_test_job_default_shell(workflow: dict) -> None:
    workflow["jobs"]["Test"]["defaults"] = {"run": {"shell": "bash -c true {0}"}}


def set_workflow_default_shell(workflow: dict) -> None:
    workflow["defaults"] = {"run": {"shell": "bash -c true {0}"}}


def collect_tests_only_in_test_step(workflow: dict) -> None:
    find_test_step(workflow)["env"] = {"PYTEST_ADDOPTS": "--collect-only"}


def collect_tests_only_in_test_job(workflow: dict) -> None:
    workflow["jobs"]["Test"]["env"] = {"PYTEST_ADDOPTS": "--collect-only"}


def collect_tests_only_in_workflow(workflow: dict) -> None:
    workflow["env"] = {"PYTEST_ADDOPTS": "--collect-only"}


@pytest.mark.parametrize(
    ("mutate", "violation"),
    [
        pytest.param(mutate, violation, id=mutate.__name__)
        for mutate, violation in [
            (continue_test_job_on_error, "Test continues on error"),
            (continue_after_test_failure, "Test continues on error"),
            (ignore_test_failure, "test step runs more than make test"),
            (skip_tests, "test step has a condition"),
            (run_test_step_in_ignoring_shell, "test step sets a shell"),
            (set_test_job_default_shell, "test step sets a shell"),
            (set_workflow_default_shell, "test step sets a shell"),
            (collect_tests_only_in_test_step, "Test sets PYTEST_ADDOPTS"),
            (collect_tests_only_in_test_job, "Test sets PYTEST_ADDOPTS"),
            (collect_tests_only_in_workflow, "Test sets PYTEST_ADDOPTS"),
        ]
    ],
)
def test_test_job_check_rejects_masked_test_failure(mutate, violation):
    workflow = yaml.safe_load(PUSH_WORKFLOW.read_text())
    mutate(workflow)

    with pytest.raises(AssertionError, match=violation):
        assert_test_job_fails_when_tests_fail(workflow)


def test_test_job_check_accepts_separate_template_test_step():
    workflow = yaml.safe_load(PUSH_WORKFLOW.read_text())
    steps = workflow["jobs"]["Test"]["steps"]
    steps.insert(
        steps.index(find_test_step(workflow)),
        {"run": "make test-country-template"},
    )

    assert_test_job_fails_when_tests_fail(workflow)


@pytest.mark.parametrize(
    ("target", "prefix"),
    [("test", "-"), ("test", "@-"), ("test", "+-"), ("test-country-template", "-")],
)
def test_make_test_check_rejects_ignored_recipe_failure(tmp_path, target, prefix):
    lines = MAKEFILE.read_text().splitlines()
    recipe_start = next(
        index for index, line in enumerate(lines) if line.startswith(f"{target}:")
    )
    lines[recipe_start + 1] = "\t" + prefix + lines[recipe_start + 1][1:]
    makefile = tmp_path / "Makefile"
    makefile.write_text("\n".join(lines) + "\n")

    with pytest.raises(
        AssertionError, match=f"make {target} ignores a failing recipe line"
    ):
        assert_make_test_fails_when_tests_fail(make_rules(makefile))
