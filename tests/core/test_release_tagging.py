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
