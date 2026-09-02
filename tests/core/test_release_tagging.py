import shutil
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_publish_workflow_tags_only_after_pypi_succeeds():
    workflow = PUSH_WORKFLOW.read_text()
    publish_job = workflow.split("  Publish:\n", maxsplit=1)[1]
    publish_step = "uses: pypa/gh-action-pypi-publish@release/v1"
    tag_step = "run: bash .github/publish-git-tag.sh"

    assert "permissions:\n      contents: write" in publish_job
    assert "fetch-depth: 0" in publish_job
    assert publish_job.index(tag_step) > publish_job.index(publish_step)
    assert "publish-git-tag.sh || true" not in publish_job
