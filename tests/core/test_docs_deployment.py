import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PUSH_WORKFLOW = WORKFLOWS / "push.yaml"
DOCS_CONFIG = ROOT / "docs" / "_config.yml"
MAKEFILE = ROOT / "Makefile"
DEPLOY_ACTION = "JamesIves/github-pages-deploy-action@"
PYPI_ACTION = "pypa/gh-action-pypi-publish@"


def load_jobs(workflow: Path) -> dict:
    return yaml.safe_load(workflow.read_text())["jobs"]


def as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def strings_in(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from strings_in(value)
    elif isinstance(node, list):
        for item in node:
            yield from strings_in(item)


def matrix_keys(job: dict) -> set | None:
    """Keys a job's matrix defines, or None when an expression builds it."""
    matrix = job.get("strategy", {}).get("matrix", {})
    if not isinstance(matrix, dict):
        return None
    keys = set(matrix) - {"include", "exclude"}
    include = matrix.get("include", [])
    if not isinstance(include, list):
        return None
    for entry in include:
        if not isinstance(entry, dict):
            return None
        keys |= set(entry)
    return keys


def jobs_using(jobs: dict, action: str) -> list:
    return [
        name
        for name, job in jobs.items()
        if any(step.get("uses", "").startswith(action) for step in job.get("steps", []))
    ]


def upstream_jobs(jobs: dict, name: str) -> set:
    seen = set()
    pending = as_list(jobs[name].get("needs"))
    while pending:
        needed = pending.pop()
        if needed not in seen:
            seen.add(needed)
            pending.extend(as_list(jobs[needed].get("needs")))
    return seen


@pytest.mark.parametrize(
    "workflow", sorted(WORKFLOWS.glob("*.y*ml")), ids=lambda path: path.name
)
def test_workflow_matrix_references_have_a_matrix(workflow: Path):
    # The Test job lost its matrix in September 2024 but kept a step guarded
    # by ``matrix.os == 'ubuntu-latest'``, so that condition never held and
    # the documentation deploy was skipped on every release afterwards.
    for name, job in load_jobs(workflow).items():
        defined = matrix_keys(job)
        if defined is None:
            continue
        referenced = {
            key
            for text in strings_in(job)
            for key in re.findall(r"\bmatrix\.([A-Za-z_][\w-]*)", text)
        }
        missing = referenced - defined
        assert not missing, (
            f"{workflow.name} job {name} references matrix keys "
            f"{sorted(missing)} that its strategy does not define"
        )


def test_release_deploys_documentation_after_publishing():
    jobs = load_jobs(PUSH_WORKFLOW)
    assert jobs_using(jobs, DEPLOY_ACTION) == ["Docs"]
    docs = jobs["Docs"]
    steps = docs["steps"]
    deploy = next(
        step for step in steps if step.get("uses", "").startswith(DEPLOY_ACTION)
    )
    build = next(step for step in steps if step.get("run") == "make documentation")

    assert docs["if"] == jobs["Test"]["if"]
    assert {"Test", "Publish"} <= set(as_list(docs.get("needs")))
    # A condition on the deploy step is how it went unnoticed for two years.
    assert "if" not in deploy
    assert steps.index(build) < steps.index(deploy)
    # GitHub Pages serves the gh-pages branch root. The action's ``token``
    # input defaults to the job token, which needs write access to push.
    assert deploy["with"]["branch"] == "gh-pages"
    assert deploy["with"]["folder"] == "docs/_build/html"
    assert docs["permissions"].get("contents") == "write"
    assert docs["concurrency"]["cancel-in-progress"] is False


def test_documentation_build_matches_deploy_folder():
    recipe = re.search(
        r"^documentation:.*\n((?:\t.*\n?)+)", MAKEFILE.read_text(), re.MULTILINE
    )
    extensions = yaml.safe_load(DOCS_CONFIG.read_text())["sphinx"]["extra_extensions"]

    # ``jb build docs`` writes HTML to docs/_build/html, the deployed folder.
    assert recipe is not None
    assert "jb build docs" in recipe.group(1)
    # Pages builds a branch source with Jekyll unless its root has .nojekyll,
    # which this extension writes into the HTML output.
    assert "sphinx.ext.githubpages" in extensions


def test_documentation_deploy_cannot_block_pypi_publishing():
    jobs = load_jobs(PUSH_WORKFLOW)

    for name in jobs_using(jobs, PYPI_ACTION):
        assert name != "Docs"
        assert "Docs" not in upstream_jobs(jobs, name)
