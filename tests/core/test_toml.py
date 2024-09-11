from pathlib import Path
import tomli
import pytest


@pytest.fixture(scope="module")
def toml_data():
    file_path = Path("/policyengine-core/pyproject.toml")
    if not file_path.exists():
        pytest.fail(f"pyproject.toml not found in the current directory.")
    with open(file_path, "rb") as f:
        return tomli.load(f)


def test_toml_syntax(toml_data):
    try:
        toml_data
    except tomli.TOMLDecodeError as e:
        pytest.fail(f"TOML syntax error: {e}")


def test_required_fields(toml_data):
    required_fields = ["name", "version", "dependencies"]
    for field in required_fields:
        assert field in toml_data.get(
            "project", {}
        ), f"Missing required field: {field}"


def test_build_system(toml_data):
    build_system = toml_data.get("build-system", {})
    assert "requires" in build_system, "Build system 'requires' is missing."
    assert (
        "build-backend" in build_system
    ), "Build system 'build-backend' is missing."
