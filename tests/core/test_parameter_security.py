import pytest

from policyengine_core.errors import ParameterParsingError
from policyengine_core.parameters import ParameterNode, homogenize_parameter_structures
from policyengine_core.parameters.helpers import _load_yaml_file
from policyengine_core.parameters.operations.homogenize_parameters import (
    MAX_DYNAMIC_BREAKDOWN_VALUES,
    evaluate_dynamic_breakdown,
)


def test_parameter_yaml_loader_rejects_python_object_tags(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(
        "os.system",
        lambda command: calls.append(command) or 0,
    )

    yaml_path = tmp_path / "malicious.yaml"
    yaml_path.write_text(
        '!!python/object/apply:os.system ["echo pwned"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ParameterParsingError):
        _load_yaml_file(str(yaml_path))

    assert calls == []


def test_homogenize_parameter_structures_rejects_dynamic_breakdown_code(
    monkeypatch,
):
    eval_calls = []

    monkeypatch.setattr(
        "builtins.eval",
        lambda expression, globals=None, locals=None: (
            eval_calls.append(expression) or range(1, 4)
        ),
    )

    root = ParameterNode(
        data={
            "value_by_category": {
                "metadata": {
                    "breakdown": ['__import__("os").system("echo pwned")'],
                },
            }
        }
    )

    with pytest.raises(ValueError, match="breakdown"):
        homogenize_parameter_structures(root, {}, default_value=0)

    assert eval_calls == []


def test_homogenize_parameter_structures_rejects_oversized_dynamic_breakdown():
    root = ParameterNode(
        data={
            "value_by_category": {
                "metadata": {
                    "breakdown": [f"list(range({MAX_DYNAMIC_BREAKDOWN_VALUES + 1}))"],
                },
            }
        }
    )

    with pytest.raises(ValueError, match="exceeds the maximum"):
        homogenize_parameter_structures(root, {}, default_value=0)


def test_homogenize_parameter_structures_rejects_overflowing_dynamic_breakdown():
    huge_stop = "1" + ("0" * 100)
    root = ParameterNode(
        data={
            "value_by_category": {
                "metadata": {
                    "breakdown": [f"range(0, {huge_stop})"],
                },
            }
        }
    )

    with pytest.raises(ValueError, match="too many values"):
        homogenize_parameter_structures(root, {}, default_value=0)


def test_parameter_yaml_loader_rejects_implicit_duplicate_keys(tmp_path):
    yaml_path = tmp_path / "duplicate-bools.yaml"
    yaml_path.write_text("true: 1\nTrue: 2\n", encoding="utf-8")

    with pytest.raises(ParameterParsingError, match="duplicate key"):
        _load_yaml_file(str(yaml_path))


def test_parameter_yaml_loader_allows_merge_key_overrides(tmp_path):
    yaml_path = tmp_path / "merge-override.yaml"
    yaml_path.write_text(
        "defaults: &defaults\n  value: 1\nmerged:\n  <<: *defaults\n  value: 2\n",
        encoding="utf-8",
    )

    result = _load_yaml_file(str(yaml_path))

    assert result["merged"]["value"] == 2


def test_evaluate_dynamic_breakdown_allows_documented_safe_forms():
    assert evaluate_dynamic_breakdown("list(range(1, 4))") == [1, 2, 3]
    assert evaluate_dynamic_breakdown("[1, 2, -3]") == [1, 2, -3]
    assert evaluate_dynamic_breakdown('("a", "b")') == ["a", "b"]


def test_homogenize_parameter_structures_ignores_empty_breakdown_lists():
    root = ParameterNode(
        data={
            "value_by_category": {
                "metadata": {
                    "breakdown": [],
                },
            }
        }
    )

    result = homogenize_parameter_structures(root, {}, default_value=0)

    assert result is root
