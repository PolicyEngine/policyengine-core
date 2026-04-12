import pytest

from policyengine_core.errors import ParameterParsingError
from policyengine_core.parameters import ParameterNode, homogenize_parameter_structures
from policyengine_core.parameters.helpers import _load_yaml_file


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
        lambda expression, globals=None, locals=None: eval_calls.append(expression)
        or range(1, 4),
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
