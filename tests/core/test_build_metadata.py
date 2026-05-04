from __future__ import annotations

import importlib.metadata
import json

import pytest

from policyengine_core import get_runtime_metadata
from policyengine_core import build_metadata


def test_runtime_metadata_has_core_identity():
    metadata = get_runtime_metadata()

    assert metadata["name"] == "policyengine-core"
    assert metadata["version"] == importlib.metadata.version("policyengine-core")


def test_runtime_metadata_is_json_compatible():
    json.dumps(get_runtime_metadata())


def test_runtime_metadata_does_not_include_local_source_path():
    assert "source_path" not in get_runtime_metadata()


def test_runtime_metadata_uses_pep_610_vcs_commit(monkeypatch):
    class Distribution:
        version = "1.2.3"

        def read_text(self, name):
            if name == "direct_url.json":
                return json.dumps(
                    {
                        "vcs_info": {
                            "vcs": "git",
                            "commit_id": "abc123",
                        }
                    }
                )

            return None

    monkeypatch.setattr(
        build_metadata.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    assert get_runtime_metadata() == {
        "name": "policyengine-core",
        "version": "1.2.3",
        "git_sha": "abc123",
    }


def test_runtime_metadata_uses_bundle_contract_when_available():
    validation = pytest.importorskip("policyengine_bundles")

    validation.load_component_metadata(get_runtime_metadata())
