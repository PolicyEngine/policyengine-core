from __future__ import annotations

import importlib.metadata
import json

import pytest

from policyengine_core import get_runtime_metadata


def test_runtime_metadata_has_core_identity():
    metadata = get_runtime_metadata()

    assert metadata["name"] == "policyengine-core"
    assert metadata["version"] == importlib.metadata.version("policyengine-core")


def test_runtime_metadata_is_json_compatible():
    json.dumps(get_runtime_metadata())


def test_runtime_metadata_uses_bundle_contract_when_available():
    validation = pytest.importorskip("policyengine_bundles")

    validation.load_component_metadata(get_runtime_metadata())
