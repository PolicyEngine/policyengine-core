from __future__ import annotations

import importlib.metadata
import json
from typing import Any, Dict, Optional


PACKAGE_NAME = "policyengine-core"

__all__ = ["get_runtime_metadata"]


def get_runtime_metadata() -> Dict[str, Any]:
    """Return JSON-compatible metadata describing this core runtime.

    The payload intentionally avoids importing the bundle orchestrator. It is
    shaped to be validated by policyengine-bundles when that package is
    available in release or integration-test workflows.
    """

    distribution = importlib.metadata.distribution(PACKAGE_NAME)
    metadata: Dict[str, Any] = {
        "name": PACKAGE_NAME,
        "version": distribution.version,
    }

    git_sha = _get_direct_url_git_sha(distribution)
    if git_sha is not None:
        metadata["git_sha"] = git_sha

    return metadata


def _get_direct_url_git_sha(
    distribution: importlib.metadata.Distribution,
) -> Optional[str]:
    direct_url = _read_direct_url(distribution)
    if direct_url is None:
        return None

    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None

    commit_id = vcs_info.get("commit_id")
    if isinstance(commit_id, str) and commit_id:
        return commit_id

    return None


def _read_direct_url(
    distribution: importlib.metadata.Distribution,
) -> Optional[Dict[str, Any]]:
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        return None

    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None

    if isinstance(direct_url, dict):
        return direct_url

    return None
