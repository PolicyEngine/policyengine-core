"""Core must not call the PolicyEngine Public API.

The Public API depends on the country packages, which depend on Core. Core
code that sends requests to api.policyengine.org reverses that dependency at
run time, so the package source must not name the host at all.
"""

import importlib.util
from pathlib import Path

from policyengine_core.reforms import Reform

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "policyengine_core"
PUBLIC_API_HOST = "api.policyengine.org"


def test_package_source_does_not_name_the_public_api_host():
    files_naming_host = sorted(
        str(path.relative_to(PACKAGE_ROOT))
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and PUBLIC_API_HOST.encode() in path.read_bytes()
    )
    assert not files_naming_host, (
        f"{files_naming_host} name {PUBLIC_API_HOST}; Core must not call the "
        "Public API."
    )


def test_reform_has_no_public_api_helpers():
    assert not hasattr(Reform, "from_api")
    assert "api_id" not in vars(Reform)


def test_public_api_chart_module_is_removed():
    assert importlib.util.find_spec("policyengine_core.charts.api") is None
