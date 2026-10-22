"""Core must not call the PolicyEngine Public API.

The Public API depends on the country packages, which depend on Core. Core
code that sends requests to api.policyengine.org reverses that dependency at
run time, so the package source must not name the host at all.
"""

import importlib.util
import inspect
from pathlib import Path

import policyengine_core.charts as charts
from policyengine_core.reforms import Reform

PACKAGE_SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "policyengine_core"
PUBLIC_API_HOST = "api.policyengine.org"

REMOVED_CHART_FUNCTION_NAMES = [
    "get_api_chart_data",
    "decile_chart",
    "intra_decile_chart",
]

REMOVED_REFORM_ATTRIBUTE_NAMES = ["from_api", "api_id"]

ATTRIBUTE_NOT_FOUND = object()


def test_package_source_does_not_name_the_public_api_host():
    # Fail instead of passing with nothing scanned if the package moves.
    assert (PACKAGE_SOURCE_DIRECTORY / "reforms" / "reform.py").is_file()
    files_naming_host = sorted(
        str(path.relative_to(PACKAGE_SOURCE_DIRECTORY))
        for path in PACKAGE_SOURCE_DIRECTORY.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and PUBLIC_API_HOST.encode() in path.read_bytes()
    )
    assert files_naming_host == []


def test_reform_has_no_public_api_functions():
    # getattr_static looks the names up without running them: reading
    # Reform.api_id ran a property that posted the reform to the Public API.
    remaining_names = [
        name
        for name in REMOVED_REFORM_ATTRIBUTE_NAMES
        if inspect.getattr_static(Reform, name, ATTRIBUTE_NOT_FOUND)
        is not ATTRIBUTE_NOT_FOUND
    ]
    assert remaining_names == []


def test_public_api_chart_module_is_removed():
    assert importlib.util.find_spec("policyengine_core.charts.api") is None


def test_charts_package_does_not_export_public_api_chart_functions():
    exported_names = [
        name for name in REMOVED_CHART_FUNCTION_NAMES if hasattr(charts, name)
    ]
    assert exported_names == []
