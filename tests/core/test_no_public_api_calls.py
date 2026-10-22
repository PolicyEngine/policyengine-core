"""Core must not call the PolicyEngine Public API.

The Public API depends on the country packages, which depend on Core. Core
code that sends requests to api.policyengine.org reverses that dependency at
run time.
"""

import importlib.util

import policyengine_core.charts as charts

REMOVED_CHART_FUNCTION_NAMES = [
    "get_api_chart_data",
    "decile_chart",
    "intra_decile_chart",
]


def test_public_api_chart_module_is_removed():
    assert importlib.util.find_spec("policyengine_core.charts.api") is None


def test_charts_package_does_not_export_public_api_chart_functions():
    exported_names = [
        name for name in REMOVED_CHART_FUNCTION_NAMES if hasattr(charts, name)
    ]
    assert exported_names == []
