"""
This module directs PyTest to include certain global fixtures when running tests.
"""

import gc
import sys

import pytest

pytest_plugins = [
    "tests.fixtures.entities",
    "tests.fixtures.simulations",
    "tests.fixtures.taxbenefitsystems",
]

# TEMPORARY DIAGNOSTIC (to be reverted): which test first leaves a traced
# parameter tree behind on Windows.
_DIAG_TRACED = set()


@pytest.fixture(autouse=True)
def _diag_parameter_trace(request):
    yield
    if sys.platform != "win32":
        return
    from policyengine_core.parameters import ParameterNode

    for obj in gc.get_objects():
        if (
            type(obj) is ParameterNode
            and obj.name == ""
            and obj.trace
            and id(obj) not in _DIAG_TRACED
        ):
            _DIAG_TRACED.add(id(obj))
            sys.__stderr__.write(
                f"\nDIAG traced root {id(obj):x} after {request.node.nodeid}\n"
            )
