"""A rerun must not carry a module-scoped fixture into later modules.

``make test`` reruns failed tests (``--reruns 2``). pytest-rerunfailures 14.0
empties pytest's setup stack before a rerun without running the finalizers on
it, and a cached fixture registers its finalizer only when it is first
created. So the module-scoped ``tax_benefit_system`` of the rerun test's
module was never torn down, and pytest handed the same cached object to every
later module, with whatever earlier modules had done to it. On Windows CI a
later module then traced it, and ``tests/core/test_parameters.py`` and
``tests/core/test_reforms.py`` failed on a system that was not theirs.

pytest-rerunfailures 15.0 puts the finalizers back after a rerun
(pytest-dev/pytest-rerunfailures#278), and the dev dependency now requires a
release that does. This test runs a small suite with the installed plugin in a
subprocess: a test in the first module fails and is rerun, and the module
after it must get a fresh fixture, created after the first module's was torn
down.
"""

import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ElementTree

import pytest

pytest.importorskip("pytest_rerunfailures")

CONFTEST = """
import pytest

CREATED = []


class System:
    def __init__(self, module):
        self.module = module
        self.torn_down = False


@pytest.fixture(scope="module")
def system(request):
    system = System(request.module.__name__)
    CREATED.append(system)
    yield system
    system.torn_down = True
"""

PASSES_ON_RERUN = """
ATTEMPTS = []


def test_rerun(system):
    ATTEMPTS.append(system)
    assert len(ATTEMPTS) > 1, "the first attempt fails"
"""

FAILS_EVERY_ATTEMPT = """
def test_rerun(system):
    assert False, "every attempt fails"
"""

LATER_MODULE = """
from conftest import CREATED


def test_gets_a_fresh_fixture(system):
    assert system.module == __name__
    earlier = [created for created in CREATED if created is not system]
    assert earlier
    assert all(created.torn_down for created in earlier)
    assert all(created.module != __name__ for created in earlier)
"""


def _outcomes(junit_xml):
    outcomes = {}
    for case in ElementTree.parse(junit_xml).iter("testcase"):
        name = f"{case.get('classname')}.{case.get('name')}"
        failed = case.find("failure") is not None or case.find("error") is not None
        outcomes[name] = "failed" if failed else "passed"
    return outcomes


@pytest.mark.parametrize(
    "rerun_module, expected_rerun_outcome",
    [(PASSES_ON_RERUN, "passed"), (FAILS_EVERY_ATTEMPT, "failed")],
    ids=["passes-on-rerun", "fails-every-attempt"],
)
def test_rerun_does_not_leak_module_fixture_into_later_modules(
    tmp_path, rerun_module, expected_rerun_outcome
):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(textwrap.dedent(CONFTEST))
    (tmp_path / "test_a_rerun.py").write_text(textwrap.dedent(rerun_module))
    (tmp_path / "test_b_later.py").write_text(textwrap.dedent(LATER_MODULE))
    junit_xml = tmp_path / "junit.xml"

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COV_", "COVERAGE_"))
    }
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_rerunfailures",
            "-p",
            "no:cacheprovider",
            "--reruns",
            "1",
            "--reruns-delay",
            "0",
            f"--junitxml={junit_xml}",
            "-q",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = completed.stdout + completed.stderr

    assert _outcomes(junit_xml) == {
        "test_a_rerun.test_rerun": expected_rerun_outcome,
        "test_b_later.test_gets_a_fresh_fixture": "passed",
    }, output
    assert "1 rerun" in output, output
