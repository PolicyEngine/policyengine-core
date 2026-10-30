"""The YAML runner pauses automatic collection while it builds a reform system.

A cache miss in ``_get_tax_benefit_system`` builds with automatic collection
paused, then, if the caller had collection on and the build returned, runs
one full collection. It leaves ``gc.isenabled()`` as it found it, whether the
build returns or raises and whether the caller had collection on or off. A
cache hit does not touch collection. The pause changes no result: the cache
still returns the identical system for a key it has seen and builds a new
one for any other key, and YAML cases produce the same outcomes and values
with or without it.
"""

import contextlib
import gc
import os

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.tools import test_runner
from policyengine_core.tools.test_runner import (
    _automatic_gc_paused,
    _get_tax_benefit_system,
)
from tests.fixtures import yaml_tests

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies


class BuildFailed(Exception):
    pass


class RecordingSystem:
    """A stand-in system that records whether collection was enabled at each
    build step, and can be told to raise at one of them."""

    def __init__(self, log, fail_at=None, reforms=(), extensions=()):
        self.log = log
        self.fail_at = fail_at
        self.reforms = reforms
        self.extensions = extensions

    def _step(self, name):
        self.log.append((name, gc.isenabled()))
        if self.fail_at == name:
            raise BuildFailed(name)

    def clone(self):
        self._step("clone")
        return RecordingSystem(self.log, self.fail_at, self.reforms, self.extensions)

    def apply_reform(self, reform_path):
        self._step("reform")
        return RecordingSystem(
            self.log, self.fail_at, self.reforms + (reform_path,), self.extensions
        )

    def load_extension(self, extension):
        self._step("extension")
        self.extensions = self.extensions + (extension,)


def callable_reform(system):
    system._step("reform")
    return RecordingSystem(
        system.log, system.fail_at, system.reforms + ("callable",), system.extensions
    )


@pytest.fixture
def restore_gc():
    was_enabled = gc.isenabled()
    yield
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


@pytest.fixture
def gc_calls(restore_gc, monkeypatch):
    """Every call to gc.disable, gc.enable and gc.collect, in order. Disabling
    and enabling take effect; a recorded collection does not run, to keep
    hundreds of builds fast."""
    calls = []
    for name in ("disable", "enable", "collect"):
        real = getattr(gc, name) if name != "collect" else lambda *args: 0

        def spy(*args, real=real, name=name):
            calls.append(name)
            return real(*args)

        monkeypatch.setattr(gc, name, spy)
    return calls


def set_gc(enabled):
    if enabled:
        gc.enable()
    else:
        gc.disable()


def test_a_cache_miss_builds_paused_then_collects_once(gc_calls):
    set_gc(True)
    log = []
    baseline = RecordingSystem(log)
    del gc_calls[:]

    system = _get_tax_benefit_system(
        baseline, ["some.reform", callable_reform], ["some.extension"]
    )

    assert [name for name, _ in log] == [
        "clone",
        "reform",
        "reform",
        "clone",
        "extension",
    ]
    assert [enabled for _, enabled in log] == [False] * 5
    assert gc_calls == ["disable", "collect", "enable"]
    assert gc.isenabled()
    assert system.reforms == ("some.reform", "callable")
    assert system.extensions == ("some.extension",)


SCENARIOS = ["miss", "hit", "clone_raises", "reform_raises", "extension_raises"]


@pytest.mark.parametrize("enabled_before", [True, False])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_collection_state_after_a_call_equals_the_state_before(
    gc_calls, enabled_before, scenario
):
    fail_at = {
        "clone_raises": "clone",
        "reform_raises": "reform",
        "extension_raises": "extension",
    }.get(scenario)
    baseline = RecordingSystem([], fail_at=fail_at)
    if scenario == "hit":
        _get_tax_benefit_system(baseline, ["some.reform"], ["some.extension"])

    set_gc(enabled_before)
    del gc_calls[:]
    if fail_at is None:
        _get_tax_benefit_system(baseline, ["some.reform"], ["some.extension"])
    else:
        with pytest.raises(BuildFailed, match=fail_at):
            _get_tax_benefit_system(baseline, ["some.reform"], ["some.extension"])
    assert gc.isenabled() is enabled_before
    # Only a build that returns, for a caller with collection on, collects.
    assert gc_calls.count("collect") == int(scenario == "miss" and enabled_before)

    # A failed build caches nothing, so the next call builds again.
    log_length = len(baseline.log)
    baseline.fail_at = None
    _get_tax_benefit_system(baseline, ["some.reform"], ["some.extension"])
    assert (len(baseline.log) > log_length) is (fail_at is not None)
    assert gc.isenabled() is enabled_before


def test_a_cache_hit_leaves_collection_alone(gc_calls):
    set_gc(True)
    baseline = RecordingSystem([])
    system = _get_tax_benefit_system(baseline, ["some.reform"], [])
    del gc_calls[:]

    assert _get_tax_benefit_system(baseline, ["some.reform"], []) is system
    assert gc_calls == []
    assert gc.isenabled()


def test_the_pause_restores_the_callers_state_when_the_body_raises(gc_calls):
    for enabled_before in (True, False):
        set_gc(enabled_before)
        del gc_calls[:]
        with pytest.raises(BuildFailed):
            with _automatic_gc_paused():
                assert not gc.isenabled()
                raise BuildFailed("body")
        assert gc.isenabled() is enabled_before
        assert "collect" not in gc_calls


REFORM_NAMES = ["a.reform", "b.reform", "c.reform"]
EXTENSION_NAMES = ["x.extension", "y.extension"]

request_strategy = st.tuples(
    st.lists(st.sampled_from(REFORM_NAMES), max_size=3),
    st.sampled_from([None, "", "gov.rate:1", "gov.rate:2"]),
    st.lists(st.sampled_from(EXTENSION_NAMES), max_size=2),
    st.booleans(),
)


@hypothesis.settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
)
@hypothesis.given(requests=st.lists(request_strategy, max_size=25))
def test_the_cache_matches_a_reference_dict_and_keeps_the_callers_state(
    gc_calls, requests
):
    log = []
    baseline = RecordingSystem(log)
    reference = {}
    for reforms, reform_key, extensions, enabled_before in requests:
        key = (tuple(reforms), reform_key, frozenset(extensions))
        clones_before = sum(name == "clone" for name, _ in log)
        set_gc(enabled_before)
        del gc_calls[:]

        system = _get_tax_benefit_system(
            baseline, list(reforms), list(extensions), reform_key=reform_key
        )

        assert gc.isenabled() is enabled_before
        assert all(not enabled for _, enabled in log)
        built = sum(name == "clone" for name, _ in log) > clones_before
        assert gc_calls.count("collect") == int(built and enabled_before)
        if key in reference:
            assert system is reference[key]
            assert not built
            assert gc_calls == []
        else:
            assert built
            assert all(system is not other for other in reference.values())
            assert system.reforms == tuple(reforms)
            assert sorted(system.extensions) == sorted(extensions)
            reference[key] = system


yaml_tests_dir = os.path.dirname(yaml_tests.__file__)

INLINE_PARAMETER_CASES = """
- name: "A dotted parameter input builds a reform system"
  period: 2017-01
  input:
    taxes.income_tax_rate: 0.2
    salary: 2000
  output:
    income_tax: 400

- name: "A second parameter value builds another"
  period: 2017-01
  input:
    taxes.income_tax_rate: 0.25
    salary: 2000
  output:
    income_tax: 500

- name: "A failing case fails either way"
  period: 2017-01
  input:
    taxes.income_tax_rate: 0.2
    salary: 2000
  output:
    income_tax: 1
"""


def run_recording(paths, pause):
    """Run the YAML runner on a fresh baseline, with the real pause or none.

    Returns the exit code, each case's outcome, each computed value, and
    whether collection was enabled at each system clone.
    """
    outcomes = []
    values = []
    gc_at_clones = []

    class Outcomes:
        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                outcomes.append((report.nodeid, report.outcome))

    assert_near = test_runner.assert_near
    clone = TaxBenefitSystem.clone

    def recording_assert_near(value, target_value, *args, **kwargs):
        values.append((kwargs.get("message"), repr(value)))
        return assert_near(value, target_value, *args, **kwargs)

    def recording_clone(self):
        gc_at_clones.append(gc.isenabled())
        return clone(self)

    baseline = CountryTaxBenefitSystem()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(test_runner, "assert_near", recording_assert_near)
        patch.setattr(TaxBenefitSystem, "clone", recording_clone)
        if not pause:
            patch.setattr(test_runner, "_automatic_gc_paused", contextlib.nullcontext)
        # run_tests' own arguments, plus a plugin that records outcomes.
        exit_code = pytest.main(
            ["--capture", "no", "--maxfail", "0", "--tb", "no", "-q", *paths],
            plugins=[test_runner.OpenFiscaPlugin(baseline, {}), Outcomes()],
        )
    return exit_code, outcomes, values, gc_at_clones


def test_yaml_cases_have_the_same_outcomes_with_and_without_the_pause(
    restore_gc, tmp_path
):
    gc.enable()
    inline = tmp_path / "inline_parameters.yaml"
    inline.write_text(INLINE_PARAMETER_CASES, encoding="utf-8")
    paths = [
        os.path.join(yaml_tests_dir, "test_with_reform.yaml"),
        os.path.join(yaml_tests_dir, "test_with_reform_2.yaml"),
        os.path.join(yaml_tests_dir, "test_with_extension.yaml"),
        str(inline),
    ]

    paused = run_recording(paths, pause=True)
    unpaused = run_recording(paths, pause=False)

    exit_code, outcomes, values, gc_at_clones = paused
    assert paused[:3] == unpaused[:3]
    assert exit_code == pytest.ExitCode.TESTS_FAILED
    assert [outcome for _, outcome in outcomes].count("failed") == 2
    assert len(outcomes) == 8
    assert len(values) >= 8
    # Six builds (two reform lists, the empty combination, one extension and
    # two parameter values) cloned the system with collection paused, and
    # with the pause removed every clone ran with collection on.
    assert len(gc_at_clones) >= 6
    assert not any(gc_at_clones)
    assert len(unpaused[3]) == len(gc_at_clones)
    assert all(unpaused[3])
    assert gc.isenabled()
