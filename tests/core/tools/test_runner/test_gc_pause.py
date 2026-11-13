"""The YAML runner pauses automatic collection while it builds a reform system.

A cache miss in ``_get_tax_benefit_system``, for the reform-free system or a
reform system (including one the cache evicted, or any build with the reform
cache disabled), builds with automatic collection paused. If the caller had
collection on, the build returned, and the cache held fewer than
``_FULL_COLLECTION_CACHE_LIMIT`` systems (each baseline's reform-free system
plus its reform systems, across every baseline, after the call's bound trims
and before the new system goes in), it then runs one full collection;
otherwise it leaves the new system to the automatic collector. While the cache
only grows, a run of misses therefore forces at most that many collections.
Eviction can keep the cache below the limit for good (a ``reform_cache_size``
of 0 or 1 on one baseline holds at most 2 systems), and then every miss forces
one. It leaves ``gc.isenabled()`` as it found it, whether the build returns or
raises (any ``BaseException``, including ``KeyboardInterrupt`` and a
``gc.collect()`` that raises) and whether the caller had collection on or
off. A cache hit does not touch collection. The pause changes no result: the cache still returns the
identical system for a key it has seen and builds a new one for any other
key, and YAML cases produce the same outcomes and values with or without it.
"""

import contextlib
import gc
import os
import weakref
from collections import OrderedDict

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
    build step, and can be told to raise ``error`` at one of them."""

    def __init__(self, log, fail_at=None, reforms=(), extensions=(), error=BuildFailed):
        self.log = log
        self.fail_at = fail_at
        self.reforms = reforms
        self.extensions = extensions
        self.error = error

    def _step(self, name):
        self.log.append((name, gc.isenabled()))
        if self.fail_at == name:
            raise self.error(name)

    def clone(self):
        self._step("clone")
        return RecordingSystem(
            self.log, self.fail_at, self.reforms, self.extensions, self.error
        )

    def apply_reform(self, reform_path):
        self._step("reform")
        return RecordingSystem(
            self.log,
            self.fail_at,
            self.reforms + (reform_path,),
            self.extensions,
            self.error,
        )

    def load_extension(self, extension):
        self._step("extension")
        self.extensions = self.extensions + (extension,)


def callable_reform(system):
    system._step("reform")
    return RecordingSystem(
        system.log,
        system.fail_at,
        system.reforms + ("callable",),
        system.extensions,
        system.error,
    )


@pytest.fixture
def restore_gc():
    was_enabled = gc.isenabled()
    yield
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    """Give each test its own empty system cache, so systems other tests
    cached do not count toward the full-collection bound."""
    monkeypatch.setattr(
        test_runner, "_tax_benefit_system_cache", weakref.WeakKeyDictionary()
    )


# gc.collect()'s default generation, the oldest: a full collection.
FULL = "collect(2)"
LIMIT = test_runner._FULL_COLLECTION_CACHE_LIMIT


@pytest.fixture
def gc_calls(restore_gc, monkeypatch):
    """Every call to gc.disable, gc.enable and gc.collect, in order, with the
    generation each collection asked for: ``FULL`` for ``gc.collect()``.
    Disabling and enabling take effect; a recorded collection does not run,
    to keep hundreds of builds fast."""
    calls = []
    for name in ("disable", "enable"):

        def spy(real=getattr(gc, name), name=name):
            calls.append(name)
            return real()

        monkeypatch.setattr(gc, name, spy)

    def collect(generation=2):
        calls.append(f"collect({generation})")
        return 0

    monkeypatch.setattr(gc, "collect", collect)
    return calls


def collections(calls):
    return [call for call in calls if call.startswith("collect")]


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
    assert gc_calls == ["disable", FULL, "enable"]
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
    expected = [FULL] if scenario == "miss" and enabled_before else []
    assert collections(gc_calls) == expected

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
        assert collections(gc_calls) == []


def miss(baseline, number, cache_size=None):
    """A cache miss: a parameter key the cache has not seen."""
    return _get_tax_benefit_system(
        baseline, [], [], reform_key=f"gov.rate:{number}", cache_size=cache_size
    )


# A reform cache large enough that the tests below never evict.
NO_EVICTION = 100


@pytest.mark.parametrize("limit", [0, 1, 2, LIMIT, LIMIT + 2])
def test_misses_collect_only_while_the_cache_holds_fewer_systems_than_the_limit(
    gc_calls, monkeypatch, limit
):
    monkeypatch.setattr(test_runner, "_FULL_COLLECTION_CACHE_LIMIT", limit)
    set_gc(True)
    baseline = RecordingSystem([])
    collected = []
    for number in range(limit + 3):
        del gc_calls[:]
        miss(baseline, number, cache_size=NO_EVICTION)
        collected.append(collections(gc_calls))
        assert gc.isenabled()
    # The last forced collection is miss number limit - 1, with limit - 1
    # systems cached; from miss number limit on, none.
    assert collected == [[FULL]] * limit + [[]] * 3


def test_a_miss_past_the_limit_still_pauses_and_restores_collection(gc_calls):
    log = []
    baseline = RecordingSystem(log)
    for enabled_before in (True, False):
        set_gc(True)
        for number in range(LIMIT):
            miss(baseline, number, cache_size=NO_EVICTION)
        del log[:]
        set_gc(enabled_before)
        del gc_calls[:]

        system = _get_tax_benefit_system(
            baseline, ["some.reform"], ["x.extension"], cache_size=NO_EVICTION
        )

        assert [name for name, _ in log] == ["clone", "reform", "clone", "extension"]
        assert not any(enabled for _, enabled in log)
        assert gc_calls == (["disable", "enable"] if enabled_before else ["disable"])
        assert gc.isenabled() is enabled_before
        assert system.reforms == ("some.reform",)
        baseline = RecordingSystem(log)


def test_the_limit_counts_systems_cached_for_every_baseline(gc_calls, monkeypatch):
    monkeypatch.setattr(test_runner, "_FULL_COLLECTION_CACHE_LIMIT", 3)
    set_gc(True)
    first = RecordingSystem([])
    miss(first, 0)
    miss(first, 1)
    second = RecordingSystem([])
    del gc_calls[:]
    miss(second, 0)  # 2 cached
    assert collections(gc_calls) == [FULL]
    del gc_calls[:]
    miss(second, 1)  # 3 cached
    assert collections(gc_calls) == []
    # A baseline that goes away takes its systems out of the count.
    del first
    del gc_calls[:]
    miss(second, 2)  # 2 cached
    assert collections(gc_calls) == [FULL]


def reform_free(baseline, cache_size=None):
    return _get_tax_benefit_system(baseline, [], [], cache_size=cache_size)


def test_a_reform_free_miss_builds_paused_and_a_hit_leaves_collection_alone(
    gc_calls,
):
    set_gc(True)
    log = []
    baseline = RecordingSystem(log)
    del gc_calls[:]

    system = reform_free(baseline)

    assert log == [("clone", False)]
    assert gc_calls == ["disable", FULL, "enable"]
    assert gc.isenabled()
    del gc_calls[:]
    assert reform_free(baseline) is system
    assert reform_free(baseline, cache_size=0) is system
    assert len(log) == 1
    assert gc_calls == []


def test_a_count_of_two_collects_and_a_count_of_three_does_not(gc_calls):
    """With the default reform cache, the reform-free system and one reform
    system make 2 (a miss collects); with a second reform system, 3 (a miss
    does not)."""
    assert LIMIT == 3
    set_gc(True)
    baseline = RecordingSystem([])
    reform_free(baseline)
    miss(baseline, 0)
    assert test_runner._cached_system_count() == 2
    del gc_calls[:]
    miss(baseline, 1)
    assert gc_calls == ["disable", FULL, "enable"]
    assert test_runner._cached_system_count() == 3
    del gc_calls[:]
    miss(baseline, 2)  # Evicts miss 0 after the build.
    assert gc_calls == ["disable", "enable"]
    assert test_runner._cached_system_count() == 3


def test_a_miss_on_an_evicted_system_rebuilds_paused(gc_calls):
    set_gc(True)
    log = []
    baseline = RecordingSystem(log)
    first = miss(baseline, 0, cache_size=1)
    miss(baseline, 1, cache_size=1)  # Evicts miss 0.
    del log[:]
    del gc_calls[:]

    rebuilt = miss(baseline, 0, cache_size=1)

    assert rebuilt is not first
    assert log == [("clone", False)]
    # One reform system cached and no reform-free one: 1 < 3, so it collects.
    assert gc_calls == ["disable", FULL, "enable"]
    assert gc.isenabled()

    # With the reform-free system and two reform systems cached, the
    # rebuild of an evicted one pauses but leaves collection to the
    # automatic collector.
    reform_free(baseline)
    miss(baseline, 1, cache_size=2)
    miss(baseline, 2, cache_size=2)  # Evicts miss 0.
    assert test_runner._cached_system_count() == 3
    del log[:]
    del gc_calls[:]
    miss(baseline, 0, cache_size=2)
    assert log == [("clone", False)]
    assert gc_calls == ["disable", "enable"]
    assert gc.isenabled()


@pytest.mark.parametrize("enabled_before", [True, False])
def test_with_the_reform_cache_disabled_every_miss_builds_paused(
    gc_calls, enabled_before
):
    set_gc(True)
    log = []
    baseline = RecordingSystem(log)
    reform_free(baseline, cache_size=0)
    systems = []
    for _ in range(LIMIT + 2):
        del log[:]
        set_gc(enabled_before)
        del gc_calls[:]
        systems.append(miss(baseline, 0, cache_size=0))
        assert log == [("clone", False)]
        assert gc.isenabled() is enabled_before
        # The cache holds only the reform-free system, below the limit, so
        # every miss collects for a caller with collection on.
        expected = [FULL] if enabled_before else []
        assert collections(gc_calls) == expected
    assert len({id(system) for system in systems}) == len(systems)
    assert test_runner._cached_system_count() == 1


def test_the_count_trims_to_this_calls_bound_first(gc_calls):
    set_gc(True)
    baseline = RecordingSystem([])
    for number in range(LIMIT):
        miss(baseline, number, cache_size=NO_EVICTION)
    assert test_runner._cached_system_count() == LIMIT
    del gc_calls[:]
    # A bound of 1 leaves one reform system before this miss counts, so
    # the count is 1 and the miss collects.
    miss(baseline, LIMIT, cache_size=1)
    assert collections(gc_calls) == [FULL]
    assert test_runner._cached_system_count() == 1


class Interrupted(KeyboardInterrupt):
    pass


@pytest.mark.parametrize("path", ["reform_free", "reform"])
@pytest.mark.parametrize("enabled_before", [True, False])
@pytest.mark.parametrize("error", [BuildFailed, Interrupted])
def test_a_build_that_raises_restores_the_callers_state_on_both_paths(
    gc_calls, path, enabled_before, error
):
    baseline = RecordingSystem([], fail_at="clone", error=error)
    set_gc(enabled_before)
    del gc_calls[:]
    with pytest.raises(error):
        if path == "reform_free":
            reform_free(baseline)
        else:
            miss(baseline, 0)
    assert gc.isenabled() is enabled_before
    assert collections(gc_calls) == []
    assert test_runner._cached_system_count() == 0


@pytest.mark.parametrize("enabled_before", [True, False])
def test_the_pause_restores_the_callers_state_after_a_keyboard_interrupt(
    gc_calls, enabled_before
):
    set_gc(enabled_before)
    del gc_calls[:]
    with pytest.raises(KeyboardInterrupt):
        with _automatic_gc_paused():
            assert not gc.isenabled()
            raise KeyboardInterrupt
    assert gc.isenabled() is enabled_before
    assert collections(gc_calls) == []


class CollectFailed(Exception):
    pass


@pytest.mark.parametrize("error", [CollectFailed, Interrupted])
@pytest.mark.parametrize("path", ["pause", "reform_free", "reform"])
def test_a_collection_that_raises_still_restores_collection(
    gc_calls, monkeypatch, path, error
):
    def failing_collect(generation=2):
        gc_calls.append(f"collect({generation})")
        raise error("collect")

    monkeypatch.setattr(gc, "collect", failing_collect)
    set_gc(True)
    baseline = RecordingSystem([])
    del gc_calls[:]
    with pytest.raises(error):
        if path == "pause":
            with _automatic_gc_paused():
                pass
        elif path == "reform_free":
            reform_free(baseline)
        else:
            miss(baseline, 0)
    assert gc_calls == ["disable", FULL, "enable"]
    assert gc.isenabled()


REFORM_NAMES = ["a.reform", "b.reform", "c.reform"]
EXTENSION_NAMES = ["x.extension", "y.extension"]

request_strategy = st.tuples(
    st.lists(st.sampled_from(REFORM_NAMES), max_size=3),
    st.sampled_from([None, "", "gov.rate:1", "gov.rate:2"]),
    st.lists(st.sampled_from(EXTENSION_NAMES), max_size=2),
    st.booleans(),
    # Each call's reform-cache bound, from disabled to never evicting.
    st.sampled_from([0, 1, 2, 3, None, NO_EVICTION]),
)


@hypothesis.settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
)
@hypothesis.given(requests=st.lists(request_strategy, max_size=25))
def test_the_cache_matches_a_reference_lru_and_keeps_the_callers_state(
    gc_calls, monkeypatch, requests
):
    # Each example starts from an empty cache.
    monkeypatch.setattr(
        test_runner, "_tax_benefit_system_cache", weakref.WeakKeyDictionary()
    )
    log = []
    baseline = RecordingSystem(log)
    # What the cache should hold: the reform-free system and an LRU of
    # reform systems.
    reform_free_system = None
    reference = OrderedDict()
    forced = 0
    ever_dropped = False
    for reforms, reform_key, extensions, enabled_before, cache_size in requests:
        bound = (
            test_runner.DEFAULT_REFORM_CACHE_SIZE if cache_size is None else cache_size
        )
        is_reform_free = not reforms and not extensions and not reform_key
        key = (tuple(reforms), reform_key, frozenset(extensions))
        # The call trims to its bound first; the count is taken after that.
        while len(reference) > bound:
            reference.popitem(last=False)
            ever_dropped = True
        count_before = (reform_free_system is not None) + len(reference)
        clones_before = sum(name == "clone" for name, _ in log)
        set_gc(enabled_before)
        del gc_calls[:]

        system = _get_tax_benefit_system(
            baseline,
            list(reforms),
            list(extensions),
            reform_key=reform_key,
            cache_size=cache_size,
        )

        assert gc.isenabled() is enabled_before
        assert all(not enabled for _, enabled in log)
        built = sum(name == "clone" for name, _ in log) > clones_before
        collects = built and enabled_before and count_before < LIMIT
        assert collections(gc_calls) == ([FULL] if collects else [])
        forced += collects
        # The bound on forced collections holds while the cache only grows.
        if not ever_dropped:
            assert forced <= LIMIT
        if is_reform_free:
            if reform_free_system is None:
                assert built
                reform_free_system = system
            assert system is reform_free_system
            assert not built or gc_calls[0] == "disable"
        elif key in reference:
            reference.move_to_end(key)
            assert system is reference[key]
            assert not built
            assert gc_calls == []
        else:
            assert built
            assert system is not reform_free_system
            assert all(system is not other for other in reference.values())
            assert system.reforms == tuple(reforms)
            assert sorted(system.extensions) == sorted(extensions)
            if bound > 0:
                reference[key] = system
                if len(reference) > bound:
                    reference.popitem(last=False)
                    ever_dropped = True
            else:
                ever_dropped = True
        assert test_runner._cached_system_count() == (
            (reform_free_system is not None) + len(reference)
        )


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
            patch.setattr(
                test_runner,
                "_automatic_gc_paused",
                lambda collect=True: contextlib.nullcontext(),
            )
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
