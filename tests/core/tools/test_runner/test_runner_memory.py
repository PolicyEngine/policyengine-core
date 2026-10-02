"""The YAML runner must not keep per-case state alive across a run.

pytest holds every collected item until the session ends. When each
``YamlItem`` kept its simulation, and the runner kept every reform system it
ever built, memory grew with every case: a policyengine-us run of ~1,500
files in one process reached 118 GB.
"""

import gc
import weakref

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.reforms import Reform, set_parameter
from policyengine_core.tools import test_runner
from policyengine_core.tools.test_runner import (
    DEFAULT_REFORM_CACHE_SIZE,
    _get_tax_benefit_system,
    _tax_benefit_system_cache,
)
from tests.fixtures.yaml_runner_memory import run_with_probe, write_cases


def _rate_reform(rate):
    """An inline reform class like the one the runner builds for a dotted
    parameter input, and the ``reform_key`` it would get."""
    modifier = set_parameter(
        "taxes.income_tax_rate", rate, return_modifier=True, period="year:2000:40"
    )

    class inline_reform_class(Reform):
        def apply(self):
            self.parameters = modifier(self.parameters)

    return [inline_reform_class], f"taxes.income_tax_rate:{rate}"


def _system_for_rate(baseline, rate, cache_size=None):
    reforms, key = _rate_reform(rate)
    return _get_tax_benefit_system(
        baseline, reforms, [], reform_key=key, cache_size=cache_size
    )


def _rate(system):
    return system.parameters.taxes.income_tax_rate("2017-01-01")


def test_reform_cache_keeps_at_most_cache_size_reform_systems():
    baseline = CountryTaxBenefitSystem()
    refs = [weakref.ref(_system_for_rate(baseline, 0.2 + i / 100)) for i in range(8)]
    gc.collect()

    cached = _tax_benefit_system_cache[baseline].reforms
    assert len(cached) == DEFAULT_REFORM_CACHE_SIZE
    # Evicted systems are freed, not just dropped from the cache.
    alive = [ref() for ref in refs if ref() is not None]
    assert len(alive) == DEFAULT_REFORM_CACHE_SIZE
    assert set(map(id, alive)) == set(map(id, cached.values()))


def test_reform_cache_evicts_least_recently_used():
    baseline = CountryTaxBenefitSystem()
    a = _system_for_rate(baseline, 0.21, cache_size=2)
    b = _system_for_rate(baseline, 0.22, cache_size=2)
    assert _system_for_rate(baseline, 0.21, cache_size=2) is a  # hit; A is newest
    _system_for_rate(baseline, 0.23, cache_size=2)  # evicts B, not A
    assert _system_for_rate(baseline, 0.21, cache_size=2) is a
    rebuilt_b = _system_for_rate(baseline, 0.22, cache_size=2)
    assert rebuilt_b is not b
    assert _rate(rebuilt_b) == pytest.approx(0.22)


def test_reform_free_system_is_kept_through_evictions():
    baseline = CountryTaxBenefitSystem()
    reform_free = _get_tax_benefit_system(baseline, [], [], reform_key="")
    for i in range(6):
        _system_for_rate(baseline, 0.3 + i / 100, cache_size=1)
    assert _get_tax_benefit_system(baseline, [], [], reform_key="") is reform_free
    assert _get_tax_benefit_system(baseline, [], []) is reform_free
    assert reform_free is not baseline
    assert _rate(reform_free) == pytest.approx(0.15)
    assert _rate(baseline) == pytest.approx(0.15)


def test_cache_size_zero_builds_a_fresh_reform_system_each_time():
    baseline = CountryTaxBenefitSystem()
    first = _system_for_rate(baseline, 0.25, cache_size=0)
    second = _system_for_rate(baseline, 0.25, cache_size=0)
    assert first is not second
    assert _rate(first) == _rate(second) == pytest.approx(0.25)
    assert len(_tax_benefit_system_cache[baseline].reforms) == 0


def test_negative_cache_size_is_rejected():
    with pytest.raises(ValueError):
        _system_for_rate(CountryTaxBenefitSystem(), 0.25, cache_size=-1)


def test_finished_cases_release_their_simulations(tmp_path):
    path = write_cases(tmp_path / "cases.yaml", 60)
    probe = run_with_probe(CountryTaxBenefitSystem(), path)
    assert len(probe.outcomes) == 60
    # Every item is still referenced by the session here; none may keep the
    # simulation it ran.
    assert probe.live_simulations == 0


def test_live_systems_do_not_grow_with_the_number_of_cases(tmp_path):
    short = run_with_probe(
        CountryTaxBenefitSystem(), write_cases(tmp_path / "short.yaml", 12)
    )
    long = run_with_probe(
        CountryTaxBenefitSystem(), write_cases(tmp_path / "long.yaml", 90)
    )
    # 4 vs 30 distinct reform systems requested; the live count must not move.
    assert long.live_systems == short.live_systems


@pytest.mark.parametrize("reform_every", [0, 3])
def test_traced_memory_stays_flat_over_hundreds_of_cases(tmp_path, reform_every):
    """A few hundred cases in one process: memory after the last case is
    within a small bound of memory after the first hundred."""
    path = write_cases(tmp_path / "cases.yaml", 300, reform_every=reform_every)
    probe = run_with_probe(CountryTaxBenefitSystem(), path)
    assert len(probe.outcomes) == 300
    # One sample every 10 cases: index 9 is after case 100.
    growth = max(probe.traced[10:]) - probe.traced[9]
    # Measured over these 200 cases: 0.5 MB with the fix (pytest's own
    # per-case reports); 4.0 MB (plain cases) and 18.4 MB (every third case a
    # reform) without it.
    assert growth < 2_000_000, f"traced memory grew {growth / 1e6:.1f} MB"


def test_reform_cache_size_option_reaches_the_cache(tmp_path, monkeypatch):
    seen = []
    original = test_runner._get_tax_benefit_system

    def spy(*args, **kwargs):
        seen.append(kwargs.get("cache_size"))
        return original(*args, **kwargs)

    monkeypatch.setattr(test_runner, "_get_tax_benefit_system", spy)
    path = write_cases(tmp_path / "cases.yaml", 3)
    run_with_probe(CountryTaxBenefitSystem(), path, {"reform_cache_size": 0})
    assert seen == [0, 0, 0]


@pytest.mark.parametrize(
    "arguments, expected", [([], None), (["--reform-cache-size", "3"], 3)]
)
def test_command_line_passes_reform_cache_size(monkeypatch, arguments, expected):
    from policyengine_core.scripts import policyengine_command, run_test

    seen = {}

    def fake_run_tests(tax_benefit_system, paths, options):
        seen.update(options)
        return 0

    monkeypatch.setattr(run_test, "run_tests", fake_run_tests)
    monkeypatch.setattr(run_test, "build_tax_benefit_system", lambda *args: object())
    monkeypatch.setattr(
        "sys.argv", ["policyengine-core", "test", "cases.yaml", *arguments]
    )
    with pytest.raises(SystemExit) as exit_info:
        run_test.main(policyengine_command.get_parser())
    assert exit_info.value.code == 0
    assert seen["reform_cache_size"] == expected
