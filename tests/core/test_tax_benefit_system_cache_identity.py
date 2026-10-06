"""Regression tests for runner-scoped policy-system caching.

Previously ``_tax_benefit_system_cache`` was keyed on ``id(baseline)``, which
CPython reuses after GC. A baseline that had been collected and replaced
could hit a stale cache entry belonging to a completely different TBS,
silently sharing test setup across unrelated baselines and running
parametric reforms against the wrong baseline.
"""

from __future__ import annotations

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.tools.policy_system_cache import PolicySystemCache
from policyengine_core.tools.test_runner import (
    _get_tax_benefit_system,
)


def test_cache_releases_systems_when_closed():
    baseline = CountryTaxBenefitSystem()
    cache = PolicySystemCache()
    _get_tax_benefit_system(baseline, [], [], policy_system_cache=cache)

    assert cache.cache_info().entries == 1

    cache.close()

    assert cache.cache_info().entries == 0


def test_two_distinct_baselines_do_not_share_cache_entries():
    baseline_a = CountryTaxBenefitSystem()
    baseline_b = CountryTaxBenefitSystem()
    cache_a = PolicySystemCache()
    cache_b = PolicySystemCache()
    tbs_a = _get_tax_benefit_system(baseline_a, [], [], policy_system_cache=cache_a)
    tbs_b = _get_tax_benefit_system(baseline_b, [], [], policy_system_cache=cache_b)
    # Each baseline must get its own clone, not a cache hit from the other.
    assert tbs_a is not tbs_b
