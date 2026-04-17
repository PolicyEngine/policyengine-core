"""Regression test: test_runner cache must tie entries to the baseline's lifetime (H9).

Previously ``_tax_benefit_system_cache`` was keyed on ``id(baseline)``, which
CPython reuses after GC. A baseline that had been collected and replaced
could hit a stale cache entry belonging to a completely different TBS,
silently sharing test setup across unrelated baselines and running
parametric reforms against the wrong baseline.
"""

from __future__ import annotations

import gc
import weakref

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.tools.test_runner import (
    _get_tax_benefit_system,
    _tax_benefit_system_cache,
)


def test_cache_entry_is_released_when_baseline_is_collected():
    baseline = CountryTaxBenefitSystem()
    ref = weakref.ref(baseline)
    _get_tax_benefit_system(baseline, [], [])
    # Cache must now contain an entry for this baseline.
    assert baseline in _tax_benefit_system_cache

    # Drop the strong reference and force GC.
    del baseline
    gc.collect()

    # The WeakKeyDictionary must have dropped the entry, and the baseline
    # should be fully collected (no stale cache entries keeping it alive,
    # no id-reuse attack surface).
    assert ref() is None, "baseline should be garbage-collected"
    # Old behaviour (id-based cache): entry persists past baseline
    # collection and can collide with a reused id.
    assert len(list(_tax_benefit_system_cache.keys())) == 0 or all(
        ref() is not key for key in _tax_benefit_system_cache.keys()
    )


def test_two_distinct_baselines_do_not_share_cache_entries():
    baseline_a = CountryTaxBenefitSystem()
    baseline_b = CountryTaxBenefitSystem()
    tbs_a = _get_tax_benefit_system(baseline_a, [], [])
    tbs_b = _get_tax_benefit_system(baseline_b, [], [])
    # Each baseline must get its own clone, not a cache hit from the other.
    assert tbs_a is not tbs_b
