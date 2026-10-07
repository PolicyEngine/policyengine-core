"""Runner-local policy-system caching checked against a reference LRU."""

from __future__ import annotations

from collections import OrderedDict

import pytest

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.tools.policy_system_cache import (
    PolicySystemCache,
    PolicySystemCacheKey,
)
from policyengine_core.tools.test_runner import _get_tax_benefit_system


class StubSystem:
    """Record reforms and extensions applied by the YAML runner."""

    def __init__(self, applied=()):
        self.applied = tuple(applied)

    def clone(self):
        return StubSystem(self.applied)

    def apply_reform(self, path):
        return StubSystem(self.applied + (("reform", path),))

    def load_extension(self, extension):
        self.applied += (("extension", extension),)


requests = st.tuples(
    st.lists(st.sampled_from(["a", "b", "c"]), max_size=2),
    st.lists(st.sampled_from(["x", "y"]), max_size=2, unique=True),
    st.dictionaries(
        st.sampled_from(["tax.rate", "benefit.amount"]),
        st.integers(min_value=0, max_value=3),
        max_size=2,
    ),
)


@hypothesis.given(
    capacity=st.integers(min_value=0, max_value=3),
    sequence=st.lists(requests, max_size=30),
)
@hypothesis.settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_runner_local_cache_matches_reference_lru(capacity, sequence):
    baseline = StubSystem()
    cache = PolicySystemCache(max_derived_entries=capacity)
    reference = OrderedDict()
    last_returned = {}
    baseline_system = None

    for reforms, extensions, parameter_overrides in sequence:
        key = PolicySystemCacheKey.from_inputs(
            reforms=reforms,
            extensions=extensions,
            parameter_overrides=parameter_overrides,
        )
        system = _get_tax_benefit_system(
            baseline,
            list(reforms),
            list(extensions),
            parameter_overrides=parameter_overrides,
            policy_system_cache=cache,
        )
        assert system is not baseline
        assert [name for kind, name in system.applied if kind == "reform"] == list(
            reforms
        )
        assert [name for kind, name in system.applied if kind == "extension"] == list(
            extensions
        )

        if key.is_baseline:
            if baseline_system is None:
                baseline_system = system
            assert system is baseline_system
        elif key in reference:
            reference.move_to_end(key)
            assert system is last_returned[key]
        else:
            assert system is not last_returned.get(key)
            if capacity:
                reference[key] = True
                while len(reference) > capacity:
                    reference.popitem(last=False)
        last_returned[key] = system

        # Inspect order without ``in``/``get``: public lookups intentionally
        # refresh recency and would perturb the behavior under test.
        cached_derived = [
            cached_key for cached_key in cache._systems if not cached_key.is_baseline
        ]
        assert cached_derived == list(reference)
        assert len(cached_derived) <= capacity
        assert (PolicySystemCacheKey.from_inputs() in cache) is (
            baseline_system is not None
        )
