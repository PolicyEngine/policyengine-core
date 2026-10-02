"""The YAML runner's reform-system cache against a reference LRU.

For any sequence of requests, each with its own cache size, the cache holds
at most that call's ``cache_size`` reform systems after the call, returns a cached object exactly when a plain
LRU of that size would, always hands back a system built for the reforms and
extensions requested, and keeps one reform-free system throughout.
``test_runner_memory.py`` pins the same behaviour with examples.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.tools.test_runner import (
    _get_tax_benefit_system,
    _tax_benefit_system_cache,
)


class StubSystem:
    """Records what the runner applied to it, in order."""

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
    st.sampled_from(["", "p:1", "p:2"]),
    # Each call's bound; runs on one baseline can ask for different ones.
    st.integers(min_value=0, max_value=3),
)


@hypothesis.given(sequence=st.lists(requests, max_size=40))
@hypothesis.settings(max_examples=300, deadline=None)
def test_cache_matches_reference_lru(sequence):
    baseline = StubSystem()
    reference = OrderedDict()
    last_returned = {}
    reform_free = None

    for reforms, extensions, reform_key, cache_size in sequence:
        # The reference applies each call's bound first, oldest out first.
        while len(reference) > cache_size:
            reference.popitem(last=False)
        system = _get_tax_benefit_system(
            baseline,
            list(reforms),
            list(extensions),
            reform_key=reform_key,
            cache_size=cache_size,
        )
        assert system is not baseline
        # Reforms apply in the order given; extension order is not part of
        # the cache key, so only the set is fixed.
        applied_reforms = [name for kind, name in system.applied if kind == "reform"]
        applied_extensions = {
            name for kind, name in system.applied if kind == "extension"
        }
        assert applied_reforms == list(reforms)
        assert applied_extensions == set(extensions)

        if not reforms and not extensions and not reform_key:
            if reform_free is None:
                reform_free = system
            assert system is reform_free
        else:
            key = (tuple(reforms), reform_key, frozenset(extensions))
            if key in reference:
                reference.move_to_end(key)
                assert system is last_returned[key]
            else:
                assert system is not last_returned.get(key)
                if cache_size:
                    reference[key] = True
                    while len(reference) > cache_size:
                        reference.popitem(last=False)
            last_returned[key] = system

        cached = _tax_benefit_system_cache[baseline]
        assert len(cached.reforms) <= cache_size
        assert list(cached.reforms) == list(reference)
        assert cached.reform_free is reform_free
