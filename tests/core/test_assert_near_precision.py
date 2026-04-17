"""Regression test: ``assert_near`` must not collapse large values via float32 (H6).

Previously both operands were cast to ``np.float32`` before the diff check,
so any pair that happened to round to the same float32 was treated as
equal — even with ``absolute_error_margin=0``. Values above ~16M are the
practical concern: $25_000_000 and $25_000_001 both round to the same
float32, so a test expecting one would silently pass on the other.
"""

from __future__ import annotations

import pytest

from policyengine_core.tools import assert_near


def test_assert_near_detects_large_integer_difference():
    # These two values differ by 1, but they both round to the same float32.
    # Under the old implementation this assertion PASSED (silent failure of
    # the test).
    with pytest.raises(AssertionError):
        assert_near(25_000_000, 25_000_001, absolute_error_margin=0)


def test_assert_near_still_passes_within_margin():
    # Float64 precision must not break legitimate near-equality checks.
    assert_near(1.0, 1.0005, absolute_error_margin=1e-2)
