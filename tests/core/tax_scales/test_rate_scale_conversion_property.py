"""Properties of converting between marginal and linear-average rate scales.

For any marginal rate scale whose first threshold is 0, with 1 to 6 brackets:

* ``to_average`` and then ``to_marginal`` give back the same thresholds and
  rates (one bracket included; it used to raise ``UnboundLocalError``);
* the average scale's last bracket starts at infinity with the top rate.

For any linear-average rate scale whose first threshold is 0, ``to_marginal``
gives a marginal scale that levies the same tax at every threshold below the
last one (and everywhere, for one bracket).
Examples are in ``test_marginal_rate_tax_scale.py`` and
``test_linear_average_rate_tax_scale.py``.
"""

from __future__ import annotations

import numpy
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core import taxscales  # noqa: E402


@st.composite
def _brackets(draw):
    upper_thresholds = draw(
        st.lists(
            st.integers(min_value=1, max_value=1_000_000),
            max_size=5,
            unique=True,
        )
    )
    thresholds = [0] + sorted(upper_thresholds)
    rates = draw(
        st.lists(
            st.integers(min_value=0, max_value=1_000).map(lambda r: r / 1_000),
            min_size=len(thresholds),
            max_size=len(thresholds),
        )
    )
    return thresholds, rates


@settings(max_examples=300, deadline=None)
@given(_brackets())
def test_marginal_to_average_round_trips(brackets):
    thresholds, rates = brackets
    marginal = taxscales.MarginalRateTaxScale()
    for threshold, rate in zip(thresholds, rates):
        marginal.add_bracket(threshold, rate)

    average = marginal.to_average()
    round_trip = average.to_marginal()

    assert average.thresholds[-1] == numpy.inf
    assert average.rates[-1] == rates[-1]
    assert round_trip.thresholds == thresholds
    numpy.testing.assert_allclose(round_trip.rates, rates, rtol=0, atol=1e-9)


@settings(max_examples=300, deadline=None)
@given(_brackets())
def test_average_to_marginal_keeps_tax_at_thresholds(brackets):
    thresholds, rates = brackets
    average = taxscales.LinearAverageRateTaxScale()
    for threshold, rate in zip(thresholds, rates):
        average.add_bracket(threshold, rate)

    marginal = average.to_marginal()

    # ``LinearAverageRateTaxScale.calc`` levies nothing from its last
    # threshold up when it has several brackets, so compare below that.
    tax_base = numpy.array(thresholds[:-1] if len(thresholds) > 1 else [0, 1, 1e6])
    numpy.testing.assert_allclose(
        marginal.calc(tax_base), average.calc(tax_base), rtol=1e-9, atol=1e-6
    )
