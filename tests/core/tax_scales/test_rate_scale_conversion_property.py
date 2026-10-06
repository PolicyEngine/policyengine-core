"""Properties of converting between marginal and linear-average rate scales.

For any marginal rate scale whose first threshold is 0, with 1 to 6 brackets:

* ``to_average`` and then ``to_marginal`` give back the same thresholds and
  rates (one bracket included; it used to raise ``UnboundLocalError``);
* the average scale's last bracket starts at infinity with the top rate.

For any linear-average rate scale whose first threshold is 0, ``to_marginal``
gives a marginal scale that levies threshold x rate at every threshold (the
average rate there), and the top rate on the whole base above the last one.
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
def test_average_to_marginal_levies_the_average_rate(brackets):
    thresholds, rates = brackets
    average = taxscales.LinearAverageRateTaxScale()
    for threshold, rate in zip(thresholds, rates):
        average.add_bracket(threshold, rate)

    marginal = average.to_marginal()

    # A linear-average scale's rate at each threshold is the average rate
    # there, so the marginal scale levies threshold x rate at every
    # threshold, and the top rate on the whole base from the last one up.
    tax_base = numpy.array(thresholds, dtype=float)
    above = 2.0 * thresholds[-1] + 1
    numpy.testing.assert_allclose(
        marginal.calc(numpy.append(tax_base, above)),
        numpy.append(tax_base * numpy.array(rates), above * rates[-1]),
        rtol=1e-9,
        atol=1e-6,
    )
