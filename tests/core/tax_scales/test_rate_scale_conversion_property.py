"""Properties of converting between marginal and linear-average rate scales.

For equal-length scales with 1 to 6 brackets, rates in [0, 1], first threshold
0, and strictly increasing positive integer upper thresholds up to 1,000,000:

* ``to_average`` and then ``to_marginal`` give back the same thresholds and
  rates (one bracket included; it used to raise ``UnboundLocalError``);
* the average scale's last bracket starts at infinity with the top rate;
* a singleton marginal conversion preserves tax at finite bases within
  [-1,000,000, 1,000,000].

For such a linear-average rate scale, ``to_marginal``
gives a marginal scale that levies threshold x rate at every threshold (the
average rate there), and the top rate on the whole base above the last one.
Examples are in ``test_marginal_rate_tax_scale.py`` and
``test_linear_average_rate_tax_scale.py``.

Appending surplus rates to a scale with at least two brackets leaves both
conversion results unchanged, preserving the last paired terminal rate.
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
def _brackets(draw, min_upper_thresholds=0):
    upper_thresholds = draw(
        st.lists(
            st.integers(min_value=1, max_value=1_000_000),
            min_size=min_upper_thresholds,
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
@given(
    rate=st.integers(min_value=0, max_value=1_000).map(lambda r: r / 1_000),
    bases=st.lists(
        st.floats(
            min_value=-1_000_000,
            max_value=1_000_000,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=10,
    ),
)
def test_singleton_marginal_to_average_preserves_calculation(rate, bases):
    marginal = taxscales.MarginalRateTaxScale()
    marginal.add_bracket(0, rate)
    tax_base = numpy.array([-1_000, -1, 0, 1, 1.5, 1_000, *bases])

    average = marginal.to_average()

    # One zero-origin marginal bracket taxes max(base, 0) * rate.
    expected = numpy.maximum(tax_base, 0) * rate
    numpy.testing.assert_allclose(average.calc(tax_base), expected, rtol=1e-12)
    numpy.testing.assert_allclose(
        average.calc(tax_base), marginal.calc(tax_base), rtol=1e-12
    )


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
    # ``MarginalRateTaxScale.calc`` shifts each threshold t by t x eps (it
    # scales them by 1 + eps), which moves the tax by about that much times
    # the bracket's rate. Close thresholds near a million can make that rate
    # large, so bound the rounding by the inputs rather than a fixed 1e-6.
    eps = numpy.finfo(numpy.float64).eps
    marginal_rates = numpy.abs(numpy.diff(tax_base * numpy.array(rates))) / (
        numpy.diff(tax_base)
    )
    rounding = 8 * eps * ((marginal_rates * tax_base[1:]).sum() + above)
    numpy.testing.assert_allclose(
        marginal.calc(numpy.append(tax_base, above)),
        numpy.append(tax_base * numpy.array(rates), above * rates[-1]),
        rtol=1e-9,
        atol=1e-6 + rounding,
    )


@pytest.mark.parametrize(
    "scale_type, conversion",
    [
        (taxscales.MarginalRateTaxScale, "to_average"),
        (taxscales.LinearAverageRateTaxScale, "to_marginal"),
    ],
)
@settings(max_examples=300, deadline=None)
@given(
    brackets=_brackets(min_upper_thresholds=1),
    surplus_rates=st.lists(
        st.integers(min_value=0, max_value=1_000).map(lambda r: r / 1_000),
        min_size=1,
        max_size=3,
    ),
)
def test_conversions_ignore_surplus_rates(
    scale_type, conversion, brackets, surplus_rates
):
    thresholds, rates = brackets
    paired = scale_type()
    for threshold, rate in zip(thresholds, rates):
        paired.add_bracket(threshold, rate)
    surplus = paired.copy()
    surplus.rates.extend(surplus_rates)

    expected = getattr(paired, conversion)()
    result = getattr(surplus, conversion)()

    assert result.thresholds == expected.thresholds
    assert result.rates == expected.rates
