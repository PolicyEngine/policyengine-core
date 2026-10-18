"""Properties of converting between marginal and linear-average rate scales.

The marginal scales have 1 to 6 brackets, rates in [-1, 1], and a first
threshold of 0 or any integer in [-1,000, 1,000]. Their upper thresholds are
strictly increasing integers up to 1,000,000, none of them 0. Examples in
``test_marginal_rate_tax_scale.py`` cover a threshold of 0 above the first.
For these scales:

* ``to_average`` raises ``ValueError`` for one bracket with a nonzero rate and
  a threshold other than 0, and converts every other scale;
* ``to_average`` and then ``to_marginal`` give back the same thresholds and
  rates (one bracket from 0 used to raise ``UnboundLocalError``, and a first
  threshold other than 0 came back as 0);
* the average scale levies the marginal tax at every threshold, and its last
  bracket starts at infinity with the top rate;
* for one bracket, the average scale levies the marginal tax at every finite
  base within [-1,000,000, 1,000,000].

With two or more brackets, the average scale keeps the average rate of the
last finite threshold above it (see ``LinearAverageRateTaxScale.calc``). There
its tax matches the marginal tax only when the top rate is that average rate,
so no property here claims it.

For a linear-average rate scale with first threshold 0, rates in [0, 1] and
such upper thresholds, ``to_marginal`` gives a marginal scale that levies
threshold x rate at every threshold (the average rate there), and the top rate
on the whole base above the last one.

Appending surplus rates to a scale with first threshold 0 leaves both
conversion results unchanged, preserving the last paired terminal rate.

Examples are in ``test_marginal_rate_tax_scale.py`` and
``test_linear_average_rate_tax_scale.py``.
"""

from __future__ import annotations

import numpy
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import assume, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core import taxscales  # noqa: E402


NONNEGATIVE_RATES = st.integers(min_value=0, max_value=1_000).map(lambda r: r / 1_000)
RATES = st.integers(min_value=-1_000, max_value=1_000).map(lambda r: r / 1_000)
FIRST_THRESHOLDS = st.one_of(st.just(0), st.integers(min_value=-1_000, max_value=1_000))


@st.composite
def _brackets(
    draw, first_threshold=st.just(0), rate=NONNEGATIVE_RATES, min_upper_thresholds=0
):
    first = draw(first_threshold)
    upper_thresholds = draw(
        st.lists(
            st.integers(min_value=first + 1, max_value=1_000_000).filter(
                lambda threshold: threshold != 0
            ),
            min_size=min_upper_thresholds,
            max_size=5,
            unique=True,
        )
    )
    thresholds = [first] + sorted(upper_thresholds)
    rates = draw(st.lists(rate, min_size=len(thresholds), max_size=len(thresholds)))
    return thresholds, rates


def _marginal(thresholds, rates):
    tax_scale = taxscales.MarginalRateTaxScale()
    for threshold, rate in zip(thresholds, rates):
        tax_scale.add_bracket(threshold, rate)
    return tax_scale


def _no_average_scale(thresholds, rates):
    # One bracket taxes rate x (base - threshold) from its threshold up, an
    # average rate of rate x (1 - threshold / base). That is linear in the base
    # only when the threshold or the rate is 0.
    return len(thresholds) == 1 and thresholds[0] != 0 and rates[0] != 0


@settings(max_examples=300, deadline=None)
@given(_brackets(first_threshold=FIRST_THRESHOLDS, rate=RATES))
@example(brackets=([50], [0.37]))
@example(brackets=([-50], [0.37]))
@example(brackets=([50], [0.0]))
@example(brackets=([0], [0.37]))
def test_to_average_rejects_only_one_bracket_away_from_0(brackets):
    thresholds, rates = brackets
    marginal = _marginal(thresholds, rates)

    if _no_average_scale(thresholds, rates):
        with pytest.raises(ValueError, match="cannot convert one bracket"):
            marginal.to_average()
    else:
        marginal.to_average()


@settings(max_examples=300, deadline=None)
@given(_brackets(first_threshold=FIRST_THRESHOLDS, rate=RATES))
@example(brackets=([0], [0.37]))
@example(brackets=([50], [0.0]))
@example(brackets=([-50], [0.0]))
@example(brackets=([50, 100], [0.1, 0.2]))
@example(brackets=([-50, 100], [0.1, 0.2]))
@example(brackets=([-100, -50, 100], [0.1, 0.2, 0.3]))
def test_marginal_to_average_round_trips(brackets):
    thresholds, rates = brackets
    assume(not _no_average_scale(thresholds, rates))
    marginal = _marginal(thresholds, rates)

    average = marginal.to_average()
    round_trip = average.to_marginal()

    assert average.thresholds == [*thresholds, numpy.inf]
    assert average.rates[-1] == rates[-1]
    assert round_trip.thresholds == thresholds
    assert round_trip.rates[-1] == rates[-1]
    # Every other rate comes back as the difference of two thresholds' tax,
    # each threshold x average rate, over the bracket's width. Those taxes
    # round to within a few eps of the sum of |rate| x width.
    eps = numpy.finfo(numpy.float64).eps
    widths = numpy.diff(numpy.array(thresholds, dtype=float))
    largest_tax = numpy.abs(numpy.array(rates[:-1])) @ widths
    tolerance = 16 * eps * largest_tax / widths + 1e-12
    error = numpy.abs(numpy.array(round_trip.rates[:-1]) - rates[:-1])
    assert numpy.all(error <= tolerance), (error, tolerance)


@settings(max_examples=300, deadline=None)
@given(_brackets(first_threshold=FIRST_THRESHOLDS, rate=RATES))
@example(brackets=([50, 100], [0.1, 0.2]))
@example(brackets=([-50, 100], [0.1, 0.2]))
@example(brackets=([-100, -50, 100], [0.1, 0.2, 0.3]))
def test_marginal_to_average_levies_the_tax_at_every_threshold(brackets):
    thresholds, rates = brackets
    assume(not _no_average_scale(thresholds, rates))
    marginal = _marginal(thresholds, rates)
    tax_base = numpy.array(thresholds, dtype=float)

    result = marginal.to_average().calc(tax_base)

    # Both sides sum bracket taxes: the average scale as it builds its rates,
    # the marginal scale with thresholds scaled by 1 + eps. Bound the rounding
    # by each bracket's endpoint magnitudes times its rate.
    eps = numpy.finfo(numpy.float64).eps
    endpoints = numpy.abs(tax_base) + numpy.abs(
        numpy.append(tax_base[1:], tax_base[-1])
    )
    rounding = (len(thresholds) + 8) * eps * (numpy.abs(rates) @ endpoints)
    numpy.testing.assert_array_less(
        numpy.abs(result - marginal.calc(tax_base)), 1e-9 + rounding
    )


@settings(max_examples=300, deadline=None)
@given(
    bracket=st.one_of(
        st.tuples(st.just(0), RATES),
        st.tuples(st.integers(min_value=-1_000, max_value=1_000), st.just(0.0)),
    ),
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
def test_singleton_marginal_to_average_preserves_calculation(bracket, bases):
    threshold, rate = bracket
    marginal = taxscales.MarginalRateTaxScale()
    marginal.add_bracket(threshold, rate)
    tax_base = numpy.array([-1_000, -1, 0, 1, 1.5, 1_000, threshold, *bases])

    average = marginal.to_average()

    # One marginal bracket taxes max(base - threshold, 0) * rate. Here that is
    # max(base, 0) * rate from 0, and 0 at rate 0.
    expected = numpy.maximum(tax_base - threshold, 0) * rate
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
    brackets=_brackets(),
    surplus_rates=st.lists(
        NONNEGATIVE_RATES,
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
