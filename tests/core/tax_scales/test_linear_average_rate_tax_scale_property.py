"""Properties of ``LinearAverageRateTaxScale.calc``.

For any scale with 1 to 6 brackets and finite thresholds, and any tax base:

* from the first threshold up, the tax is the base times the rate that
  ``numpy.interp`` gives: linear between thresholds, and the last rate from the
  last threshold up. Below the first threshold the tax is 0, except that a
  scale with one bracket applies its rate to every base;
* at each threshold the tax is the threshold times its rate, and above the
  first threshold the tax has no jump there;
* adding a last threshold of infinity, at any rate, does not change the tax;
* when the first threshold is 0, ``to_marginal`` levies the same tax at each
  threshold and above the last one, within accumulated floating-point error.

Examples are in ``test_linear_average_rate_tax_scale.py``.
"""

from __future__ import annotations

import numpy
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core import taxscales  # noqa: E402

RATES = st.integers(min_value=-1_000, max_value=1_000).map(lambda r: r / 1_000)
BASES = st.lists(
    st.floats(min_value=-1e7, max_value=1e7, allow_nan=False),
    max_size=20,
)


@st.composite
def _brackets(
    draw, first_threshold=st.integers(min_value=0, max_value=1_000), min_size=1
):
    first = draw(first_threshold)
    upper_thresholds = draw(
        st.lists(
            st.integers(min_value=first + 1, max_value=1_000_000),
            min_size=min_size - 1,
            max_size=5,
            unique=True,
        )
    )
    thresholds = [first] + sorted(upper_thresholds)
    rates = draw(st.lists(RATES, min_size=len(thresholds), max_size=len(thresholds)))
    return thresholds, rates


def _scale(thresholds, rates):
    tax_scale = taxscales.LinearAverageRateTaxScale()
    for threshold, rate in zip(thresholds, rates):
        tax_scale.add_bracket(threshold, rate)
    return tax_scale


def _bases(thresholds, extra_bases):
    # Every threshold, its neighbours on both sides, and points well above
    # the last threshold, where bases used to get 0.
    near = [
        numpy.nextafter(threshold, direction)
        for threshold in thresholds
        for direction in (-numpy.inf, numpy.inf)
    ]
    far = [thresholds[-1] + 1, 2 * thresholds[-1] + 1, 1e9]
    return numpy.array(thresholds + near + far + extra_bases, dtype=float)


@settings(max_examples=300, deadline=None)
@given(_brackets(), BASES)
def test_calc_interpolates_average_rate(brackets, extra_bases):
    thresholds, rates = brackets
    tax_base = _bases(thresholds, extra_bases)

    result = _scale(thresholds, rates).calc(tax_base)

    expected = tax_base * numpy.interp(tax_base, thresholds, rates)
    if len(thresholds) > 1:
        expected = numpy.where(tax_base >= thresholds[0], expected, 0)
    numpy.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-6)


@settings(max_examples=300, deadline=None)
@given(_brackets())
@example(brackets=([0, 999_999, 1_000_000], [0, 1, 0]))
def test_calc_is_continuous_at_thresholds(brackets):
    thresholds, rates = brackets
    tax_scale = _scale(thresholds, rates)
    at_threshold = numpy.array(thresholds, dtype=float)
    below_threshold = numpy.nextafter(at_threshold, -numpy.inf)

    result = tax_scale.calc(at_threshold)

    numpy.testing.assert_allclose(
        result, at_threshold * numpy.array(rates), rtol=1e-9, atol=1e-6
    )
    # Below the first threshold is outside the scale, so start at the second.
    rates_array = numpy.array(rates)
    slope = numpy.diff(rates_array) / numpy.diff(at_threshold)
    below = below_threshold[1:]
    distance = at_threshold[1:] - below
    # For T(x) = x * r(x), T(t) - T(b) = (t-b) * (r(t) + b*slope).
    # The explicit example has distance 2**-33 and slope -1, so its tax
    # movement is (1_000_000 - 2**-33) * 2**-33 = 0.0001164153218269.
    movement = distance * (numpy.abs(rates_array[1:]) + numpy.abs(below * slope))
    # Budget interpolation arithmetic using its terms, including cancellation
    # near a zero rate, rather than only the resulting tax's magnitude.
    rounding = (
        8
        * numpy.finfo(numpy.float64).eps
        * at_threshold[1:]
        * (numpy.abs(rates_array[:-1]) + numpy.abs((below - at_threshold[:-1]) * slope))
    )
    numpy.testing.assert_array_less(
        numpy.abs(tax_scale.calc(below_threshold)[1:] - result[1:]),
        1e-6 + movement + rounding,
    )


@settings(max_examples=300, deadline=None)
@given(_brackets(), RATES, BASES)
def test_infinite_last_threshold_changes_nothing(brackets, top_rate, extra_bases):
    thresholds, rates = brackets
    tax_base = _bases(thresholds, extra_bases)

    if len(thresholds) == 1:
        # One bracket applies its rate below its threshold too; two do not.
        tax_base = tax_base[tax_base >= thresholds[0]]

    open_scale = _scale(thresholds, rates)
    closed_scale = _scale(thresholds + [numpy.inf], rates + [top_rate])

    numpy.testing.assert_array_equal(
        closed_scale.calc(tax_base), open_scale.calc(tax_base)
    )


@settings(max_examples=300, deadline=None)
@given(_brackets(first_threshold=st.just(0), min_size=2))
@example(brackets=([0, 999_999, 1_000_000], [0, 1, 0]))
def test_calc_agrees_with_to_marginal_at_thresholds_and_above(brackets):
    # One bracket is left out: ``to_marginal`` cannot convert it yet.
    thresholds, rates = brackets
    tax_scale = _scale(thresholds, rates)
    tax_base = numpy.array(
        thresholds + [thresholds[-1] + 1, 2 * thresholds[-1] + 1, 1e9], dtype=float
    )

    marginal_scale = tax_scale.to_marginal()
    lower = numpy.array(marginal_scale.thresholds, dtype=float)
    upper = numpy.append(lower[1:], numpy.inf)
    # Marginal calc shifts thresholds by 1 + eps. Each active bracket's
    # endpoint magnitudes bound this shift and width subtraction; weighting
    # by |marginal rate| also covers cancellation between large bracket taxes.
    endpoints = lower + numpy.minimum(tax_base[:, None], upper)
    endpoints = numpy.where(tax_base[:, None] >= lower, endpoints, 0)
    # The stored threshold*rate taxes telescope in exact arithmetic. Allow
    # rounding in secant conversion, widths, products and the bracket sum.
    rounding = (
        (len(lower) + 8)
        * numpy.finfo(numpy.float64).eps
        * numpy.dot(endpoints, numpy.abs(marginal_scale.rates))
    )
    # In the explicit example marginal rates are [1, -999_999, 0]. Shifting
    # 999_999 by 2**-32 leaves residual tax 1_000_000 * 2**-32
    # = 0.0002328306436539 at the final threshold, whose average tax is 0.
    numpy.testing.assert_array_less(
        numpy.abs(tax_scale.calc(tax_base) - marginal_scale.calc(tax_base)),
        1e-6 + rounding,
    )
