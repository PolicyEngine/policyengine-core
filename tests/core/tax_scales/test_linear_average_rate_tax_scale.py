import numpy
import pytest

from policyengine_core import parameters, periods, taxscales, tools


def test_bracket_indices():
    tax_base = numpy.array([0, 1, 2, 3, 4, 5])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(2, 0)
    tax_scale.add_bracket(4, 0)

    result = tax_scale.bracket_indices(tax_base)

    tools.assert_near(result, [0, 0, 0, 1, 1, 2])


def test_bracket_indices_with_factor():
    tax_base = numpy.array([0, 1, 2, 3, 4, 5])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(2, 0)
    tax_scale.add_bracket(4, 0)

    result = tax_scale.bracket_indices(tax_base, factor=2.0)

    tools.assert_near(result, [0, 0, 0, 0, 1, 1])


def test_bracket_indices_with_round_decimals():
    tax_base = numpy.array([0, 1, 2, 3, 4, 5])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(2, 0)
    tax_scale.add_bracket(4, 0)

    result = tax_scale.bracket_indices(tax_base, round_decimals=0)

    tools.assert_near(result, [0, 0, 1, 1, 2, 2])


def test_bracket_indices_without_tax_base():
    tax_base = numpy.array([])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(2, 0)
    tax_scale.add_bracket(4, 0)

    with pytest.raises(taxscales.EmptyArgumentError):
        tax_scale.bracket_indices(tax_base)


def test_bracket_indices_without_brackets():
    tax_base = numpy.array([0, 1, 2, 3, 4, 5])
    tax_scale = taxscales.LinearAverageRateTaxScale()

    with pytest.raises(taxscales.EmptyArgumentError):
        tax_scale.bracket_indices(tax_base)


def test_to_dict():
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(100, 0.1)

    result = tax_scale.to_dict()

    assert result == {"0": 0.0, "100": 0.1}


def test_calc():
    tax_base = numpy.array([0.5, 1, 1.5, 2, 2.5, 100])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(1, 0.1)
    tax_scale.add_bracket(2, 0.2)

    result = tax_scale.calc(tax_base)

    # From the last threshold up the average rate stays at 0.2. These bases
    # used to get 0 because they fall in no bracket.
    tools.assert_near(
        result, [0.025, 0.1, 0.225, 0.4, 0.5, 20], absolute_error_margin=1e-10
    )


def test_calc_below_first_threshold():
    tax_base = numpy.array([-5, 0, 5, 10, 15, 20, 30])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(10, 0.1)
    tax_scale.add_bracket(20, 0.2)

    result = tax_scale.calc(tax_base)

    tools.assert_near(result, [0, 0, 0, 1, 2.25, 4, 6], absolute_error_margin=1e-10)


@pytest.mark.parametrize("top_rate", [0, 0.2, 0.9])
def test_calc_ignores_an_infinite_last_threshold(top_rate):
    # ``MarginalRateTaxScale.to_average`` ends scales with an infinite
    # threshold. The bracket it closes then has a slope of 0, which is what a
    # scale without it does from its last finite threshold up.
    tax_base = numpy.array([0.5, 1, 1.5, 2, 2.5, 100, 1e12])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(1, 0.1)
    tax_scale.add_bracket(2, 0.2)
    closed_tax_scale = tax_scale.copy()
    closed_tax_scale.add_bracket(numpy.inf, top_rate)

    tools.assert_near(
        closed_tax_scale.calc(tax_base),
        tax_scale.calc(tax_base),
        absolute_error_margin=1e-10,
    )


def test_calc_agrees_with_to_marginal_from_last_threshold_up():
    tax_base = numpy.array([2, 2.5, 100, 1e9])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(1, 0.1)
    tax_scale.add_bracket(2, 0.2)

    result = tax_scale.calc(tax_base)

    tools.assert_near(
        result, tax_scale.to_marginal().calc(tax_base), absolute_error_margin=1e-6
    )


def test_calc_of_average_rate_parameter_scale():
    # A parameter scale with ``average_rate`` brackets loads as a
    # LinearAverageRateTaxScale, with no infinite last threshold.
    data = {
        "description": "Average rate income tax",
        "brackets": [
            {
                "threshold": {"2020-01-01": {"value": threshold}},
                "average_rate": {"2020-01-01": {"value": rate}},
            }
            for threshold, rate in [
                (1_100, 0.02),
                (10_000, 0.25),
                (100_000, 0.6),
            ]
        ],
    }
    scale = parameters.ParameterScale("average_rate_scale", data, "")
    tax_scale = scale.get_at_instant(periods.Instant((2020, 6, 1)))

    result = tax_scale.calc(numpy.array([500, 1_100, 100_000, 250_000]))

    assert isinstance(tax_scale, taxscales.LinearAverageRateTaxScale)
    tools.assert_near(result, [0, 22, 60_000, 150_000], absolute_error_margin=1e-6)


def test_to_marginal():
    tax_base = numpy.array([1, 1.5, 2, 2.5])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, 0)
    tax_scale.add_bracket(1, 0.1)
    tax_scale.add_bracket(2, 0.2)

    result = tax_scale.to_marginal()

    assert result.thresholds == [0, 1, 2]
    # ``assert_near`` now compares in float64 instead of float32 (bug H6).
    # Values like 0.3 are not exactly representable, so ULP-level error is
    # visible and we need a small tolerance here.
    tools.assert_near(result.rates, [0.1, 0.3, 0.2], absolute_error_margin=1e-10)
    tools.assert_near(
        result.calc(tax_base),
        [0.1, 0.25, 0.4, 0.5],
        absolute_error_margin=1e-10,
    )


@pytest.mark.parametrize("rate", [0, 0.37])
def test_to_marginal_of_a_single_bracket(rate):
    # One bracket used to raise UnboundLocalError: the top rate was read from
    # a loop variable that only the second and later brackets set.
    tax_base = numpy.array([0, 1, 1.5, 1_000])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(0, rate)

    result = tax_scale.to_marginal()

    assert result.thresholds == [0]
    assert result.rates == [rate]
    tools.assert_near(
        result.calc(tax_base), tax_scale.calc(tax_base), absolute_error_margin=1e-10
    )


@pytest.mark.parametrize("rates_type", [list, numpy.array])
def test_to_marginal_uses_last_paired_rate(rates_type):
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.thresholds = [0, 100]
    tax_scale.rates = rates_type([0.1, 0.2, 0.8])

    result = tax_scale.to_marginal()

    assert result.thresholds == [0, 100]
    # Tax at 100 is 100 * 0.2 = 20, so the first marginal rate is 20 / 100.
    # The terminal rate is the last paired rate 0.2, ignoring surplus 0.8.
    assert result.rates == [0.2, 0.2]


@pytest.mark.parametrize("rates_type", [list, numpy.array])
def test_to_marginal_uses_last_paired_rate_at_infinity(rates_type):
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.thresholds = [0, 100, numpy.inf]
    tax_scale.rates = rates_type([0.1, 0.2, 0.3, 0.8])

    result = tax_scale.to_marginal()

    assert result.thresholds == [0, 100]
    # The finite knot gives 100 * 0.2 / 100 = 0.2; the infinity knot
    # supplies the terminal rate 0.3 even though its tax is not integrated.
    assert result.rates == [0.2, 0.3]


@pytest.mark.parametrize("first_threshold", [50, -50])
def test_to_marginal_starts_at_a_first_threshold_with_rate_0(first_threshold):
    # The scale taxes nothing up to and at its first threshold, so the
    # marginal scale starts there. It used to start at 0, which from 50 taxed
    # a base of 50 at 50 * 0.2 = 10 instead of 0.
    tax_base = numpy.array([first_threshold, 100, 1_000])
    tax_scale = taxscales.LinearAverageRateTaxScale()
    tax_scale.add_bracket(first_threshold, 0)
    tax_scale.add_bracket(100, 0.2)

    result = tax_scale.to_marginal()

    # Tax at 100 is 100 * 0.2 = 20, spread over the bracket from the first
    # threshold.
    assert result.thresholds == [first_threshold, 100]
    tools.assert_near(
        result.rates, [20 / (100 - first_threshold), 0.2], absolute_error_margin=1e-12
    )
    tools.assert_near(
        result.calc(tax_base), tax_scale.calc(tax_base), absolute_error_margin=1e-10
    )


def test_to_marginal_of_no_brackets():
    result = taxscales.LinearAverageRateTaxScale().to_marginal()

    assert result.thresholds == []
    assert result.rates == []
