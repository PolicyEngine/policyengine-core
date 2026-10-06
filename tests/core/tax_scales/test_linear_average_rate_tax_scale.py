import numpy
import pytest

from policyengine_core import taxscales, tools


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


def test_to_marginal_of_no_brackets():
    result = taxscales.LinearAverageRateTaxScale().to_marginal()

    assert result.thresholds == []
    assert result.rates == []
