import numpy
from pytest import fixture

from policyengine_core import parameters, periods, taxscales, tools


@fixture
def data():
    return {
        "description": "Social security contribution tax scale",
        "metadata": {
            "type": "single_amount",
            "threshold_unit": "currency-EUR",
            "rate_unit": "/1",
        },
        "brackets": [
            {
                "threshold": {"2017-10-01": {"value": 0.23}},
                "amount": {
                    "2017-10-01": {"value": 6},
                },
            }
        ],
    }


def test_calc():
    tax_base = numpy.array([1, 8, 10])
    tax_scale = taxscales.SingleAmountTaxScale()
    tax_scale.add_bracket(6, 0.23)
    tax_scale.add_bracket(9, 0.29)

    result = tax_scale.calc(tax_base)

    tools.assert_near(result, [0, 0.23, 0.29])


def test_to_dict():
    tax_scale = taxscales.SingleAmountTaxScale()
    tax_scale.add_bracket(6, 0.23)
    tax_scale.add_bracket(9, 0.29)

    result = tax_scale.to_dict()

    assert result == {"6": 0.23, "9": 0.29}


def test_calc_preserves_boolean_dtype():
    # A single_amount scale whose amounts are booleans (amount_unit: bool)
    # must return a boolean array, not an int64 0/1 array. Otherwise logical
    # NOT (~) silently becomes a bitwise negation on the result.
    tax_scale = taxscales.SingleAmountTaxScale()
    tax_scale.add_bracket(0, True)
    tax_scale.add_bracket(16, False)

    result = tax_scale.calc(numpy.array([10.0, 20.0]))

    assert result.dtype == numpy.bool_
    assert result.tolist() == [True, False]


def test_calc_boolean_result_supports_logical_not():
    # The regression this guards: ~ on the int64 output was a bitwise negation
    # (~1 == -2, ~0 == -1, both truthy), so negating a bool bracket produced
    # wrong results. On a real bool array, ~ is the intended logical NOT.
    tax_scale = taxscales.SingleAmountTaxScale()
    tax_scale.add_bracket(0, True)
    tax_scale.add_bracket(16, False)

    result = tax_scale.calc(numpy.array([10.0, 20.0]))
    negated = ~result

    assert negated.dtype == numpy.bool_
    assert negated.tolist() == [False, True]


def test_calc_preserves_integer_dtype():
    # Numeric int amounts must be unaffected by the boolean-dtype fix: the
    # result stays int64 with identical values.
    tax_scale = taxscales.SingleAmountTaxScale()
    tax_scale.add_bracket(0, 100)
    tax_scale.add_bracket(16, 200)

    result = tax_scale.calc(numpy.array([10.0, 20.0]))

    assert result.dtype == numpy.int64
    assert result.tolist() == [100, 200]


def test_calc_preserves_float_dtype():
    # Numeric float amounts must be unaffected: the result stays float64.
    tax_scale = taxscales.SingleAmountTaxScale()
    tax_scale.add_bracket(6, 0.23)
    tax_scale.add_bracket(9, 0.29)

    result = tax_scale.calc(numpy.array([1, 8, 10]))

    assert result.dtype == numpy.float64
    tools.assert_near(result, [0, 0.23, 0.29])


def test_calc_out_of_range_bool_returns_false_sentinel():
    # Inputs that fall outside every bracket must get the zero-valued sentinel
    # in the amounts' own dtype: False for a boolean scale, not int 0.
    tax_scale = taxscales.SingleAmountTaxScale()
    # digitize guards with [-inf, threshold..., inf]; the -inf..first-threshold
    # cell is a real bracket, so use a scale where only interior cells map to a
    # bracket and confirm the sentinel dtype anyway via a below-all base.
    tax_scale.add_bracket(0, True)

    result = tax_scale.calc(numpy.array([-5.0]))

    assert result.dtype == numpy.bool_


def test_calc_empty_scale_stays_int():
    # An empty scale has no amounts to infer a dtype from; preserve the
    # historical behaviour of returning an int64 zero array.
    tax_scale = taxscales.SingleAmountTaxScale()

    result = tax_scale.calc(numpy.array([10.0]))

    assert result.dtype == numpy.int64
    assert result.tolist() == [0]


def test_calc_boolean_dtype_end_to_end_via_parameter_scale():
    # Mirror how policyengine-us declares boolean single_amount brackets
    # (amount_unit: bool, amount: true/false) and confirm the dtype survives
    # the full ParameterScale -> get_at_instant -> SingleAmountTaxScale path.
    data = {
        "description": "Boolean age-exemption single_amount scale",
        "metadata": {
            "type": "single_amount",
            "threshold_unit": "year",
            "amount_unit": "bool",
        },
        "brackets": [
            {
                "threshold": {"2022-01-01": {"value": 0}},
                "amount": {"2022-01-01": {"value": True}},
            },
            {
                "threshold": {"2022-01-01": {"value": 16}},
                "amount": {"2022-01-01": {"value": False}},
            },
        ],
    }
    scale = parameters.ParameterScale("bool_scale", data, "")
    scale_at_instant = scale.get_at_instant(periods.Instant((2022, 6, 1)))

    result = scale_at_instant.calc(numpy.array([10.0, 20.0]))

    assert result.dtype == numpy.bool_
    assert result.tolist() == [True, False]
    assert (~result).tolist() == [False, True]


# TODO: move, as we're testing Scale, not SingleAmountTaxScale
def test_assign_thresholds_on_creation(data):
    scale = parameters.ParameterScale("amount_scale", data, "")
    first_jan = periods.Instant((2017, 11, 1))
    scale_at_instant = scale.get_at_instant(first_jan)

    result = scale_at_instant.thresholds

    assert result == [0.23]


# TODO: move, as we're testing Scale, not SingleAmountTaxScale
def test_assign_amounts_on_creation(data):
    scale = parameters.ParameterScale("amount_scale", data, "")
    first_jan = periods.Instant((2017, 11, 1))
    scale_at_instant = scale.get_at_instant(first_jan)

    result = scale_at_instant.amounts

    assert result == [6]


# TODO: move, as we're testing Scale, not SingleAmountTaxScale
def test_dispatch_scale_type_on_creation(data):
    scale = parameters.ParameterScale("amount_scale", data, "")
    first_jan = periods.Instant((2017, 11, 1))

    result = scale.get_at_instant(first_jan)

    assert isinstance(result, taxscales.SingleAmountTaxScale)
