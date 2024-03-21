import pytest


def test_parameter_uprating():
    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {
                    "2015-01-01": 1,
                    "2016-01-01": 2,
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-01-01": 1,
                    "2017-01-01": 2,
                    "2018-01-01": 3,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    interpolated = uprate_parameters(root)

    # Interpolate halfway

    assert interpolated.to_be_uprated("2018-01-01") == 2 * 3


def test_parameter_uprating_with_rounding():
    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {
                    "2015-01-01": 1,
                    "2016-01-01": 2,
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "rounding": {
                            "interval": 1,
                            "type": "upwards",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-01-01": 1,
                    "2017-01-01": 1,
                    "2018-01-01": 1.75,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    interpolated = uprate_parameters(root)

    # Interpolate halfway

    assert interpolated.to_be_uprated("2018-01-01") == 4


def test_parameter_uprating_with_self():
    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {
                    "2015-01-01": 1,
                    "2016-01-01": 2,
                },
                "metadata": {
                    "uprating": {
                        "parameter": "self",
                    },
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    # Interpolate halfway

    assert uprated.to_be_uprated("2018-01-01") == 8


def test_parameter_uprating_with_cadence():
    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2002-04-01",
                            "start": "2000-10-01",
                            "end": "2001-10-01",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 2,
                    "2015-12-01": 1,
                    "2016-10-01": 4,
                    "2016-12-01": 1,
                    "2017-10-01": 6,
                    "2017-12-01": 1,
                    "2018-10-01": 3,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    # Interpolate halfway

    assert uprated.to_be_uprated("2018-04-01") == 6
    assert uprated.to_be_uprated("2019-04-01") == 3


def test_parameter_uprating_two_year_cadence():
    """
    Test that cadence-based uprating can handle two-year
    cadence, assessed every year
    """

    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2003-04-01",
                            "start": "2000-10-01",
                            "end": "2002-10-01",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 1,
                    "2015-12-01": 1,
                    "2016-10-01": 2,
                    "2016-12-01": 1,
                    "2017-10-01": 4,
                    "2017-12-01": 1,
                    "2018-10-01": 8,
                    "2019-10-01": 16,
                    "2020-10-01": 32,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    assert uprated.to_be_uprated("2018-04-01") == 16
    assert uprated.to_be_uprated("2019-04-01") == 64
    assert uprated.to_be_uprated("2020-04-01") == 256
    assert uprated.to_be_uprated("2021-04-01") == 1024


def test_parameter_uprating_two_year_offset():
    """
    Test that cadence-based uprating can handle a
    cadence, assessed every year, with a two-year offset
    """

    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2003-04-01",
                            "start": "2000-10-01",
                            "end": "2001-10-01",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 1,
                    "2015-12-01": 1,
                    "2016-10-01": 2,
                    "2016-12-01": 1,
                    "2017-10-01": 4,
                    "2017-12-01": 1,
                    "2018-10-01": 8,
                    "2019-10-01": 16,
                    "2020-10-01": 32,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    assert uprated.to_be_uprated("2018-04-01") == 8
    assert uprated.to_be_uprated("2019-04-01") == 16
    assert uprated.to_be_uprated("2020-04-01") == 32
    assert uprated.to_be_uprated("2021-04-01") == 64


def test_parameter_uprating_cadence_custom_effective():
    """
    Test custom effective date for uprating
    """

    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2002-04-01",
                            "start": "2000-10-01",
                            "end": "2001-10-01",
                            "effective": "2020-04-01",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 1,
                    "2015-12-01": 1,
                    "2016-10-01": 2,
                    "2016-12-01": 1,
                    "2017-10-01": 4,
                    "2017-12-01": 1,
                    "2018-10-01": 8,
                    "2019-10-01": 16,
                    "2020-10-01": 32,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    assert uprated.to_be_uprated("2018-04-01") == 4
    assert uprated.to_be_uprated("2019-04-01") == 4
    assert uprated.to_be_uprated("2020-04-01") == 8
    assert uprated.to_be_uprated("2021-04-01") == 16


def test_parameter_uprating_cadence_custom_interval():
    """
    Test custom uprating interval
    """

    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2002-04-01",
                            "start": "2000-10-01",
                            "end": "2001-10-01",
                            "interval": "month",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 1,
                    "2016-10-01": 2,
                    "2017-10-01": 4,
                    "2018-10-01": 8,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    assert uprated.to_be_uprated("2017-05-01") == 8
    assert uprated.to_be_uprated("2017-06-01") == 16
    assert uprated.to_be_uprated("2017-07-01") == 32
    assert uprated.to_be_uprated("2017-08-01") == 64
    assert uprated.to_be_uprated("2017-09-01") == 128
    assert uprated.to_be_uprated("2017-10-01") == 256
    assert uprated.to_be_uprated("2017-11-01") == 512
    assert uprated.to_be_uprated("2017-12-01") == 1024
    assert uprated.to_be_uprated("2018-01-01") == 2048


def test_parameter_uprating_custom_cadence_tight():
    """
    Test custom monthly uprating when applied within a
    tighter timeframe
    """
    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {
                    "2015-01-01": 1,
                    "2015-02-01": 2,
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2000-03-01",
                            "start": "2000-01-01",
                            "end": "2000-02-01",
                            "interval": "month",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2014-01-01": 1,
                    "2015-01-01": 1,
                    "2015-02-01": 2,
                    "2015-03-01": 4,
                    "2015-04-01": 8,
                    "2015-05-01": 16,
                    "2015-06-01": 32,
                    "2015-07-01": 64,
                    "2015-08-01": 128,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    assert uprated.to_be_uprated("2015-01-01") == 1
    assert uprated.to_be_uprated("2015-04-01") == 8
    assert uprated.to_be_uprated("2015-07-01") == 64
    assert uprated.to_be_uprated("2016-04-01") == 256


def test_parameter_uprating_cadence_custom_effective_malformed():
    """
    Test that malformed custom effective date for uprating raises error
    """

    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2002-04-01",
                            "start": "2000-10-01",
                            "end": "2001-10-01",
                            "effective": "dworkin",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 1,
                    "2015-12-01": 1,
                    "2016-10-01": 2,
                    "2016-12-01": 1,
                    "2017-10-01": 4,
                    "2017-12-01": 1,
                    "2018-10-01": 8,
                    "2019-10-01": 16,
                    "2020-10-01": 32,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    with pytest.raises(SyntaxError):
        uprated = uprate_parameters(root)


def test_parameter_uprating_cadence_date_malformed():
    """
    Test that malformed cadence start/end date for uprating raises error
    """

    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2002-04-01",
                            "start": "2000-10",
                            "end": "2001-10-01",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 1,
                    "2015-12-01": 1,
                    "2016-10-01": 2,
                    "2016-12-01": 1,
                    "2017-10-01": 4,
                    "2017-12-01": 1,
                    "2018-10-01": 8,
                    "2019-10-01": 16,
                    "2020-10-01": 32,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    with pytest.raises(SyntaxError):
        uprated = uprate_parameters(root)


def test_parameter_uprating_cadence_interval_malformed():
    """
    Test that malformed uprating cadence interval raises error
    """

    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2002-04-01",
                            "start": "2000-10-01",
                            "end": "2001-10-01",
                            "interval": "dworkin",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 1,
                    "2015-12-01": 1,
                    "2016-10-01": 2,
                    "2016-12-01": 1,
                    "2017-10-01": 4,
                    "2017-12-01": 1,
                    "2018-10-01": 8,
                    "2019-10-01": 16,
                    "2020-10-01": 32,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    with pytest.raises(SyntaxError):
        uprated = uprate_parameters(root)


def test_parameter_uprating_missing_data():
    """
    Test that, if missing a cadence start value,
    an error is properly raised; this test should raise
    a SyntaxError, as the value at uprating start (2016-10-01)
    is not present within the uprater
    """
    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2002-04-01",
                            "start": "2000-10-01",
                            "end": "2001-10-01",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2017-10-01": 4,
                    "2017-12-01": 1,
                    "2018-10-01": 8,
                    "2019-10-01": 16,
                    "2020-10-01": 32,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    with pytest.raises(ValueError):
        uprated = uprate_parameters(root)


def test_parameter_uprating_with_cadence_malformed_syntax():
    """
    Ensure that uprating properly fails if cadence options are malformed
    """
    from policyengine_core.parameters import ParameterNode

    # Create the parameter

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter",
                "values": {"2015-04-01": 1, "2016-04-01": 2, "2017-04-01": 4},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "2002-04-01",
                            "start": "2000-10-01",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2015-10-01": 2,
                    "2015-12-01": 1,
                    "2016-10-01": 4,
                    "2016-12-01": 1,
                    "2017-10-01": 6,
                    "2017-12-01": 1,
                    "2018-10-01": 3,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    with pytest.raises(SyntaxError):
        uprated = uprate_parameters(root)


def test_parameter_uprating_cadence_minimal_data():
    """
    Test that cadence-based uprating works when:
    1. There are only two uprating values
    2. There is only one value given against which to uprate
    """
    from policyengine_core.parameters import ParameterNode

    root = ParameterNode(
        data={
            "to_be_uprated": {
                "description": "Example parameter based off UK CPI uprating",
                "values": {"2023-01-01": 1},
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "start": "2000-09-01",
                            "end": "2001-09-01",
                            "enactment": "2002-04-01",
                        },
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2021-09-01": 112.4,
                    "2022-09-01": 123.8,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    assert round(uprated.to_be_uprated("2023-04-01"), 3) == 1.101
