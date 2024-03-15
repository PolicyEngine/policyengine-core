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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0002-04-01",
                            "start": "0000-10-01",
                            "end": "0001-10-01"
                        }
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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0003-04-01",
                            "start": "0000-10-01",
                            "end": "0002-10-01"
                        }
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
                    "2020-10-01": 32
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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0003-04-01",
                            "start": "0000-10-01",
                            "end": "0001-10-01"
                        }
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
                    "2020-10-01": 32
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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0002-04-01",
                            "start": "0000-10-01",
                            "end": "0001-10-01",
                            "effective": "2020-04-01"
                        }
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
                    "2020-10-01": 32
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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0002-04-01",
                            "start": "0000-10-01",
                            "end": "0001-10-01",
                            "interval": "month"
                        }
                    },
                },
            },
            "uprater": {
                "description": "Uprater",
                "values": {
                    "2016-10-01": 2,
                    "2016-11-01": 4,
                    "2016-12-01": 8,
                    "2017-01-01": 16,
                    "2017-10-01": 4,
                    "2017-11-01": 8,
                    "2017-12-01": 16,
                    "2018-01-01": 32,
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    uprated = uprate_parameters(root)

    assert uprated.to_be_uprated("2018-04-01") == 8
    assert uprated.to_be_uprated("2018-05-01") == 16
    assert uprated.to_be_uprated("2018-06-01") == 32
    assert uprated.to_be_uprated("2018-07-01") == 64

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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0002-04-01",
                            "start": "0000-10-01",
                            "end": "0001-10-01",
                            "effective": "dworkin"
                        }
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
                    "2020-10-01": 32
                },
            },
        }
    )

    from policyengine_core.parameters import uprate_parameters

    with pytest.raises(ValueError):
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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0002-04-01",
                            "start": "0000-10",
                            "end": "0001-10-01",
                        }
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
                    "2020-10-01": 32
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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0002-04-01",
                            "start": "0000-10-01",
                            "end": "0001-10-01",
                            "interval": "dworkin"
                        }
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
                    "2020-10-01": 32
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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0002-04-01",
                            "start": "0000-10-01",
                            "end": "0001-10-01"
                        }
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
                    "2020-10-01": 32
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
                "values": {
                    "2015-04-01": 1,
                    "2016-04-01": 2,
                    "2017-04-01": 4
                },
                "metadata": {
                    "uprating": {
                        "parameter": "uprater",
                        "at_defined_interval": {
                            "enactment": "0002-04-01",
                            "start": "0000-10-01",
                        }
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