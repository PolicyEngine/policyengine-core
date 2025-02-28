import pytest
from policyengine_core.parameters import ParameterNode
from policyengine_core.parameters.operations import get_parameter


def test_parameter_not_found_message():
    """Test that not found errors have better messages."""
    parameters = ParameterNode(
        data={
            "tax": {
                "income_tax": {
                    "rate": {
                        "values": {
                            "2022-01-01": 0.2,
                        }
                    },
                },
                "sales_tax": {
                    "rate": {
                        "values": {
                            "2022-01-01": 0.1,
                        }
                    },
                }
            }
        }
    )
    
    # Test parameter not found
    with pytest.raises(ValueError) as excinfo:
        get_parameter(parameters, "tax.property_tax.rate")
    
    # Ensure the error message contains useful information
    error_message = str(excinfo.value)
    assert "property_tax" in error_message
    assert "not found" in error_message
    
    # Test parameter not found but with similar names
    with pytest.raises(ValueError) as excinfo:
        get_parameter(parameters, "tax.income")
    
    # Check that the suggestion for "income" includes "income_tax"
    error_message = str(excinfo.value)
    assert "income" in error_message
    assert "Did you mean" in error_message
    assert "income_tax" in error_message


def test_invalid_bracket_syntax_message():
    """Test error handling with invalid bracket syntax."""
    parameters = ParameterNode(
        data={
            "tax": {
                "brackets": [
                    {
                        "threshold": {
                            "values": {"2022-01-01": 0}
                        },
                        "rate": {
                            "values": {"2022-01-01": 0.1}
                        }
                    },
                ],
            }
        }
    )
    
    # Test invalid bracket syntax (missing closing bracket)
    with pytest.raises(ValueError) as excinfo:
        get_parameter(parameters, "tax.brackets[0.rate")
    
    assert "Invalid bracket syntax" in str(excinfo.value)
    
    # Test invalid bracket index (not an integer)
    with pytest.raises(ValueError) as excinfo:
        get_parameter(parameters, "tax.brackets[a].rate")
    
    assert "Invalid bracket syntax" in str(excinfo.value)