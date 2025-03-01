import pytest
from policyengine_core.parameters import ParameterNode
from policyengine_core.parameters.operations import get_parameter
from policyengine_core.errors import ParameterPathError
from policyengine_core.parameters.operations.get_parameter import (
    _find_similar_parameters,
    _navigate_to_node,
    _access_indexed_value as _access_bracket,
    _handle_bracket_access,
)


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
                },
            }
        }
    )

    # Test parameter not found
    with pytest.raises(ParameterPathError) as excinfo:
        get_parameter(parameters, "tax.property_tax.rate")

    # Ensure the error message contains useful information
    error_message = str(excinfo.value)
    assert "property_tax" in error_message
    assert "not found" in error_message
    assert excinfo.value.parameter_path == "tax.property_tax.rate"
    assert excinfo.value.failed_at == "property_tax"

    # Test parameter not found but with similar names
    with pytest.raises(ParameterPathError) as excinfo:
        get_parameter(parameters, "tax.income")

    # Check that the suggestion for "income" includes "income_tax"
    error_message = str(excinfo.value)
    assert "income" in error_message
    assert "Did you mean" in error_message
    assert "income_tax" in error_message
    assert excinfo.value.parameter_path == "tax.income"
    assert excinfo.value.failed_at == "income"


def test_invalid_bracket_syntax_message():
    """Test error handling with invalid bracket syntax."""
    parameters = ParameterNode(
        data={
            "tax": {
                "brackets": [
                    {
                        "threshold": {"values": {"2022-01-01": 0}},
                        "rate": {"values": {"2022-01-01": 0.1}},
                    },
                ],
            }
        }
    )

    # Test invalid bracket syntax (missing closing bracket)
    with pytest.raises(ParameterPathError) as excinfo:
        get_parameter(parameters, "tax.brackets[0.rate")

    assert "Invalid bracket syntax" in str(excinfo.value)
    assert excinfo.value.parameter_path == "tax.brackets[0.rate"
    # Don't assert the exact failed_at value since it depends on implementation details


def test_bracket_on_non_bracket_parameter():
    """Test error when trying to use bracket notation on a non-bracket parameter."""
    parameters = ParameterNode(
        data={"tax": {"simple_rate": {"values": {"2022-01-01": 0.2}}}}
    )

    with pytest.raises(ParameterPathError) as excinfo:
        get_parameter(parameters, "tax.simple_rate[0]")

    assert "does not support bracket indexing" in str(excinfo.value)
    assert excinfo.value.parameter_path == "tax.simple_rate[0]"
    # Don't assert the exact failed_at value since it depends on implementation details


def test_parse_path_component_with_invalid_bracket():
    """Test error when parsing invalid bracket syntax in a path component."""
    from policyengine_core.parameters.operations.get_parameter import (
        _parse_path_component,
    )

    # Test parsing with invalid bracket syntax (missing closing bracket)
    with pytest.raises(ParameterPathError) as excinfo:
        _parse_path_component("brackets[0", "tax.brackets[0")

    assert "Invalid bracket syntax" in str(excinfo.value)
    assert "Use format: parameter_name[index]" in str(excinfo.value)
    assert excinfo.value.parameter_path == "tax.brackets[0"
    assert "brackets[0" in str(excinfo.value.failed_at)


def test_parse_path_component_with_non_integer():
    """Test error when parsing a non-integer bracket index."""
    parameters = ParameterNode(
        data={
            "tax": {
                "brackets": [
                    {
                        "threshold": {"values": {"2022-01-01": 0}},
                        "rate": {"values": {"2022-01-01": 0.1}},
                    },
                ],
            }
        }
    )

    # Test with invalid bracket index
    with pytest.raises(ParameterPathError) as excinfo:
        get_parameter(parameters, "tax.brackets[abc]")

    # The error should be about invalid syntax rather than invalid index
    # This is because the parser detects a syntax issue before trying to convert to int
    assert "Invalid bracket syntax" in str(excinfo.value)
    assert excinfo.value.parameter_path == "tax.brackets[abc]"


def test_parse_path_component_with_multiple_brackets():
    """Test error when parsing a path component with multiple opening brackets."""
    from policyengine_core.parameters.operations.get_parameter import (
        _parse_path_component,
    )

    # Test parsing with multiple brackets
    with pytest.raises(ParameterPathError) as excinfo:
        _parse_path_component("brackets[0][1]", "tax.brackets[0][1]")

    assert "Invalid bracket syntax" in str(excinfo.value)
    assert "Use format: parameter_name[index]" in str(excinfo.value)
    assert excinfo.value.parameter_path == "tax.brackets[0][1]"
    assert excinfo.value.failed_at == "brackets[0][1]"


def test_access_child_of_parameter():
    """Test error when trying to access a child of a parameter (not a node)."""
    parameters = ParameterNode(
        data={
            "tax": {
                "simple_rate": {"values": {"2022-01-01": 0.2}},
            }
        }
    )

    # Try to access a child of a parameter
    with pytest.raises(ParameterPathError) as excinfo:
        get_parameter(parameters, "tax.simple_rate.child")

    assert "Cannot access 'child'" in str(excinfo.value)
    assert "not a parameter node with children" in str(excinfo.value)
    assert excinfo.value.parameter_path == "tax.simple_rate.child"
    assert excinfo.value.failed_at == "child"


def test_multiple_brackets_in_path_component():
    """Test error handling with multiple brackets in a path component."""
    parameters = ParameterNode(
        data={
            "tax": {
                "brackets": [
                    {
                        "threshold": {"values": {"2022-01-01": 0}},
                        "rate": {"values": {"2022-01-01": 0.1}},
                    },
                ],
            }
        }
    )

    # Test path with multiple bracket components
    with pytest.raises(ParameterPathError) as excinfo:
        get_parameter(parameters, "tax.brackets[0][1]")

    assert "Invalid bracket syntax" in str(excinfo.value)
    assert excinfo.value.parameter_path == "tax.brackets[0][1]"
    assert excinfo.value.failed_at == "brackets[0][1]"


def test_find_similar_parameters():
    """Test the _find_similar_parameters helper function."""
    # Create a node with some children
    parameters = ParameterNode(
        data={
            "tax": {
                "income_tax": {},
                "property_tax": {},
                "sales_tax": {},
                "inheritance_tax": {},
            }
        }
    )

    # Get the "tax" node
    tax_node = parameters.children["tax"]

    # Test finding similar parameters
    similar = _find_similar_parameters(tax_node, "tax")
    assert "income_tax" in similar
    assert "property_tax" in similar
    assert "sales_tax" in similar
    assert "inheritance_tax" in similar

    # Test with case insensitivity
    similar = _find_similar_parameters(tax_node, "TAX")
    assert "income_tax" in similar
    assert "property_tax" in similar

    # Test with partial match
    similar = _find_similar_parameters(tax_node, "inc")
    assert "income_tax" in similar
    assert "inheritance_tax" not in similar

    # Test with no matches
    similar = _find_similar_parameters(tax_node, "xyz")
    assert len(similar) == 0

    # Test with a non-node (no children attribute)
    parameter = get_parameter(parameters, "tax.income_tax")
    similar = _find_similar_parameters(parameter, "anything")
    assert len(similar) == 0


def test_handle_bracket_access_index_out_of_range():
    """Test index out of range in _handle_bracket_access function."""
    from policyengine_core.parameters.parameter_scale import ParameterScale
    import os

    # Create a scale with brackets
    scale = ParameterScale(
        "test.brackets",
        {
            "brackets": [
                {
                    "threshold": {"values": {"2022-01-01": 0}},
                    "rate": {"values": {"2022-01-01": 0.1}},
                },
            ]
        },
        os.path.join(os.getcwd(), "test.yaml"),
    )

    # Test accessing out-of-range bracket
    with pytest.raises(ParameterPathError) as excinfo:
        _handle_bracket_access(scale, "[5]", "test.brackets[5]")

    assert "Bracket index out of range" in str(excinfo.value)
    assert "Valid indices are 0 to 0" in str(excinfo.value)
    assert excinfo.value.parameter_path == "test.brackets[5]"
    assert excinfo.value.failed_at == "[5]"


def test_access_bracket_index_out_of_range():
    """Test index out of range in _access_bracket function."""
    from policyengine_core.parameters.parameter_scale import ParameterScale
    import os

    # Create a scale with brackets
    scale = ParameterScale(
        "test.brackets",
        {
            "brackets": [
                {
                    "threshold": {"values": {"2022-01-01": 0}},
                    "rate": {"values": {"2022-01-01": 0.1}},
                },
            ]
        },
        os.path.join(os.getcwd(), "test.yaml"),
    )

    # Test accessing out-of-range bracket
    with pytest.raises(ParameterPathError) as excinfo:
        _access_bracket(
            scale, "brackets", 5, "brackets[5]", "test.brackets[5]"
        )

    assert "Bracket index out of range" in str(excinfo.value)
    assert "Valid indices are 0 to 0" in str(excinfo.value)
    assert excinfo.value.parameter_path == "test.brackets[5]"
    assert excinfo.value.failed_at == "brackets[5]"
