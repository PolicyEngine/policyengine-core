"""Test NumPy 2.x compatibility with structured arrays in vectorial parameters.

This test reproduces the issue where numpy.select() fails with NumPy 2.x
when using structured arrays with numeric string field names that get
sorted differently (e.g., '1', '10', '2' vs '1', '2', '10').
"""

import numpy as np
import pytest
from policyengine_core.parameters import ParameterNode
from policyengine_core.parameters.vectorial_parameter_node_at_instant import (
    VectorialParameterNodeAtInstant,
)


def test_structured_array_with_numeric_string_fields():
    """Test that vectorial parameters work with NumPy 2.x structured arrays.

    This reproduces the issue where field names like '1', '2', ..., '10', '11'
    get sorted alphabetically by NumPy ('1', '10', '11', '2', ...) causing
    dtype mismatches in numpy.select().

    The real ACA rating_area parameters have 67 rating areas for some states,
    which triggers alphabetic vs numeric sorting issues.
    """
    # Create a structured array with ALL numeric string field names 1-67
    # This mimics the actual ACA state_rating_area_cost structure
    field_names = [str(i) for i in range(1, 68)]  # '1' through '67'
    dtype = np.dtype([(name, float) for name in field_names])

    # Create test data with 3 rows (simulates 3 age brackets or similar)
    test_values = []
    for row in range(3):
        row_data = tuple(float(i * 10 + row) for i in range(1, 68))
        test_values.append(row_data)
    data = np.array(test_values, dtype=dtype)

    # Create a vectorial parameter node
    vector = data.view(np.recarray)
    instant_str = "2024-01-01"
    name = "test_rating_areas"

    node = VectorialParameterNodeAtInstant(name, vector, instant_str)

    # Test with keys that will trigger alphabetic sorting issues
    # When you have '1', '2', '10', NumPy might sort them as '1', '10', '2'
    keys = np.array(["1", "2", "10"])

    # This should work but may fail with NumPy 2.x due to dtype field ordering
    result = node[keys]

    # Verify results
    assert len(result) == 3
    assert result[0] == pytest.approx(10.0)  # Row 0, field '1'
    assert result[1] == pytest.approx(21.0)  # Row 1, field '2'
    assert result[2] == pytest.approx(102.0)  # Row 2, field '10'


def test_structured_array_field_order_preservation():
    """Test that field order is preserved when creating default arrays.

    NumPy 2.x is stricter about dtype matching - field order must be identical.
    """
    # Create array with fields in non-alphabetical order
    dtype = np.dtype([("z", float), ("a", float), ("m", float)])
    data = np.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=dtype)
    vector = data.view(np.recarray)

    node = VectorialParameterNodeAtInstant("test", vector, "2024-01-01")

    # Test with array key - must match vector length
    keys = np.array(["z", "m"])
    result = node[keys]

    # Should return correct values
    assert len(result) == 2
    assert result[0] == pytest.approx(1.0)  # Row 0, field 'z'
    assert result[1] == pytest.approx(6.0)  # Row 1, field 'm'
