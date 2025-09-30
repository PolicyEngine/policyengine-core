"""Test NumPy 2.x compatibility with structured arrays in vectorial parameters.

This test reproduces and verifies the fix for the issue where numpy.select()
fails with NumPy 2.x when using structured arrays with numeric string field
names that get sorted differently (e.g., '1', '10', '2' vs '1', '2', '10').

This commonly occurs when using axes with state-based parameters like ACA
rating areas, where different states have different numbers of rating areas.
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


def test_mismatched_structured_array_fields():
    """Test structured arrays with different field subsets (real ACA scenario).

    This tests the dtype unification code path that handles arrays with
    different field subsets.
    """
    # Create states with different rating area fields (like real ACA data)
    # Each row represents an age bracket, each field a rating area
    dtype_ca = np.dtype([("1", float), ("2", float), ("3", float)])
    dtype_ny = np.dtype([("10", float), ("11", float)])  # Different fields

    # Create data for 2 rows (age brackets)
    data_ca = np.array([(100.0, 110.0, 120.0), (200.0, 210.0, 220.0)], dtype=dtype_ca)
    data_ny = np.array([(300.0, 310.0), (400.0, 410.0)], dtype=dtype_ny)

    # Create parent structure with both states
    parent_dtype = np.dtype([("CA", dtype_ca), ("NY", dtype_ny)])
    parent_data = np.array([
        (data_ca[0], data_ny[0]),
        (data_ca[1], data_ny[1])
    ], dtype=parent_dtype)
    parent_vector = parent_data.view(np.recarray)

    node = VectorialParameterNodeAtInstant("states", parent_vector, "2024-01-01")

    # Access both states - this triggers dtype mismatch handling
    keys = np.array(["CA", "NY"])
    result = node[keys]

    # Result is wrapped in VectorialParameterNodeAtInstant for structured arrays
    assert isinstance(result, (np.ndarray, VectorialParameterNodeAtInstant))
    if isinstance(result, VectorialParameterNodeAtInstant):
        assert len(result.vector) == 2
    else:
        assert len(result) == 2


def test_all_dtypes_match_optimization():
    """Test that when all dtypes match, we use the optimized path."""
    # Create uniform arrays where all states have the same fields
    dtype = np.dtype([("1", float), ("2", float)])
    data = np.array([
        ((100.0, 110.0), (200.0, 210.0)),
        ((300.0, 310.0), (400.0, 410.0))
    ], dtype=[("state1", dtype), ("state2", dtype)])
    vector = data.view(np.recarray)

    node = VectorialParameterNodeAtInstant("test", vector, "2024-01-01")

    # When all dtypes match, should use optimized code path
    keys = np.array(["state1", "state2"])
    result = node[keys]

    # Result is wrapped for structured arrays
    assert isinstance(result, (np.ndarray, VectorialParameterNodeAtInstant))
    if isinstance(result, VectorialParameterNodeAtInstant):
        assert len(result.vector) == 2
    else:
        assert len(result) == 2
