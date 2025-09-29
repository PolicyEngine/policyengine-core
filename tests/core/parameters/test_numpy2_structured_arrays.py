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
    """
    # Create a structured array with numeric string field names like ACA rating areas
    # This mimics the structure of ACA state_rating_area_cost parameters
    dtype = np.dtype([
        ('1', float), ('2', float), ('3', float), ('4', float), ('5', float),
        ('10', float), ('11', float), ('12', float)
    ])

    # Create test data - this simulates parameter data with multiple rating areas
    data = np.array([
        (100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0),
        (200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0),
    ], dtype=dtype)

    # Create a vectorial parameter node (this simulates how PolicyEngine stores parameters)
    vector = data.view(np.recarray)
    instant_str = "2024-01-01"
    name = "test_rating_areas"

    node = VectorialParameterNodeAtInstant(name, vector, instant_str)

    # Test indexing with a key array (simulates axes creating multiple values)
    # This is what fails in the real scenario - when axes creates multiple
    # households with different rating areas
    keys = np.array(['1', '2', '10'])

    # This should work but fails with NumPy 2.x due to dtype field ordering
    result = node[keys]

    # Verify results
    assert len(result) == 3
    # First household gets rating area '1' values
    assert result[0] == pytest.approx(100.0) or result[0] == pytest.approx(200.0)
    # Second household gets rating area '2' values
    assert result[1] == pytest.approx(110.0) or result[1] == pytest.approx(210.0)
    # Third household gets rating area '10' values
    assert result[2] == pytest.approx(150.0) or result[2] == pytest.approx(250.0)


def test_structured_array_field_order_preservation():
    """Test that field order is preserved when creating default arrays.

    NumPy 2.x is stricter about dtype matching - field order must be identical.
    """
    # Create array with fields in non-alphabetical order
    dtype = np.dtype([('z', float), ('a', float), ('m', float)])
    data = np.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=dtype)
    vector = data.view(np.recarray)

    node = VectorialParameterNodeAtInstant("test", vector, "2024-01-01")

    # Test with array key
    keys = np.array(['z', 'm', 'a'])
    result = node[keys]

    # Should return correct values
    assert len(result) == 3
    assert result[0] in [1.0, 4.0]  # 'z' field
    assert result[1] in [3.0, 6.0]  # 'm' field
    assert result[2] in [2.0, 5.0]  # 'a' field