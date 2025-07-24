"""Test the is_in function from commons.formulas."""

import numpy as np
import pytest
from policyengine_core.commons.formulas import is_in


class TestIsInFunction:
    """Test the is_in function for checking membership."""

    def test_is_in_basic(self):
        """Test basic is_in functionality."""
        values = np.array([1, 2, 3, 4, 5])
        result = is_in(values, 2, 4)
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_in_with_list(self):
        """Test is_in with a list of targets."""
        values = np.array([1, 2, 3, 4, 5])
        result = is_in(values, [2, 4])
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_in_with_strings(self):
        """Test is_in with string values."""
        values = np.array(["apple", "banana", "cherry", "date"])
        result = is_in(values, "banana", "date")
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_in_with_mixed_types(self):
        """Test is_in with mixed numeric types."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = is_in(values, 2, 4)  # int targets, float values
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_in_single_value(self):
        """Test is_in with a single value."""
        value = 5
        assert is_in(value, 5) == True
        assert is_in(value, 1, 2, 3, 4, 5) == True
        assert is_in(value, 1, 2, 3) == False
        assert is_in(value, [1, 2, 3, 4, 5]) == True

    def test_is_in_empty_targets(self):
        """Test is_in with empty targets."""
        values = np.array([1, 2, 3])
        result = is_in(values, [])
        expected = np.array([False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_in_empty_values(self):
        """Test is_in with empty values array."""
        values = np.array([])
        result = is_in(values, 1, 2, 3)
        assert len(result) == 0

    def test_is_in_with_none(self):
        """Test is_in with None values."""
        values = np.array([1, 2, None, 4], dtype=object)
        result = is_in(values, None)
        expected = np.array([False, False, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_in_all_match(self):
        """Test is_in where all values match."""
        values = np.array([1, 1, 1, 1])
        result = is_in(values, 1)
        expected = np.array([True, True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_in_no_match(self):
        """Test is_in where no values match."""
        values = np.array([1, 2, 3, 4])
        result = is_in(values, 5, 6, 7)
        expected = np.array([False, False, False, False])
        np.testing.assert_array_equal(result, expected)
