"""Test the between function from commons.formulas."""

import numpy as np
import pytest
from policyengine_core.commons.formulas import between


class TestBetweenFunction:
    """Test the between function for checking if values are within bounds."""

    def test_between_inclusive_both(self):
        """Test between with both bounds inclusive (default)."""
        values = np.array([1, 2, 3, 4, 5])
        result = between(values, 2, 4)
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_between_inclusive_left(self):
        """Test between with only left bound inclusive."""
        values = np.array([1, 2, 3, 4, 5])
        result = between(values, 2, 4, inclusive="left")
        expected = np.array([False, True, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_between_inclusive_right(self):
        """Test between with only right bound inclusive."""
        values = np.array([1, 2, 3, 4, 5])
        result = between(values, 2, 4, inclusive="right")
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_between_inclusive_neither(self):
        """Test between with neither bound inclusive."""
        values = np.array([1, 2, 3, 4, 5])
        result = between(values, 2, 4, inclusive="neither")
        expected = np.array([False, False, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_between_with_floats(self):
        """Test between with float values."""
        values = np.array([1.5, 2.5, 3.5, 4.5])
        result = between(values, 2.0, 4.0)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_between_with_negative_values(self):
        """Test between with negative values."""
        values = np.array([-3, -2, -1, 0, 1, 2, 3])
        result = between(values, -2, 2)
        expected = np.array([False, True, True, True, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_between_single_value(self):
        """Test between with a single value."""
        value = 5
        assert between(value, 0, 10).item() == True
        assert between(value, 0, 4).item() == False
        assert between(value, 5, 10).item() == True
        assert between(value, 5, 10, inclusive="left").item() == True
        assert between(value, 5, 10, inclusive="neither").item() == False

    def test_between_edge_cases(self):
        """Test between with edge cases."""
        # Empty array
        values = np.array([])
        result = between(values, 0, 10)
        assert len(result) == 0

        # All values equal to bounds
        values = np.array([5, 5, 5])
        result = between(values, 5, 5)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

        # Bounds in reverse order (upper < lower)
        values = np.array([1, 2, 3, 4, 5])
        result = between(values, 4, 2)  # This should return all False
        expected = np.array([False, False, False, False, False])
        np.testing.assert_array_equal(result, expected)
