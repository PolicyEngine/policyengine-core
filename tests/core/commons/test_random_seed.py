"""Test the random function with large entity IDs to ensure no overflow."""

import numpy as np
import pytest
from unittest.mock import Mock
from policyengine_core.commons.formulas import random


class TestRandomSeed:
    """Test random seed handling to prevent NumPy overflow errors."""

    def test_random_with_large_entity_ids(self):
        """Test that random() handles large entity IDs without overflow."""
        # Create a mock population with simulation
        population = Mock()
        population.simulation = Mock()
        population.simulation.count_random_calls = 0
        population.entity = Mock()
        population.entity.key = "person"
        
        # Mock the get_holder and get_known_periods
        holder = Mock()
        holder.get_known_periods.return_value = []
        population.simulation.get_holder.return_value = holder
        population.simulation.default_calculation_period = Mock()
        
        # Test with very large entity IDs that would cause overflow
        # if not handled properly
        large_ids = np.array([
            np.iinfo(np.int64).max - 1000,  # Very large positive ID
            np.iinfo(np.int64).max // 2,    # Large positive ID
            1234567890123456789,             # Another large ID
        ])
        
        # Mock the population call to return large IDs
        population.side_effect = lambda key, period: large_ids
        
        # This should not raise a ValueError about negative seeds
        result = random(population)
        
        # Check that we got valid random values
        assert isinstance(result, np.ndarray)
        assert len(result) == len(large_ids)
        assert all(0 <= val <= 1 for val in result)
        
    def test_random_seed_consistency(self):
        """Test that random() produces consistent results for same inputs."""
        # Create mock population
        population = Mock()
        population.simulation = Mock()
        population.simulation.count_random_calls = 0
        population.entity = Mock()
        population.entity.key = "household"
        
        holder = Mock()
        holder.get_known_periods.return_value = []
        population.simulation.get_holder.return_value = holder
        population.simulation.default_calculation_period = Mock()
        
        # Use same IDs
        ids = np.array([1, 2, 3])
        population.side_effect = lambda key, period: ids
        
        # First call
        result1 = random(population)
        
        # Reset count to simulate same conditions
        population.simulation.count_random_calls = 0
        
        # Second call with same conditions
        result2 = random(population)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        
    def test_random_increments_call_count(self):
        """Test that random() increments the call counter."""
        population = Mock()
        population.simulation = Mock()
        population.simulation.count_random_calls = 0
        population.entity = Mock()
        population.entity.key = "person"
        
        holder = Mock()
        holder.get_known_periods.return_value = []
        population.simulation.get_holder.return_value = holder
        population.simulation.default_calculation_period = Mock()
        
        ids = np.array([1, 2, 3])
        population.side_effect = lambda key, period: ids
        
        # First call
        random(population)
        assert population.simulation.count_random_calls == 1
        
        # Second call
        random(population)
        assert population.simulation.count_random_calls == 2
        
    def test_random_handles_negative_ids(self):
        """Test that random() handles negative IDs properly."""
        population = Mock()
        population.simulation = Mock()
        population.simulation.count_random_calls = 0
        population.entity = Mock()
        population.entity.key = "person"
        
        holder = Mock()
        holder.get_known_periods.return_value = []
        population.simulation.get_holder.return_value = holder
        population.simulation.default_calculation_period = Mock()
        
        # Include negative IDs
        ids = np.array([-100, -1, 0, 1, 100])
        population.side_effect = lambda key, period: ids
        
        # Should handle negative IDs without errors
        result = random(population)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(ids)
        assert all(0 <= val <= 1 for val in result)
        
    def test_no_negative_seed_error_with_overflow(self):
        """Test that seed calculation overflow doesn't cause negative seed error."""
        population = Mock()
        population.simulation = Mock()
        population.simulation.count_random_calls = 999999999  # Large count
        population.entity = Mock()
        population.entity.key = "person"
        
        holder = Mock()
        holder.get_known_periods.return_value = []
        population.simulation.get_holder.return_value = holder
        population.simulation.default_calculation_period = Mock()
        
        # Use the exact ID that would cause overflow in old implementation
        # This ID when multiplied by 100 and added to count_random_calls
        # would overflow int64 and become negative
        overflow_id = np.array([np.iinfo(np.int64).max // 100])
        population.side_effect = lambda key, period: overflow_id
        
        # In the old implementation, this would raise:
        # ValueError: Seed must be between 0 and 2**32 - 1
        # With the fix using abs(), it should work fine
        result = random(population)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert 0 <= result[0] <= 1