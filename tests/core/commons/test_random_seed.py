"""Test the random function with name-based salting."""

import numpy as np
import pytest
from unittest.mock import Mock
from policyengine_core.commons.formulas import random, _stable_string_hash


def create_mock_population(entity_key="person", variable_name="test_variable"):
    """Helper to create a mock population with tracer stack."""
    population = Mock()

    # Create simulation as a simple object to avoid Mock's hasattr behavior
    class MockSimulation:
        pass

    simulation = MockSimulation()
    simulation.tracer = Mock()
    simulation.tracer.stack = [
        {"name": variable_name, "period": "2024", "branch_name": "default"}
    ]

    holder = Mock()
    holder.get_known_periods.return_value = []
    simulation.get_holder = Mock(return_value=holder)
    simulation.default_calculation_period = Mock()

    population.simulation = simulation
    population.entity = Mock()
    population.entity.key = entity_key

    return population


class TestStableStringHash:
    """Test the stable string hash function."""

    def test_hash_consistency(self):
        """Test that hash produces consistent results."""
        s = "test_variable:1:"
        h1 = _stable_string_hash(s)
        h2 = _stable_string_hash(s)
        assert h1 == h2

    def test_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        h1 = _stable_string_hash("variable_a:1:")
        h2 = _stable_string_hash("variable_b:1:")
        assert h1 != h2

    def test_hash_returns_uint64(self):
        """Test that hash returns numpy uint64."""
        h = _stable_string_hash("test")
        assert isinstance(h, np.uint64)


class TestRandomSeed:
    """Test random seed handling with name-based salting."""

    def test_random_with_large_entity_ids(self):
        """Test that random() handles large entity IDs without overflow."""
        population = create_mock_population()
        large_ids = np.array(
            [
                np.iinfo(np.int64).max - 1000,
                np.iinfo(np.int64).max // 2,
                1234567890123456789,
            ]
        )
        population.side_effect = lambda key, period: large_ids

        result = random(population)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(large_ids)
        assert all(0 <= val <= 1 for val in result)

    def test_random_seed_consistency(self):
        """Test that random() produces consistent results for same inputs."""
        population = create_mock_population(entity_key="household")
        ids = np.array([1, 2, 3])
        population.side_effect = lambda key, period: ids

        result1 = random(population)

        # Reset call counts to simulate same conditions
        population.simulation.random_call_counts = {}

        result2 = random(population)

        np.testing.assert_array_equal(result1, result2)

    def test_random_increments_per_variable_counter(self):
        """Test that random() increments the per-variable call counter."""
        population = create_mock_population()
        ids = np.array([1, 2, 3])
        population.side_effect = lambda key, period: ids

        random(population)
        assert population.simulation.random_call_counts["test_variable:"] == 1

        random(population)
        assert population.simulation.random_call_counts["test_variable:"] == 2

    def test_random_handles_negative_ids(self):
        """Test that random() handles negative IDs properly."""
        population = create_mock_population()
        ids = np.array([-100, -1, 0, 1, 100])
        population.side_effect = lambda key, period: ids

        result = random(population)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(ids)
        assert all(0 <= val <= 1 for val in result)

    def test_random_order_independence(self):
        """Test that variable order doesn't affect random values."""
        ids = np.array([1, 2, 3])

        # Simulate calling variable_a first, then variable_b
        pop_a = create_mock_population(variable_name="variable_a")
        pop_a.side_effect = lambda key, period: ids
        result_a_first = random(pop_a)

        pop_b = create_mock_population(variable_name="variable_b")
        pop_b.side_effect = lambda key, period: ids
        result_b_after_a = random(pop_b)

        # Now simulate calling variable_b first, then variable_a
        pop_b2 = create_mock_population(variable_name="variable_b")
        pop_b2.side_effect = lambda key, period: ids
        result_b_first = random(pop_b2)

        pop_a2 = create_mock_population(variable_name="variable_a")
        pop_a2.side_effect = lambda key, period: ids
        result_a_after_b = random(pop_a2)

        # Same variable should produce same results regardless of order
        np.testing.assert_array_equal(result_a_first, result_a_after_b)
        np.testing.assert_array_equal(result_b_after_a, result_b_first)

    def test_random_multiple_calls_same_formula(self):
        """Test that multiple calls within same formula produce different values."""
        population = create_mock_population()
        ids = np.array([1, 2, 3])
        population.side_effect = lambda key, period: ids

        result1 = random(population)
        result2 = random(population)
        result3 = random(population)

        # Each call should produce different values
        assert not np.array_equal(result1, result2)
        assert not np.array_equal(result2, result3)
        assert not np.array_equal(result1, result3)

    def test_random_with_salt(self):
        """Test that salt parameter creates distinct random streams."""
        population = create_mock_population()
        ids = np.array([1, 2, 3])
        population.side_effect = lambda key, period: ids

        result_no_salt = random(population)

        # Reset counter
        population.simulation.random_call_counts = {}

        result_with_salt = random(population, salt="custom_salt")

        # Different salts should produce different values
        assert not np.array_equal(result_no_salt, result_with_salt)

    def test_random_outside_formula_raises(self):
        """Test that calling random() outside formula context without salt raises."""
        population = create_mock_population()
        population.simulation.tracer.stack = []  # Empty stack
        ids = np.array([1, 2, 3])
        population.side_effect = lambda key, period: ids

        with pytest.raises(
            ValueError, match="random\\(\\) called outside formula context"
        ):
            random(population)

    def test_random_outside_formula_with_salt(self):
        """Test that salt allows random() to work outside formula context."""
        population = create_mock_population()
        population.simulation.tracer.stack = []  # Empty stack
        ids = np.array([1, 2, 3])
        population.side_effect = lambda key, period: ids

        result = random(population, salt="explicit_salt")

        assert isinstance(result, np.ndarray)
        assert len(result) == len(ids)
        assert all(0 <= val <= 1 for val in result)

    def test_different_variables_produce_different_values(self):
        """Test that different variable names produce different random values."""
        ids = np.array([1, 2, 3])

        pop_a = create_mock_population(variable_name="variable_a")
        pop_a.side_effect = lambda key, period: ids
        result_a = random(pop_a)

        pop_b = create_mock_population(variable_name="variable_b")
        pop_b.side_effect = lambda key, period: ids
        result_b = random(pop_b)

        # Different variables should produce different values
        assert not np.array_equal(result_a, result_b)

    def test_same_entity_same_variable_same_value(self):
        """Test reproducibility: same entity + variable = same random value."""
        ids = np.array([42])

        pop1 = create_mock_population(variable_name="my_variable")
        pop1.side_effect = lambda key, period: ids
        result1 = random(pop1)

        pop2 = create_mock_population(variable_name="my_variable")
        pop2.side_effect = lambda key, period: ids
        result2 = random(pop2)

        np.testing.assert_array_equal(result1, result2)
