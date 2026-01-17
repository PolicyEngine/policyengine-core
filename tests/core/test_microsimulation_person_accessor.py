"""
Test for Python 3.14 compatibility issue #407.

The person() accessor in Microsimulation should return unweighted values
(numpy arrays), not weighted MicroSeries. This test verifies that the
isinstance check correctly identifies Microsimulation instances.

Issue #407 reports that in Python 3.14, person() returns unweighted values
when it previously returned weighted values in Python 3.13. This test
ensures consistent behavior across Python versions.
"""

import numpy as np
import pytest
from microdf import MicroSeries

from policyengine_core.country_template import Microsimulation


class TestMicrosimulationPersonAccessor:
    """Tests for person() accessor behavior in Microsimulation."""

    def test_person_accessor_returns_unweighted_in_microsimulation(self):
        """
        Verify that person() accessor returns unweighted numpy arrays.

        The person() accessor is used internally in formulas and should return
        unweighted values for performance. This is the intended behavior per
        the code comment: "Internal simulation code shouldn't use weights in
        order to avoid performance slowdowns."
        """
        sim = Microsimulation()
        result = sim.person("salary", "2022-01")

        # The result should be a numpy array, not MicroSeries
        assert isinstance(result, np.ndarray), (
            f"Expected numpy.ndarray but got {type(result).__name__}. "
            "person() should return unweighted arrays for performance."
        )

    def test_calculate_returns_weighted_microseries(self):
        """
        Verify that sim.calculate() returns weighted MicroSeries by default.

        This is the expected behavior for user-facing calculations.
        """
        sim = Microsimulation()
        result = sim.calculate("salary", "2022-01")

        assert isinstance(result, MicroSeries), (
            f"Expected MicroSeries but got {type(result).__name__}. "
            "sim.calculate() should return weighted MicroSeries by default."
        )

    def test_isinstance_check_works_for_microsimulation(self):
        """
        Directly test that isinstance check works for Microsimulation.

        This ensures the isinstance check in Population.__call__ correctly
        identifies Microsimulation instances across Python versions.
        """
        from policyengine_core.simulations.microsimulation import (
            Microsimulation as CoreMicrosimulation,
        )

        sim = Microsimulation()

        assert isinstance(sim, CoreMicrosimulation), (
            f"isinstance(sim, Microsimulation) returned False. "
            f"sim type: {type(sim)}, MRO: {type(sim).__mro__}"
        )

    def test_person_accessor_kwargs_passed_correctly(self):
        """
        Test that the person() accessor passes the correct kwargs to calculate().

        This test verifies that use_weights=False is passed to avoid
        performance issues in internal calculations.
        """
        sim = Microsimulation()

        # Call person() which should pass use_weights=False
        result_person = sim.person("salary", "2022-01")

        # Call calculate() with use_weights=False directly
        result_calculate = sim.calculate(
            "salary", "2022-01", use_weights=False
        )

        # Both should return numpy arrays with the same values
        assert isinstance(result_person, np.ndarray)
        assert isinstance(result_calculate, np.ndarray)
        np.testing.assert_array_equal(result_person, result_calculate)
