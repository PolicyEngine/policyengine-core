"""
Tests for pandas 3.0.0 compatibility.

These tests verify that policyengine-core works correctly with pandas 3.0.0,
which introduces:
1. PyArrow-backed strings as default (StringDtype)
2. Copy-on-Write by default
"""

import numpy as np
import pandas as pd
import pytest


class TestFilledArrayWithStringDtype:
    """Test that filled_array works with pandas StringDtype."""

    def test_filled_array_with_string_dtype(self):
        """
        In pandas 3.0.0, string columns use StringDtype by default.
        numpy.full() cannot handle StringDtype, so we need to handle this case.
        """
        from policyengine_core.populations.population import Population
        from policyengine_core.entities import Entity

        # Create a minimal entity for testing
        entity = Entity(
            key="person",
            plural="people",
            label="Person",
            doc="Test person entity",
        )

        # Create a population with some count
        population = Population(entity)
        population.count = 5

        # Test with regular numpy dtype - should work
        result = population.filled_array("test_value", dtype=object)
        assert len(result) == 5
        assert all(v == "test_value" for v in result)

        # Test with pandas StringDtype - this is what pandas 3 uses by default
        # This should NOT raise an error
        string_dtype = pd.StringDtype()
        result = population.filled_array("test_value", dtype=string_dtype)
        assert len(result) == 5
        assert all(v == "test_value" for v in result)

    def test_filled_array_with_pyarrow_string_dtype(self):
        """
        Test with PyArrow-backed string dtype, which pandas 3 uses by default.
        """
        pa = pytest.importorskip("pyarrow")

        from policyengine_core.populations.population import Population
        from policyengine_core.entities import Entity

        entity = Entity(
            key="person",
            plural="people",
            label="Person",
            doc="Test person entity",
        )
        population = Population(entity)
        population.count = 5

        # PyArrow string dtype (proper way to create it)
        arrow_string_dtype = pd.ArrowDtype(pa.string())
        result = population.filled_array(
            "test_value", dtype=arrow_string_dtype
        )
        assert len(result) == 5


class TestParameterLookupWithStringArray:
    """Test that parameter lookup works with pandas StringArray."""

    def test_parameter_node_getitem_with_string_array(self):
        """
        In pandas 3.0.0, series.values.astype(str) returns a StringArray
        instead of a numpy array. ParameterNodeAtInstant.__getitem__ should
        handle this.
        """
        # Create a pandas StringArray (what pandas 3 returns)
        string_array = pd.array(["value1", "value2", "value3"], dtype="string")

        # Verify it's a StringArray (not numpy array)
        assert not isinstance(string_array, np.ndarray)
        assert hasattr(string_array, "__array__")

        # Convert to numpy - this is what the fix should do
        numpy_array = np.asarray(string_array)
        assert isinstance(numpy_array, np.ndarray)

    def test_vectorial_parameter_node_with_string_array(self):
        """
        VectorialParameterNodeAtInstant.__getitem__ should handle pandas
        StringArray by converting it to numpy array.
        """
        from policyengine_core.parameters.vectorial_parameter_node_at_instant import (
            VectorialParameterNodeAtInstant,
        )

        # Create a simple vectorial node for testing with proper structure
        vector = np.array(
            [(1.0, 2.0)],
            dtype=[("zone_1", "float"), ("zone_2", "float")],
        ).view(np.recarray)

        node = VectorialParameterNodeAtInstant("test", vector, "2024-01-01")

        # Test with numpy array - should work
        key_numpy = np.array(["zone_1", "zone_2"])
        result_numpy = node[key_numpy]
        assert len(result_numpy) == 2

        # Test with pandas StringArray - this is what pandas 3 returns
        key_string_array = pd.array(["zone_1", "zone_2"], dtype="string")

        # This should NOT raise TypeError: unhashable type: 'StringArray'
        # The node should accept StringArray by converting to numpy
        result_string_array = node[key_string_array]
        assert len(result_string_array) == 2

        # Results should be the same
        np.testing.assert_array_equal(result_numpy, result_string_array)


class TestMicroSeriesCompatibility:
    """Test that MicroSeries operations work with pandas 3."""

    def test_series_subclass_preserved(self):
        """
        Pandas 3.0.0 may change how Series subclasses are handled.
        Operations should return the subclass, not plain Series.
        """
        # This test documents expected behavior that may break in pandas 3
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Test that operations preserve Series type
        result = df["a"] + df["b"]
        assert isinstance(result, pd.Series)

        # With pandas 3, some operations may return different types
        result = df["a"].astype(str)
        # In pandas 3, this might return StringArray-backed Series
        assert isinstance(result, pd.Series)


class TestStringDtypeConversion:
    """Test utilities for converting pandas StringDtype to numpy-compatible types."""

    def test_convert_string_dtype_to_object(self):
        """
        When pandas StringDtype is passed to numpy functions,
        we should convert it to object dtype.
        """
        string_dtype = pd.StringDtype()

        # numpy.full doesn't understand StringDtype
        with pytest.raises(TypeError):
            np.full(5, "test", dtype=string_dtype)

        # But it works with object dtype
        result = np.full(5, "test", dtype=object)
        assert len(result) == 5

    def test_is_pandas_extension_dtype(self):
        """Test detection of pandas extension dtypes."""
        # pandas StringDtype is an ExtensionDtype
        assert isinstance(pd.StringDtype(), pd.api.extensions.ExtensionDtype)

        # numpy dtypes are not
        assert not isinstance(
            np.dtype("float64"), pd.api.extensions.ExtensionDtype
        )
        assert not isinstance(
            np.dtype("object"), pd.api.extensions.ExtensionDtype
        )
