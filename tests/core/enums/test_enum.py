import pytest
import numpy as np
import logging
from policyengine_core.enums.enum import Enum
from policyengine_core.enums.enum_array import EnumArray


def test_enum_creation():
    """
    Test to make sure that various types of numpy arrays
    are correctly encoded to int-typed EnumArray instances;
    check enum_array.py to see why int-typed
    """

    test_simple_array = ["MAXWELL", "DWORKIN", "MAXWELL"]

    class Sample(Enum):
        MAXWELL = "maxwell"
        DWORKIN = "dworkin"

    sample_string_array = np.array(test_simple_array)
    sample_item_array = np.array(
        [Sample.MAXWELL, Sample.DWORKIN, Sample.MAXWELL]
    )
    explicit_s_array = np.array(test_simple_array, "S")

    encoded_array = Sample.encode(sample_string_array)
    assert len(encoded_array) == 3
    assert isinstance(encoded_array, EnumArray)
    assert encoded_array.dtype.kind == "i"

    encoded_array = Sample.encode(sample_item_array)
    assert len(encoded_array) == 3
    assert isinstance(encoded_array, EnumArray)
    assert encoded_array.dtype.kind == "i"

    encoded_array = Sample.encode(explicit_s_array)
    assert len(encoded_array) == 3
    assert isinstance(encoded_array, EnumArray)
    assert encoded_array.dtype.kind == "i"


def test_enum_encode_invalid_values_logs_warning(caplog):
    """Test that encoding invalid enum string values logs a warning."""

    class Sample(Enum):
        MAXWELL = "maxwell"
        DWORKIN = "dworkin"

    # Array with invalid values mixed in
    array_with_invalid = np.array(["MAXWELL", "INVALID_VALUE", "DWORKIN"])

    with caplog.at_level(logging.WARNING):
        encoded = Sample.encode(array_with_invalid)

    # Should still return an array (with 0 for invalid)
    assert len(encoded) == 3
    assert encoded[0] == Sample.MAXWELL.index
    assert encoded[1] == 0  # Invalid defaults to 0
    assert encoded[2] == Sample.DWORKIN.index

    # Should have logged a warning
    assert any("INVALID_VALUE" in record.message for record in caplog.records)
    assert any("Sample" in record.message for record in caplog.records)


def test_enum_encode_all_invalid_logs_warning(caplog):
    """Test that encoding all invalid values logs a warning."""

    class Sample(Enum):
        MAXWELL = "maxwell"
        DWORKIN = "dworkin"

    all_invalid = np.array(["FOO", "BAR", "BAZ"])

    with caplog.at_level(logging.WARNING):
        encoded = Sample.encode(all_invalid)

    # All should be 0
    assert all(encoded == 0)

    # Should have logged warnings
    assert len(caplog.records) > 0
