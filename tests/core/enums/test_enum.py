import pytest
import numpy as np
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


def test_enum_encode_invalid_values_raises_error():
    """Test that encoding invalid enum string values raises ValueError."""

    class Sample(Enum):
        MAXWELL = "maxwell"
        DWORKIN = "dworkin"

    # Array with invalid values mixed in
    array_with_invalid = np.array(["MAXWELL", "INVALID_VALUE", "DWORKIN"])

    with pytest.raises(ValueError) as exc_info:
        Sample.encode(array_with_invalid)

    error_message = str(exc_info.value)
    assert "INVALID_VALUE" in error_message
    assert "Sample" in error_message
    assert "MAXWELL" in error_message  # Valid values listed
    assert "DWORKIN" in error_message  # Valid values listed


def test_enum_encode_all_invalid_raises_error():
    """Test that encoding all invalid values raises ValueError."""

    class Sample(Enum):
        MAXWELL = "maxwell"
        DWORKIN = "dworkin"

    all_invalid = np.array(["FOO", "BAR", "BAZ"])

    with pytest.raises(ValueError) as exc_info:
        Sample.encode(all_invalid)

    error_message = str(exc_info.value)
    # Should mention all unique invalid values
    assert (
        "FOO" in error_message
        or "BAR" in error_message
        or "BAZ" in error_message
    )


def test_enum_encode_empty_string_raises_error():
    """Test that encoding empty strings raises ValueError."""

    class Sample(Enum):
        MAXWELL = "maxwell"
        DWORKIN = "dworkin"

    array_with_empty = np.array(["MAXWELL", "", "DWORKIN"])

    with pytest.raises(ValueError) as exc_info:
        Sample.encode(array_with_empty)

    # Empty string should be in the error message (represented as '')
    assert "''" in str(exc_info.value) or '""' in str(exc_info.value)
