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
