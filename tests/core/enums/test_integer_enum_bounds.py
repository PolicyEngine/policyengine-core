"""Integer enum positions are zero-based and validated before narrowing."""

import numpy as np
import pytest

from policyengine_core.enums import Enum, EnumArray
from policyengine_core.enums.config import ENUM_ARRAY_DTYPE


class PositionChoice(Enum):
    FIRST = "First"
    SECOND = "Second"
    LAST = "Last"


INTEGER_DTYPES = [
    pytest.param(np.int8, id="int8"),
    pytest.param(np.int16, id="int16"),
    pytest.param(np.int32, id="int32"),
    pytest.param(np.int64, id="int64"),
    pytest.param(np.uint8, id="uint8"),
    pytest.param(np.uint16, id="uint16"),
    pytest.param(np.uint32, id="uint32"),
    pytest.param(np.uint64, id="uint64"),
]

INVALID_ARRAYS = [
    pytest.param(np.array([-1], dtype=np.int8), "-1", id="negative-int8"),
    pytest.param(
        np.array([np.iinfo(np.int64).min], dtype=np.int64),
        str(np.iinfo(np.int64).min),
        id="signed-minimum",
    ),
    pytest.param(np.array([3], dtype=np.int8), "3", id="upper-bound-int8"),
    pytest.param(np.array([3], dtype=np.uint8), "3", id="upper-bound-uint8"),
    pytest.param(np.array([65536], dtype=np.int64), "65536", id="wide-signed"),
    pytest.param(np.array([65535], dtype=np.uint16), "65535", id="wide-unsigned"),
    pytest.param(
        np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
        str(np.iinfo(np.uint64).max),
        id="unsigned-maximum",
    ),
    pytest.param(np.array([0, -1, 2, 4], dtype=np.int64), "4", id="mixed-bounds"),
]


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_integer_positions_keep_zero_based_order(dtype):
    values = np.array([0, 2, 1, 0], dtype=dtype)
    original = values.copy()
    encoded = PositionChoice.encode(values)
    assert isinstance(encoded, EnumArray)
    assert encoded.possible_values is PositionChoice
    assert encoded.dtype == np.dtype(ENUM_ARRAY_DTYPE)
    np.testing.assert_array_equal(encoded.view(np.ndarray), [0, 2, 1, 0])
    np.testing.assert_array_equal(
        encoded.decode_to_str(), ["FIRST", "LAST", "SECOND", "FIRST"]
    )
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("values,bad_value", INVALID_ARRAYS)
def test_invalid_integer_positions_raise_contextual_value_error(values, bad_value):
    original = values.copy()
    with pytest.raises(ValueError) as error:
        PositionChoice.encode(values)
    assert "PositionChoice" in str(error.value)
    assert bad_value in str(error.value)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64], ids=["signed", "unsigned"])
def test_empty_typed_integer_array_is_still_supported(dtype):
    encoded = PositionChoice.encode(np.array([], dtype=dtype))
    assert isinstance(encoded, EnumArray)
    assert encoded.possible_values is PositionChoice
    assert encoded.shape == (0,)
    assert encoded.dtype == np.dtype(ENUM_ARRAY_DTYPE)


@pytest.mark.parametrize("dtype", [np.int16, np.uint64], ids=["stored", "wide"])
def test_valid_same_enum_array_retains_identity(dtype):
    values = EnumArray(np.array([0, 2], dtype=dtype), PositionChoice)
    assert PositionChoice.encode(values) is values
    assert values.possible_values is PositionChoice
    assert values.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(values.view(np.ndarray), [0, 2])


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array([-1], dtype=np.int16), id="negative"),
        pytest.param(np.array([3], dtype=np.int16), id="upper-bound"),
        pytest.param(np.array([65536], dtype=np.int64), id="wide-signed"),
        pytest.param(
            np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
            id="unsigned-maximum",
        ),
    ],
)
def test_enum_array_cannot_bypass_integer_bounds(values):
    encoded = EnumArray(values.copy(), PositionChoice)
    with pytest.raises(ValueError, match="PositionChoice"):
        PositionChoice.encode(encoded)
    assert encoded.possible_values is PositionChoice
    np.testing.assert_array_equal(encoded.view(np.ndarray), values)


def test_valid_foreign_enum_array_keeps_existing_identity_without_relabelling():
    class OtherChoice(Enum):
        ONE = "One"
        TWO = "Two"

    values = EnumArray(np.array([0, 1], dtype=np.int16), OtherChoice)
    assert PositionChoice.encode(values) is values
    assert values.possible_values is OtherChoice


def test_valid_unbound_enum_array_is_not_silently_relabelled():
    values = EnumArray(np.array([0, 2], dtype=np.int16))
    assert PositionChoice.encode(values) is values
    assert values.possible_values is None


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array(["FIRST", "LAST"]), id="names"),
        pytest.param(np.array([b"FIRST", b"LAST"]), id="utf8-bytes"),
        pytest.param(
            np.array([PositionChoice.FIRST, PositionChoice.LAST]), id="enum-items"
        ),
    ],
)
def test_existing_name_and_item_routes_keep_their_positions(values):
    encoded = PositionChoice.encode(values)
    np.testing.assert_array_equal(encoded.view(np.ndarray), [0, 2])


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array([0.5]), id="float"),
        pytest.param(np.array([True]), id="boolean"),
    ],
)
def test_integer_bounds_do_not_add_new_dtype_coercions(values):
    with pytest.raises(ValueError, match="Unsupported array dtype"):
        PositionChoice.encode(values)


def test_foreign_bound_array_valid_for_its_owner_retains_identity():
    class LongChoice(Enum):
        FIRST = "First"
        SECOND = "Second"
        THIRD = "Third"
        FOURTH = "Fourth"

    values = EnumArray(np.array([3], dtype=np.int16), LongChoice)
    assert PositionChoice.encode(values) is values
    assert values.possible_values is LongChoice
    assert values.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(values.view(np.ndarray), [3])
    np.testing.assert_array_equal(values.decode_to_str(), ["FOURTH"])


def test_foreign_bound_array_invalid_for_its_owner_is_rejected():
    class ShortChoice(Enum):
        FIRST = "First"
        SECOND = "Second"

    values = EnumArray(np.array([2], dtype=np.int16), ShortChoice)
    with pytest.raises(ValueError, match="ShortChoice") as error:
        PositionChoice.encode(values)
    assert "2" in str(error.value)
    assert values.possible_values is ShortChoice
    assert values.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(values.view(np.ndarray), [2])
