import datetime

import numpy

from policyengine_core import enums
from policyengine_core.enums import Enum

VALUE_TYPES = {
    bool: {
        "dtype": numpy.bool_,
        "default": False,
        "json_type": "boolean",
        "formatted_value_type": "Boolean",
        "is_period_size_independent": True,
    },
    int: {
        "dtype": numpy.int32,
        "default": 0,
        "json_type": "integer",
        "formatted_value_type": "Int",
        "is_period_size_independent": False,
    },
    float: {
        "dtype": numpy.float32,
        "default": 0,
        "json_type": "number",
        "formatted_value_type": "Float",
        "is_period_size_independent": False,
    },
    str: {
        "dtype": object,
        "default": "",
        "json_type": "string",
        "formatted_value_type": "String",
        "is_period_size_independent": True,
    },
    Enum: {
        "dtype": enums.ENUM_ARRAY_DTYPE,
        "json_type": "string",
        "formatted_value_type": "String",
        "is_period_size_independent": True,
    },
    datetime.date: {
        "dtype": "datetime64[D]",
        # ``datetime.date.fromtimestamp(0)`` is timezone-dependent (returns
        # ``1970-01-01`` in UTC, ``1969-12-31`` in any zone west of UTC), so
        # the default value for ``datetime.date`` variables changed
        # depending on the deployment timezone. Use an explicit literal
        # (bug M13).
        "default": datetime.date(1970, 1, 1),
        "json_type": "string",
        "formatted_value_type": "Date",
        "is_period_size_independent": True,
    },
}


FORMULA_NAME_PREFIX = "formula"
