# Transitional imports to ensure non-breaking changes.
# Could be deprecated in the next major release.
#
# How imports are being used today:
#
#   >>> from policyengine_core.module import symbol
#
# The previous example provokes cyclic dependency problems
# that prevent us from modularizing the different components
# of the library so to make them easier to test and to maintain.
#
# How could them be used after the next major release:
#
#   >>> from policyengine_core import module
#   >>> module.symbol()
#
# And for classes:
#
#   >>> from policyengine_core.module import Symbol
#   >>> Symbol()
#
# See: https://www.python.org/dev/peps/pep-0008/#imports

from .config import (
    DAY,
    MONTH,
    YEAR,
    ETERNITY,
    INSTANT_PATTERN,
    date_by_instant_cache,
    str_by_instant_cache,
    year_or_month_or_day_re,
)

from .helpers import (
    N_,
    instant,
    instant_date,
    period,
    key_period_size,
    unit_weights,
    unit_weight,
)

from .instant_ import Instant
from .period_ import Period
