"""Regression test: ``period.get_subperiods`` on ETERNITY gives a clear error (H7).

ETERNITY periods carry ``size=float("inf")``. ``get_subperiods`` previously
did ``range(self.size)`` / ``range(self.size_in_months)`` without checking,
so any code that called ``eternity_period.get_subperiods(...)`` crashed with
``TypeError: 'float' object cannot be interpreted as an integer``.

Fail fast with a descriptive ``ValueError`` instead.
"""

from __future__ import annotations

import pytest

from policyengine_core import periods


def test_eternity_get_subperiods_raises_value_error():
    eternity = periods.period(periods.ETERNITY)
    with pytest.raises(ValueError, match="ETERNITY"):
        eternity.get_subperiods(periods.MONTH)


def test_eternity_get_subperiods_raises_value_error_for_year():
    eternity = periods.period(periods.ETERNITY)
    with pytest.raises(ValueError, match="ETERNITY"):
        eternity.get_subperiods(periods.YEAR)
