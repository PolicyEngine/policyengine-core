"""Deprecated module -- kept only to preserve an import path.

Formula-time randomness is no longer forbidden by a runtime monkeypatch of the
process-global ``numpy.random`` (that guard was not safe under concurrent
simulations; see policyengine-core#518). It is now rejected *statically*, when a
variable is registered, by
:func:`policyengine_core.variables.formula_randomness.check_formula_determinism`.

``NonDeterministicFormulaError`` is re-exported here so existing
``from policyengine_core.simulations.randomness_guard import
NonDeterministicFormulaError`` imports keep working.
"""

from policyengine_core.variables.formula_randomness import (
    NonDeterministicFormulaError,
)

__all__ = ["NonDeterministicFormulaError"]
