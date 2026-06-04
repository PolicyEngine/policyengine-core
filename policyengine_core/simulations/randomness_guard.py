"""Forbid non-deterministic randomness inside variable formulas.

A rules engine must be a pure function of its inputs: identical inputs must
always produce identical outputs. Calling a random number generator inside a
formula breaks that contract — the same household can get different results on
different runs — and makes whole datasets non-reproducible.

While a formula is executing, this guard replaces the callables exposed by
``numpy.random`` and the standard library ``random`` module with functions that
raise :class:`NonDeterministicFormulaError`. This includes the RNG constructors
``np.random.default_rng``/``Generator``/``RandomState``, so *building* a
generator inside a formula — even a seeded one — also raises. Seeding does not
make randomness acceptable in a formula: stochastic inputs belong in the dataset
(computed once, deterministically, and stored), not in the formula. The guard is
re-entrant, so nested formula evaluation is handled correctly, and it restores
the originals once the outermost guarded formula returns.

Known limitations (pinned by tests in ``test_randomness_guard`` so they cannot
drift silently):

* A ``Generator``/``RandomState`` instance built *before* the formula runs
  (e.g. ``rng = np.random.default_rng(0)`` at module scope) is not intercepted
  when its methods are called inside the formula. ``numpy``'s generator classes
  are immutable C extension types, so their bound methods cannot be patched.
* A *bare drawing function* bound into another module's namespace before the
  guard installs (e.g. ``from numpy.random import random``) is not intercepted,
  because the guard patches the ``numpy.random`` module attribute, not every
  rebinding of it.

Both require deliberately hoisting randomness out of the ``np.random.<fn>`` form;
use ``np.random.<fn>(...)`` or construct the generator inside the formula (both
caught) rather than importing or pre-building drawing callables.
"""

from __future__ import annotations

import random as _stdlib_random
from typing import Callable

import numpy as np


class NonDeterministicFormulaError(RuntimeError):
    """Raised when a formula invokes a random number generator."""


# Public callables exposed by each randomness namespace, captured once at
# import so the per-formula swap is a cheap dict iteration rather than a
# fresh ``dir()`` scan.
def _public_callables(module) -> dict[str, Callable]:
    return {
        name: getattr(module, name)
        for name in dir(module)
        if not name.startswith("_") and callable(getattr(module, name))
    }


_GUARDED_NAMESPACES = (
    ("numpy.random", np.random, _public_callables(np.random)),
    ("random", _stdlib_random, _public_callables(_stdlib_random)),
)

# Re-entrancy bookkeeping: only the outermost guarded formula installs and
# removes the patches; the active variable name is tracked as a stack so the
# error message always names the formula that actually made the call.
_depth = 0
_variable_stack: list[str] = []


def _make_raiser(namespace: str, attribute: str) -> Callable:
    qualified = f"{namespace}.{attribute}"

    def _raise(*args, **kwargs):
        variable = _variable_stack[-1] if _variable_stack else "<unknown>"
        raise NonDeterministicFormulaError(
            f"The formula for '{variable}' called {qualified}(), but rules-engine "
            f"formulas must be deterministic functions of their inputs. Remove the "
            f"random call. If you need a stochastic input, compute it once when "
            f"building the dataset (with a seeded generator) and store it as an "
            f"input variable instead."
        )

    return _raise


# Pre-build the raisers once per (namespace, attribute).
_RAISERS = {
    id(module): {name: _make_raiser(namespace, name) for name in originals}
    for namespace, module, originals in _GUARDED_NAMESPACES
}


def _install() -> None:
    for _namespace, module, originals in _GUARDED_NAMESPACES:
        raisers = _RAISERS[id(module)]
        for name in originals:
            setattr(module, name, raisers[name])


def _restore() -> None:
    for _namespace, module, originals in _GUARDED_NAMESPACES:
        for name, original in originals.items():
            setattr(module, name, original)


class forbid_randomness:
    """Context manager that bans RNG use while a formula runs.

    Re-entrant: nested formulas reuse the single installed patch set and only
    the outermost context restores the originals.
    """

    __slots__ = ("variable_name",)

    def __init__(self, variable_name: str):
        self.variable_name = variable_name

    def __enter__(self) -> "forbid_randomness":
        global _depth
        if _depth == 0:
            _install()
        _depth += 1
        _variable_stack.append(self.variable_name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _depth
        _variable_stack.pop()
        _depth -= 1
        if _depth == 0:
            _restore()
        return False
