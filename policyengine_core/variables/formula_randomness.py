"""Static detection of formula-time randomness (policyengine-core#518).

A rules-engine formula must be a pure, deterministic function of its inputs.
Rather than forbidding randomness with a runtime monkeypatch of the
process-global ``numpy.random`` -- which is not safe under concurrent
simulations (see #518: the guard installed by one thread's formula leaks onto
another thread's legitimate setup ``np.random.seed(0)``) -- this module scans a
formula's bytecode *once, when the variable is constructed* and raises if it
references numpy's or the standard library's random facilities.

Detection is by **identity**, not by name: a local variable, a string argument,
or an attribute that merely happens to be spelled ``random`` is never flagged.
The bytecode is walked so that only genuine *global* / *closure* loads are
resolved against the formula's namespace (attribute loads such as ``foo.random``
are handled separately), which is what keeps the check free of name-collision
false positives.

Because it is static, this also catches two forms the old runtime guard
explicitly could not: ``from numpy.random import default_rng`` (a drawing
callable imported by name) and a generator hoisted to module scope
(``rng = np.random.default_rng(0)``) and used inside the formula.
"""

from __future__ import annotations

import dis
import random as _stdlib_random
from types import ModuleType
from typing import Iterator, Optional

import numpy


class NonDeterministicFormulaError(RuntimeError):
    """Raised when a variable's formula references a random number generator."""


_RANDOM_MODULES = (numpy.random, _stdlib_random)

# Scanning is memoised by code object: each unique formula is inspected once,
# no matter how many tax-benefit systems instantiate the variable.
_cache: dict = {}


def _is_random_module(obj: object) -> bool:
    """True if ``obj`` is the ``numpy.random`` or stdlib ``random`` module."""
    return any(obj is module for module in _RANDOM_MODULES)


def _is_random_member(obj: object) -> bool:
    """True if ``obj`` is a callable / generator that belongs to a random module.

    Catches ``from numpy.random import default_rng`` / ``from random import
    random`` (imported drawing callables), pre-built ``Generator`` /
    ``RandomState`` instances hoisted to module scope, and the retired
    ``commons.formulas.random`` helper (kept as a raiser since #494).
    """
    module = getattr(obj, "__module__", None)
    if isinstance(module, str) and (
        module == "random" or module.startswith("numpy.random")
    ):
        return True
    if isinstance(obj, (numpy.random.Generator, numpy.random.RandomState)):
        return True
    if (
        getattr(obj, "__module__", None) == "policyengine_core.commons.formulas"
        and getattr(obj, "__name__", None) == "random"
    ):
        return True
    return False


def _describe(obj: object) -> str:
    """A short, human-readable name for the offending reference."""
    if isinstance(obj, ModuleType):
        return obj.__name__
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    if isinstance(obj, (numpy.random.Generator, numpy.random.RandomState)):
        return f"numpy.random.{type(obj).__name__} instance"
    return repr(obj)


def _global_and_closure_loads(formula) -> Iterator[object]:
    """Yield the objects a formula loads as globals or closure variables.

    Only ``LOAD_GLOBAL`` / ``LOAD_NAME`` / ``LOAD_DEREF`` are resolved -- never
    ``LOAD_ATTR`` -- so ``foo.random`` (an attribute access) is not mistaken for
    a reference to the ``random`` module, even when ``random`` is also imported
    at module scope.
    """
    global_namespace = formula.__globals__
    freevars: dict = {}
    if formula.__closure__:
        for name, cell in zip(formula.__code__.co_freevars, formula.__closure__):
            try:
                freevars[name] = cell.cell_contents
            except ValueError:
                continue  # empty cell
    for instruction in dis.get_instructions(formula):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            if instruction.argval in global_namespace:
                yield global_namespace[instruction.argval]
        elif instruction.opname in ("LOAD_DEREF", "LOAD_CLASSDEREF"):
            if instruction.argval in freevars:
                yield freevars[instruction.argval]


def _numpy_random_attr_access(formula) -> bool:
    """True if the formula accesses ``<numpy-module>.random`` (e.g. ``np.random.x``).

    A global that resolves to the numpy top-level module immediately followed by
    a ``LOAD_ATTR random``.
    """
    global_namespace = formula.__globals__
    instructions = list(dis.get_instructions(formula))
    for current, following in zip(instructions, instructions[1:]):
        if current.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            if (
                global_namespace.get(current.argval) is numpy
                and following.opname in ("LOAD_ATTR", "LOAD_METHOD")
                and following.argval == "random"
            ):
                return True
    return False


def formula_uses_randomness(formula) -> Optional[str]:
    """Return a description of the randomness a formula references, or ``None``.

    ``formula`` is a plain function (a ``def formula(...)`` collected from a
    ``Variable`` subclass). Objects without a code object (e.g. ``None``) return
    ``None``. Memoised by code object.
    """
    code = getattr(formula, "__code__", None)
    if code is None:
        return None
    if code in _cache:
        return _cache[code]

    result: Optional[str] = None
    for obj in _global_and_closure_loads(formula):
        if _is_random_module(obj) or _is_random_member(obj):
            result = _describe(obj)
            break
    if result is None and _numpy_random_attr_access(formula):
        result = "numpy.random"

    _cache[code] = result
    return result


def check_formula_determinism(variable) -> None:
    """Raise ``NonDeterministicFormulaError`` if any of a variable's formulas
    references randomness.

    Intended to be called from ``Variable.__init__`` alongside
    ``check_computation_modes`` so a randomness-using formula fails fast, at
    variable construction, with the offending variable named.
    """
    for formula in variable.formulas.values():
        offending = formula_uses_randomness(formula)
        if offending is not None:
            raise NonDeterministicFormulaError(
                f"The formula for '{variable.name}' references {offending}, but "
                f"rules-engine formulas must be deterministic functions of their "
                f"inputs. Remove the random call. If you need a stochastic input, "
                f"compute it once when building the dataset (with a seeded "
                f"generator) and store it as an input variable instead."
            )
