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
Only genuine *global* and *closure* loads are resolved against the formula's
namespace (attribute loads such as ``foo.random`` are not), which keeps the
check free of name-collision false positives.

Because it is static, this also catches forms the old runtime guard explicitly
could not: ``from numpy.random import default_rng`` (a drawing callable imported
by name) and a generator hoisted to module scope or captured through a closure
(``rng = np.random.default_rng(0)``) and used inside the formula.
"""

from __future__ import annotations

import dis
import random as _stdlib_random
from types import ModuleType
from typing import Optional

import numpy


class NonDeterministicFormulaError(RuntimeError):
    """Raised when a variable's formula references a random number generator."""


_RANDOM_MODULES = (numpy.random, _stdlib_random)
_RNG_INSTANCE_TYPES = (numpy.random.Generator, numpy.random.RandomState)

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
    if isinstance(obj, _RNG_INSTANCE_TYPES):
        return True
    if (
        module == "policyengine_core.commons.formulas"
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
    if isinstance(obj, _RNG_INSTANCE_TYPES):
        return f"numpy.random.{type(obj).__name__} instance"
    return repr(obj)


def _freevars(formula) -> dict:
    """Map each of a formula's closure variable names to its current value."""
    result: dict = {}
    if formula.__closure__:
        for name, cell in zip(formula.__code__.co_freevars, formula.__closure__):
            try:
                result[name] = cell.cell_contents
            except ValueError:
                continue  # empty cell
    return result


def _resolve_load(instruction, global_namespace: dict, freevars: dict) -> object:
    """Resolve a name-load instruction to the object it loads, or ``None``.

    Only global and closure loads are resolved -- never ``LOAD_ATTR`` -- so an
    attribute merely spelled ``random`` is not mistaken for the module, even
    when ``random`` is also imported at module scope.
    """
    if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
        return global_namespace.get(instruction.argval)
    if instruction.opname in ("LOAD_DEREF", "LOAD_CLASSDEREF"):
        return freevars.get(instruction.argval)
    return None


def _referenced_randomness(formula) -> Optional[str]:
    """Describe the randomness a formula references, or ``None``.

    A single pass over the bytecode: each global/closure load is checked by
    identity, and a numpy top-level module load followed by a ``.random``
    attribute access (e.g. ``np.random.seed``) is flagged too.
    """
    global_namespace = formula.__globals__
    freevars = _freevars(formula)
    previous = None
    for instruction in dis.get_instructions(formula):
        loaded = _resolve_load(instruction, global_namespace, freevars)
        if loaded is not None and (
            _is_random_module(loaded) or _is_random_member(loaded)
        ):
            return _describe(loaded)
        if (
            instruction.opname in ("LOAD_ATTR", "LOAD_METHOD")
            and instruction.argval == "random"
            and previous is not None
            and _resolve_load(previous, global_namespace, freevars) is numpy
        ):
            return "numpy.random"
        previous = instruction
    return None


def formula_uses_randomness(formula) -> Optional[str]:
    """Return a description of the randomness a formula references, or ``None``.

    ``formula`` is a plain function (a ``def formula(...)`` collected from a
    ``Variable`` subclass). Objects without a code object (e.g. ``None``) return
    ``None``. Memoised by code object.
    """
    code = getattr(formula, "__code__", None)
    if code is None:
        return None
    if code not in _cache:
        _cache[code] = _referenced_randomness(formula)
    return _cache[code]


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
