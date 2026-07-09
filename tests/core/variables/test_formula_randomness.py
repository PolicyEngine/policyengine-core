"""Unit tests for the static formula-randomness detector (policyengine-core#518).

These exercise ``formula_uses_randomness`` / ``check_formula_determinism``
directly on hand-written functions -- no tax-benefit system is built here. The
detector is wired into ``Variable.__init__`` in a later commit; these tests pin
its contract independently of that wiring.
"""

import random  # stdlib, imported at module scope on purpose (collision bait)

import numpy
import numpy as np
import numpy.random as npr
import pytest
from numpy.random import default_rng
from numpy.random import random as bare_np_random

from policyengine_core.commons.formulas import random as commons_random
from policyengine_core.variables.formula_randomness import (
    NonDeterministicFormulaError,
    check_formula_determinism,
    formula_uses_randomness,
)

# A generator hoisted to module scope -- the "pre-built generator" the runtime
# guard could not catch.
MODULE_RNG = np.random.default_rng(0)


# --- positives: every form of formula-time randomness must be detected --------


def _np_random_seed(person, period, parameters):
    np.random.seed(0)
    return person("salary", period)


def _np_random_draw(person, period, parameters):
    return np.random.random(len(person("salary", period)))


def _numpy_random_fullname(person, period, parameters):
    return numpy.random.choice([0, 1], size=3)


def _stdlib_random(person, period, parameters):
    return random.random()


def _aliased_numpy_random_module(person, period, parameters):
    npr.seed(0)
    return person("salary", period)


def _imported_default_rng(person, period, parameters):
    return default_rng(0).random(3)


def _bare_imported_drawing_function(person, period, parameters):
    return bare_np_random(3)


def _prebuilt_module_generator(person, period, parameters):
    return MODULE_RNG.random(3)


def _retired_commons_helper(person, period, parameters):
    return commons_random(person)


POSITIVES = [
    _np_random_seed,
    _np_random_draw,
    _numpy_random_fullname,
    _stdlib_random,
    _aliased_numpy_random_module,
    _imported_default_rng,
    _bare_imported_drawing_function,
    _prebuilt_module_generator,
    _retired_commons_helper,
]


@pytest.mark.parametrize("formula", POSITIVES, ids=lambda f: f.__name__)
def test_randomness_is_detected(formula):
    assert formula_uses_randomness(formula) is not None


def test_bare_imported_drawing_function_is_now_caught():
    # Former runtime-guard gap: `from numpy.random import random` imported by name.
    assert formula_uses_randomness(_bare_imported_drawing_function) is not None


def test_prebuilt_module_generator_is_now_caught():
    # Former runtime-guard gap: a Generator built before the formula runs.
    assert formula_uses_randomness(_prebuilt_module_generator) is not None


# --- negatives: deterministic numpy use and name collisions must NOT flag -----


def _deterministic_numpy(person, period, parameters):
    salary = person("salary", period)
    return np.where(salary > 0, np.maximum(salary, 1), 0)


def _attribute_named_random(person, period, parameters):
    # `.random` here is an attribute access on a non-random object; `random` is
    # also imported at module scope -- the detector must not conflate the two.
    return person.random


def _local_named_random(person, period, parameters):
    random = 5  # a local, not the module
    return random


def _string_argument_random(person, period, parameters):
    # "random_draw" is a precomputed input variable, read like any other.
    return person("random_draw", period)


NEGATIVES = [
    _deterministic_numpy,
    _attribute_named_random,
    _local_named_random,
    _string_argument_random,
]


@pytest.mark.parametrize("formula", NEGATIVES, ids=lambda f: f.__name__)
def test_deterministic_formula_is_not_flagged(formula):
    assert formula_uses_randomness(formula) is None


def test_non_function_returns_none():
    assert formula_uses_randomness(None) is None
    assert formula_uses_randomness("not a formula") is None


# --- closures: randomness captured through a closure is resolved too ----------


def _closure_over_generator():
    rng = np.random.default_rng(0)

    def formula(person, period, parameters):
        return rng.random(3)

    return formula


def _closure_over_drawing_function():
    draw = np.random.random

    def formula(person, period, parameters):
        return draw(3)

    return formula


def _closure_over_numpy_module():
    local_numpy = np

    def formula(person, period, parameters):
        return local_numpy.random.random(3)

    return formula


def _closure_over_deterministic_numpy():
    op = np.maximum

    def formula(person, period, parameters):
        return op(person("salary", period), 0)

    return formula


@pytest.mark.parametrize(
    "factory",
    [
        _closure_over_generator,
        _closure_over_drawing_function,
        _closure_over_numpy_module,
    ],
    ids=lambda f: f.__name__,
)
def test_randomness_captured_through_closure_is_detected(factory):
    assert formula_uses_randomness(factory()) is not None


def test_deterministic_closure_is_not_flagged():
    assert formula_uses_randomness(_closure_over_deterministic_numpy()) is None


# --- check_formula_determinism (the variable-level entry point) ----------------


class _FakeVariable:
    def __init__(self, name, formulas):
        self.name = name
        self.formulas = formulas


def test_check_raises_naming_the_variable():
    variable = _FakeVariable("uses_randomness", {"0001-01-01": _np_random_seed})
    with pytest.raises(NonDeterministicFormulaError, match="uses_randomness"):
        check_formula_determinism(variable)


def test_check_passes_for_deterministic_formulas():
    variable = _FakeVariable(
        "clean",
        {"0001-01-01": _deterministic_numpy, "2020-01-01": _string_argument_random},
    )
    check_formula_determinism(variable)  # must not raise


def test_check_passes_for_formula_less_variable():
    check_formula_determinism(_FakeVariable("input_only", {}))  # must not raise


def test_exception_importable_from_legacy_guard_path():
    # Back-compat: the runtime guard module was reduced to a shim, but the
    # exception must still import from its old location as the same class.
    from policyengine_core.simulations.randomness_guard import (
        NonDeterministicFormulaError as LegacyImport,
    )

    assert LegacyImport is NonDeterministicFormulaError
