"""Regression tests: cache-manipulation methods on ``Simulation`` must not
crash when private cache attributes are missing.

``Simulation.__init__`` sets ``self._fast_cache = {}`` and
``self.invalidated_caches = set()`` as part of initialisation.
Country-package subclasses (e.g. ``policyengine_uk.Simulation``) can
legitimately override ``__init__`` without calling ``super().__init__``
— instead they set the handful of attributes they need directly. In
that case the skipped attribute never gets initialised, so any
cache-mutation path that assumes the attribute exists raised
``AttributeError`` during ``build_from_single_year_dataset`` /
``set_input`` / ``delete_arrays`` / ``purge_cache_of_invalid_values`` /
``invalidate_cache_entry``.

The defensive fix is to guard the bare ``.pop`` / ``.items`` / ``.add``
sites the same way the read-side fast path in ``calculate()`` already
does — ``getattr(self, "<attr>", None)`` and skip or lazily initialise
when it's missing. Core owns this protection so every downstream
subclass doesn't have to mirror the attributes.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from policyengine_core.simulations import Simulation


def _bare_simulation():
    """Create a ``Simulation`` instance that bypasses ``__init__`` entirely.

    Mirrors the pattern a country subclass would hit when overriding
    ``__init__`` and forgetting to initialise ``_fast_cache``.
    """
    return Simulation.__new__(Simulation)


def test_set_input_without_fast_cache_attribute():
    sim = _bare_simulation()

    # Stand-ins for the parts ``set_input`` touches — we're not exercising
    # them here; we just need the cache-pop step not to crash.
    sim.start_instant = None
    sim.branch_name = "default"
    sim.tax_benefit_system = types.SimpleNamespace(
        get_variable=lambda name, check_existence=True: types.SimpleNamespace(end=None)
    )
    sim.get_holder = lambda name: types.SimpleNamespace(
        set_input=lambda period, value, branch: None
    )

    # The actual assertion: this line previously raised
    # ``AttributeError: 'Simulation' object has no attribute '_fast_cache'``.
    sim.set_input("variable_name", "2024", [1, 2, 3])


def test_delete_arrays_without_fast_cache_attribute():
    sim = _bare_simulation()
    sim.get_holder = lambda name: types.SimpleNamespace(
        delete_arrays=lambda period: None
    )
    # No _fast_cache attribute — must not crash
    sim.delete_arrays("variable", period=None)
    sim.delete_arrays("variable", period="2024")


def test_purge_cache_of_invalid_values_without_fast_cache_attribute():
    sim = _bare_simulation()
    sim.tracer = types.SimpleNamespace(stack=[])
    sim.invalidated_caches = {("variable_name", "2024")}
    sim.get_holder = lambda name: types.SimpleNamespace(
        delete_arrays=lambda period: None
    )

    sim.purge_cache_of_invalid_values()
    assert sim.invalidated_caches == set()


def test_purge_cache_of_invalid_values_without_invalidated_caches_attribute():
    sim = _bare_simulation()
    sim.tracer = types.SimpleNamespace(stack=[])
    # No invalidated_caches, no _fast_cache — must not crash
    sim.get_holder = lambda name: types.SimpleNamespace(
        delete_arrays=lambda period: None
    )

    sim.purge_cache_of_invalid_values()


def test_invalidate_cache_entry_without_invalidated_caches_attribute():
    sim = _bare_simulation()
    # No invalidated_caches — method should lazily initialise it
    sim.invalidate_cache_entry("variable_name", "2024")
    assert sim.invalidated_caches == {("variable_name", "2024")}
    # Subsequent invalidations should accumulate
    sim.invalidate_cache_entry("other_variable", "2025")
    assert sim.invalidated_caches == {
        ("variable_name", "2024"),
        ("other_variable", "2025"),
    }
