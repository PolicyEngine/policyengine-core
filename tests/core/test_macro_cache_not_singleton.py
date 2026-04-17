"""Regression test: ``SimulationMacroCache`` is NOT a singleton (C3).

Previously it used a ``Singleton`` metaclass, which meant:

* ``country_version`` was captured from the first ``TaxBenefitSystem`` to
  construct the cache; subsequent systems with a different country-package
  version were silently ignored, breaking cache invalidation on version
  bumps.
* ``cache_folder_path`` / ``cache_file_path`` were shared across calls, so
  two simulations running sequentially could write one variable's cache
  into the other dataset's folder.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from policyengine_core.simulations.simulation_macro_cache import SimulationMacroCache


def _make_tbs(version: str):
    tbs = MagicMock()
    tbs.get_package_metadata.return_value = {"version": version, "name": "whatever"}
    return tbs


def test_each_construction_returns_a_fresh_instance():
    tbs_a = _make_tbs("1.0.0")
    tbs_b = _make_tbs("2.0.0")
    cache_a = SimulationMacroCache(tbs_a)
    cache_b = SimulationMacroCache(tbs_b)
    # Previously this returned the same singleton and ``country_version``
    # stayed at "1.0.0" forever.
    assert cache_a is not cache_b
    assert cache_a.country_version == "1.0.0"
    assert cache_b.country_version == "2.0.0"


def test_cache_paths_are_not_shared_across_instances(tmp_path):
    tbs = _make_tbs("1.0.0")
    cache_a = SimulationMacroCache(tbs)
    cache_b = SimulationMacroCache(tbs)
    cache_a.set_cache_path(tmp_path, "dataset_a", "var", "2024", "default")
    cache_b.set_cache_path(tmp_path, "dataset_b", "var", "2024", "default")
    # Each instance must remember its own path. Under the old singleton,
    # setting cache_b's path overwrote cache_a's — so the next read from
    # cache_a pointed at dataset_b.
    assert "dataset_a" in str(cache_a.cache_file_path)
    assert "dataset_b" in str(cache_b.cache_file_path)
