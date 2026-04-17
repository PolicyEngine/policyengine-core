"""Regression tests for deterministic seeding in ``Simulation``.

Python's built-in ``hash()`` is randomized per process for strings, so any seed
derived from it changes from one ``python`` invocation to the next. This module
ensures ``Simulation`` uses a stable hash so results involving ``random()`` are
reproducible across runs (issue C6 in the 2026-04 bug hunt, related to #412).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from policyengine_core.simulations.simulation import _stable_hash_to_seed


def test_stable_hash_to_seed_is_deterministic_across_runs():
    # Run a subprocess with the default PYTHONHASHSEED (which is randomized by
    # default in CPython 3.3+). If the hash depended on the built-in ``hash()``
    # over a string, the value would differ each run.
    script = textwrap.dedent(
        """
        from policyengine_core.simulations.simulation import _stable_hash_to_seed
        print(_stable_hash_to_seed("employment_income-2024"))
        """
    ).strip()
    out_a = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    out_b = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Also compute the expected value in-process to assert stability in CI.
    expected = str(_stable_hash_to_seed("employment_income-2024"))
    assert out_a == out_b == expected


def test_stable_hash_to_seed_covers_seed_range():
    # numpy.random.seed accepts integers in [0, 2**32). Our output must be
    # bounded (we truncate mod 1_000_000 to preserve compatibility with
    # pre-existing seed ranges).
    seed = _stable_hash_to_seed("variable_name-2024-01")
    assert isinstance(seed, int)
    assert 0 <= seed < 1_000_000


def test_sort_keys_makes_equivalent_inputs_share_a_seed():
    # Two equivalent situations constructed with different dict insertion order
    # must produce the same hash / seed so calls to ``random()`` are stable.
    a = {"person": {"you": {"employment_income": 1000, "age": 30}}}
    b = {"person": {"you": {"age": 30, "employment_income": 1000}}}
    seed_a = _stable_hash_to_seed(json.dumps(a, sort_keys=True))
    seed_b = _stable_hash_to_seed(json.dumps(b, sort_keys=True))
    assert seed_a == seed_b
