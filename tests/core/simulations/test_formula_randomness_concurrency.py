"""Concurrent simulations must not spuriously raise (policyengine-core#518).

The removed runtime guard mutated the process-global ``numpy.random`` while a
formula ran, so one thread's formula could make another thread's legitimate
``Simulation.__init__`` ``np.random.seed(0)`` raise
``NonDeterministicFormulaError`` -- intermittently, and misattributed to an
innocent formula. Now that enforcement is static (at variable registration),
building and calculating simulations concurrently is safe.

This test reproduces that race: several threads each repeatedly build a default
simulation (each build calls the legitimate setup ``np.random.seed(0)``) and
compute a formula variable, with a barrier to maximise overlap. It passes with
the static check and fails against the reinstated runtime guard.
"""

import threading

import numpy as np

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables.formula_randomness import (
    NonDeterministicFormulaError,
)

PERIOD = "2013-01"
THREADS = 12
ITERATIONS = 60


def test_concurrent_build_and_calculate_is_safe():
    barrier = threading.Barrier(THREADS)
    errors: list = []

    def worker():
        # A per-thread system isolates unrelated state; the #518 race is on the
        # shared global numpy.random, which per-thread systems still exercise.
        system = CountryTaxBenefitSystem()
        barrier.wait()  # release all threads together to maximise overlap
        try:
            for _ in range(ITERATIONS):
                simulation = SimulationBuilder().build_default_simulation(system)
                simulation.calculate("disposable_income", PERIOD)
        except Exception as exc:  # noqa: BLE001 - captured for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    nondeterministic = [
        e for e in errors if isinstance(e, NonDeterministicFormulaError)
    ]
    assert not nondeterministic, (
        f"concurrent simulations raised spurious randomness errors: "
        f"{nondeterministic!r}"
    )
    assert not errors, f"concurrent simulations raised: {errors!r}"
    # numpy randomness still works normally in the main thread afterwards.
    assert isinstance(float(np.random.random()), float)
