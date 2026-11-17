"""Property: ``build_default_simulation`` takes exactly the counts ``range`` takes.

For any count: a Python integer (negative ones included), a NumPy integer of
any width or signedness, a 0-d integer array, a bool, a Python or NumPy float
(whole or not, NaN and infinities included), a 0-d float array, a NumPy bool,
a string, ``None``, a ``Decimal`` or a ``Fraction``:

* building raises ``TypeError`` exactly when ``range(count)`` does, and then
  leaves the simulation it was given as it was;
* otherwise every population's ids, and every group's memberships, are
  ``range(count)`` as ``int64``.

Examples are in ``test_simulation_builder.py`` and
``test_group_population_empty.py``.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.simulations import Simulation, SimulationBuilder  # noqa: E402
from policyengine_core.taxbenefitsystems import TaxBenefitSystem  # noqa: E402
from tests.core.test_group_population_empty import (  # noqa: E402
    HOUSEHOLD,
    PERSON,
)

SYSTEM = TaxBenefitSystem([PERSON, HOUSEHOLD])
NUMPY_INTEGER_TYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.intp,
    np.uintp,
]
_sizes = st.integers(0, 40)
_counts = st.one_of(
    st.integers(-3, 40),
    st.tuples(st.sampled_from(NUMPY_INTEGER_TYPES), _sizes).map(
        lambda draw: draw[0](draw[1])
    ),
    _sizes.map(np.array),
    st.booleans(),
    # Bounded, so a regression fails instead of allocating ``np.arange(1e300)``.
    st.floats(-3, 40),
    st.sampled_from([float("nan"), float("inf"), float("-inf")]),
    _sizes.map(float),
    st.floats(-3, 40).map(np.float64),
    st.floats(-3, 40, width=32).map(np.float32),
    st.floats(-3, 40).map(np.array),
    st.booleans().map(np.bool_),
    _sizes.map(str),
    st.none(),
    _sizes.map(Decimal),
    _sizes.map(Fraction),
)


@settings(max_examples=300, deadline=None)
@given(_counts)
def test_default_simulation_takes_the_counts_range_takes(count):
    try:
        expected_ids = list(range(count))
    except TypeError:
        expected_ids = None
    simulation = Simulation(
        tax_benefit_system=SYSTEM, populations=SYSTEM.instantiate_entities()
    )

    if expected_ids is None:
        with pytest.raises(TypeError):
            SimulationBuilder().build_default_simulation(
                SYSTEM, count, simulation=simulation
            )
        for population in simulation.populations.values():
            assert population.count == 0
            assert len(population.ids) == 0
        return

    built = SimulationBuilder().build_default_simulation(
        SYSTEM, count, simulation=simulation
    )
    for population in built.populations.values():
        assert population.count == count
        assert population.ids.dtype == np.int64
        assert population.ids.tolist() == expected_ids
        if not population.entity.is_person:
            assert population.members_entity_id.dtype == np.int64
            assert population.members_entity_id.tolist() == expected_ids
