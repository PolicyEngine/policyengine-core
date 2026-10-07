"""Each clone and branch records its own cache invalidations.

``Simulation.clone`` (and so ``get_branch``) copied ``invalidated_caches``, a
set, by reference, so a simulation and its clones and branches shared it
until one of them purged (``purge_cache_of_invalid_values`` rebinds it). An
invalidation one of them recorded, such as a spiral in a branch, made the
other delete its own cached values for those variables and periods at its
next purge, inputs among them.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable

MONTH = periods.period("2020-06")
PREVIOUS_MONTH = MONTH.last_month
INPUT = 100.0


class spiral_a(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Depends on spiral_b last month"

    def formula(person, period):
        return 5 + person("spiral_b", period.last_month)


class spiral_b(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Depends on spiral_a this month"

    def formula(person, period):
        return 6 + person("spiral_a", period)


@pytest.fixture(scope="module")
def system():
    system = CountryTaxBenefitSystem()
    system.add_variables(spiral_a, spiral_b)
    return system


def _simulation(system):
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.max_spiral_loops = 2
    return simulation


MAKERS = [
    pytest.param(lambda s: s.clone(), id="clone"),
    pytest.param(
        lambda s: s.clone(clone_tax_benefit_system=False),
        id="clone-sharing-system",
    ),
    pytest.param(lambda s: s.get_branch("branch"), id="branch"),
    pytest.param(
        lambda s: s.get_branch("branch", clone_system=True),
        id="branch-with-own-system",
    ),
]
CLONE_MAKERS = MAKERS[:2]


@pytest.mark.parametrize("make", MAKERS)
def test_a_copy_records_invalidations_in_a_set_of_its_own(system, make):
    source = _simulation(system)
    pending = ("salary", MONTH)
    source.invalidate_cache_entry(*pending)

    copy = make(source)

    # It starts from the source's pending invalidations: its cached arrays
    # start as copies of the source's.
    assert copy.invalidated_caches == {pending}
    assert copy.invalidated_caches is not source.invalidated_caches
    copy.invalidate_cache_entry("income_tax", MONTH)
    source.invalidate_cache_entry("basic_income", MONTH)
    assert source.invalidated_caches == {pending, ("basic_income", MONTH)}
    assert copy.invalidated_caches == {pending, ("income_tax", MONTH)}


@pytest.mark.parametrize("make", MAKERS)
def test_a_spiral_in_a_copy_keeps_the_sources_input(system, make):
    source = _simulation(system)
    copy = make(source)
    # Set after copying, so the copy calculates spiral_b where the source
    # reads its input.
    source.set_input("spiral_b", PREVIOUS_MONTH, [INPUT])

    copy.calculate("spiral_a", MONTH)  # spirals
    assert ("spiral_b", PREVIOUS_MONTH) not in source.invalidated_caches
    source.calculate("salary", MONTH)  # ends with a purge

    np.testing.assert_array_equal(
        source.get_holder("spiral_b").get_array(PREVIOUS_MONTH), [INPUT]
    )
    np.testing.assert_array_equal(source.calculate("spiral_a", MONTH), [5 + INPUT])


@pytest.mark.parametrize("make", CLONE_MAKERS)
def test_a_spiral_in_the_source_keeps_a_clones_input(system, make):
    source = _simulation(system)
    clone = make(source)
    clone.set_input("spiral_b", PREVIOUS_MONTH, [INPUT])

    source.calculate("spiral_a", MONTH)  # spirals
    assert ("spiral_b", PREVIOUS_MONTH) not in clone.invalidated_caches
    clone.calculate("salary", MONTH)  # ends with a purge

    np.testing.assert_array_equal(
        clone.get_holder("spiral_b").get_array(PREVIOUS_MONTH), [INPUT]
    )
    np.testing.assert_array_equal(clone.calculate("spiral_a", MONTH), [5 + INPUT])


@pytest.mark.parametrize("make", MAKERS)
def test_a_pending_invalidation_purges_each_copy_once(system, make):
    source = _simulation(system)
    source.set_input("salary", MONTH, [1.0])
    source.invalidate_cache_entry("salary", MONTH)

    copy = make(source)
    copy.purge_cache_of_invalid_values()

    assert copy.get_holder("salary").get_array(MONTH) is None
    assert copy.invalidated_caches == set()
    # The source still holds its value until it purges.
    np.testing.assert_array_equal(source.get_holder("salary").get_array(MONTH), [1])
    assert source.invalidated_caches == {("salary", MONTH)}
    source.purge_cache_of_invalid_values()
    assert source.get_holder("salary").get_array(MONTH) is None
