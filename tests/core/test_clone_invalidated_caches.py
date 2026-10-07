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
from policyengine_core.simulations.simulation_result_cache import ResultCacheKey
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
    assert copy.result_cache.invalidated == {pending}
    assert copy.result_cache.invalidated is not source.result_cache.invalidated
    copy.invalidate_cache_entry("income_tax", MONTH)
    source.invalidate_cache_entry("basic_income", MONTH)
    assert source.result_cache.invalidated == {pending, ("basic_income", MONTH)}
    assert copy.result_cache.invalidated == {pending, ("income_tax", MONTH)}


@pytest.mark.parametrize("make", MAKERS)
def test_a_spiral_in_a_copy_keeps_the_sources_input(system, make):
    source = _simulation(system)
    copy = make(source)
    # Set after copying, so the copy calculates spiral_b where the source
    # reads its input.
    source.set_input("spiral_b", PREVIOUS_MONTH, [INPUT])

    copy.calculate("spiral_a", MONTH)  # spirals
    assert ("spiral_b", PREVIOUS_MONTH) not in source.result_cache.invalidated
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
    assert ("spiral_b", PREVIOUS_MONTH) not in clone.result_cache.invalidated
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
    assert copy.result_cache.invalidated == set()
    # The source still holds its value until it purges.
    np.testing.assert_array_equal(source.get_holder("salary").get_array(MONTH), [1])
    assert source.result_cache.invalidated == {("salary", MONTH)}
    source.purge_cache_of_invalid_values()
    assert source.get_holder("salary").get_array(MONTH) is None


@pytest.mark.parametrize("make", MAKERS)
@pytest.mark.parametrize(
    "pending",
    [
        pytest.param((), id="no-pending-invalidations"),
        pytest.param((ResultCacheKey("salary", MONTH),), id="typed-pending-key"),
        pytest.param((("salary", MONTH),), id="legacy-pending-tuple"),
        pytest.param(
            (
                ResultCacheKey("salary", MONTH),
                ResultCacheKey("salary", PREVIOUS_MONTH),
            ),
            id="same-variable-distinct-periods",
        ),
        pytest.param(
            (
                ResultCacheKey("salary", MONTH),
                ResultCacheKey("income_tax", MONTH),
            ),
            id="distinct-variables-same-period",
        ),
    ],
)
def test_copy_inherits_exact_pending_state_without_sharing_its_set(
    system, make, pending
):
    source = _simulation(system)
    # The compatibility view can contain old tuple keys as well as typed keys.
    for name, input_period in pending:
        source.invalidate_cache_entry(name, input_period)
    original_set = source.result_cache.invalidated

    copy = make(source)

    assert copy.result_cache.invalidated == set(pending)
    assert copy.result_cache.invalidated is not original_set
    assert source.result_cache.invalidated == original_set
    assert source.result_cache.invalidated == set(pending)
    # Even an initially empty invalidation set must be independently owned.
    copy.invalidate_cache_entry("basic_income", MONTH)
    assert copy.result_cache.invalidated == set(pending) | {
        ResultCacheKey("basic_income", MONTH)
    }
    assert source.result_cache.invalidated == set(pending)


@pytest.mark.parametrize("make", MAKERS)
@pytest.mark.parametrize(
    "operation",
    [
        "invalidate-source",
        "invalidate-copy",
        "consume-source",
        "consume-copy",
        "discard-copy-key",
    ],
)
def test_pending_invalidation_mutations_are_local_after_copy(system, make, operation):
    source = _simulation(system)
    pending = ResultCacheKey("salary", MONTH)
    other_period = ResultCacheKey("salary", PREVIOUS_MONTH)
    added = ResultCacheKey("income_tax", MONTH)
    initial = {pending, other_period}
    for key in initial:
        source.invalidate_cache_entry(*key)

    copy = make(source)
    assert copy.result_cache.invalidated == initial

    if operation == "invalidate-source":
        source.invalidate_cache_entry(*added)
        assert source.result_cache.invalidated == initial | {added}
        assert copy.result_cache.invalidated == initial
    elif operation == "invalidate-copy":
        copy.invalidate_cache_entry(*added)
        assert copy.result_cache.invalidated == initial | {added}
        assert source.result_cache.invalidated == initial
    elif operation == "consume-source":
        assert source.result_cache.take_invalidated() == initial
        assert source.result_cache.take_invalidated() == set()
        assert copy.result_cache.invalidated == initial
    elif operation == "consume-copy":
        assert copy.result_cache.take_invalidated() == initial
        assert copy.result_cache.take_invalidated() == set()
        assert source.result_cache.invalidated == initial
    else:
        copy.result_cache.discard_invalidated(pending)
        assert copy.result_cache.invalidated == {other_period}
        assert source.result_cache.invalidated == initial
