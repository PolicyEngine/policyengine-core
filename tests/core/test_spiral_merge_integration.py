"""Recursion scheduling preserves input invalidation and option-cache rules."""

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template.entities import Person
from policyengine_core.variables import QuantityType, Variable
from tests.fixtures.spirals import build


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("store", [True, False])
def test_finished_deeper_periods_follow_an_input_changed_by_their_caller(traced, store):
    class source(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Input used by the recursive calculation"

    class recursive(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Anchored recurrence over a mutable input"

        def formula_2010(person, period):
            return person("recursive", period.last_year) + person("source", "2010")

    class caller(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Caller that replaces the recursive calculation's input"

        def formula(person, period):
            result = person("recursive", period)
            # The outer calculation must run again with this input. Deeper
            # periods completed by the scheduler must run again as well.
            person.simulation.set_input("source", "2010", np.array([2.0]))
            return result

    simulation = build([source, recursive, caller], {("source", 2010): 1}, 1)
    simulation.trace = traced
    branch = simulation.get_branch("changed")
    if not store:
        for name in ("recursive", "caller"):
            branch.get_holder(name)._do_not_store = True

    fresh = build([source, recursive, caller], {("source", 2010): 2}, 1)
    fresh.trace = traced
    expected = fresh.calculate("caller", "2024")

    np.testing.assert_array_equal(expected, [30.0])
    np.testing.assert_array_equal(branch.calculate("caller", "2024"), expected)
    assert branch.tracer.stack == []
    assert branch._calculations_in_flight == 0


@pytest.mark.parametrize("traced", [False, True])
def test_finished_deeper_periods_follow_a_foreign_input_change(traced):
    class source(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Input read in another simulation"

    class recursive(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Anchored recurrence with a foreign input"

        def formula_2010(person, period):
            inputs = person.simulation.get_branch("inputs")
            return person("recursive", period.last_year) + inputs.calculate(
                "source", "2010"
            )

    class caller(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Caller that replaces a foreign input"

        def formula(person, period):
            result = person("recursive", period)
            person.simulation.get_branch("inputs").set_input(
                "source", "2010", np.array([2.0])
            )
            return result

    simulation = build([source, recursive, caller], {("source", 2010): 1}, 1)
    simulation.trace = traced
    simulation.get_branch("inputs")
    # Isolate the scheduler's temporary values. Persistent values from
    # already-ended calculations intentionally survive foreign mutations.
    del simulation._fast_cache
    for name in ("recursive", "caller"):
        simulation.get_holder(name)._do_not_store = True

    np.testing.assert_array_equal(simulation.calculate("caller", "2024"), [30.0])
    assert simulation.tracer.stack == []
    assert simulation._calculations_in_flight == 0


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("settles", [True, False])
def test_repeated_input_changes_settle_or_stop_without_reusing_stale_values(
    traced, settles
):
    calls = []

    class source(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Input changed on successive attempts"

    class recursive(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Anchored recurrence whose input changes repeatedly"

        def formula_2010(person, period):
            return person("recursive", period.last_year) + person("source", "2010")

    class caller(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Caller that repeatedly changes a recursive input"

        def formula(person, period):
            calls.append(str(period))
            # Fail promptly if reopening completed keys bypasses the
            # existing bound on retries after input changes.
            assert len(calls) <= 200, "Input changes kept reopening deferred periods"
            result = person("recursive", period)
            old = person("source", "2010")
            new = np.minimum(old + 1, 7) if settles else 3 - old
            person.simulation.set_input("source", "2010", new)
            return result

    simulation = build([source, recursive, caller], {("source", 2010): 1}, 1)
    simulation.trace = traced
    branch = simulation.get_branch("changed")

    result = branch.calculate("caller", "2024")

    if settles:
        np.testing.assert_array_equal(result, [105.0])
        np.testing.assert_array_equal(branch.calculate("recursive", "2024"), [105.0])
    else:
        # A continuously changing input has no stable numerical answer.
        # The retry limit still returns a finite result and keeps no caller
        # result calculated before its last input change.
        assert np.all(np.isfinite(result))
        assert branch.get_array("caller", "2024") is None
    assert branch.tracer.stack == []
    assert branch._calculations_in_flight == 0


@pytest.mark.parametrize("traced", [False, True])
def test_an_unsettled_deferred_period_is_never_published_as_finished(traced):
    calls = []

    class source(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Input continuously changed inside a deferred calculation"

    class recursive(Variable):
        value_type = float
        entity = Person
        definition_period = periods.YEAR
        label = "Recurrence with an unsettled deeper period"

        def formula_2010(person, period):
            if period.start.year == 2023:
                calls.append(str(period))
                assert len(calls) <= 200, "An unsettled deferred period kept retrying"
                result = person("source", "2010")
                person.simulation.set_input("source", "2010", 3 - result)
                # A nested calculation observes the mutation generation
                # before the scheduler receives this still-stale result.
                person("source", "2010")
                return result
            return person("recursive", period.last_year) + 1

    simulation = build([source, recursive], {("source", 2010): 1}, 1)
    simulation.trace = traced
    branch = simulation.get_branch("changed")

    result = branch.calculate("recursive", "2024")

    assert calls
    assert np.all(np.isfinite(result))
    # The input never settles. Neither the exhausted deferred result nor
    # a caller using it may be kept as though calculated at settled inputs.
    assert branch.get_array("recursive", "2023") is None
    assert branch.get_array("recursive", "2024") is None
    assert branch.tracer.stack == []
    assert branch._calculations_in_flight == 0


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("operation", ["add", "divide"])
def test_direct_option_does_not_keep_a_value_derived_from_a_recursion_cut(
    traced, operation
):
    definition_period = periods.MONTH if operation == "add" else periods.YEAR

    class recursive(Variable):
        value_type = float
        entity = Person
        quantity_type = QuantityType.FLOW
        label = "Unanchored recurrence used through an option"

        def formula(person, period):
            previous = period.offset(-1, definition_period)
            return person("recursive", previous) + 1

    recursive.definition_period = definition_period
    used = build([recursive], max_spiral_loops=1)
    used.trace = traced
    fresh = build([recursive], max_spiral_loops=1)
    fresh.trace = traced
    target = "2020" if operation == "add" else "2020-01"
    method = used.calculate_add if operation == "add" else used.calculate_divide

    result = method("recursive", target)

    # The direct option has a different recursion depth from a plain read
    # of the converted period. Its incomplete result must not answer that
    # subsequent plain read, nor remain available for cloning or export.
    expected_option = [12.0] if operation == "add" else [1.0 / 12]
    np.testing.assert_allclose(result, expected_option)
    assert used.get_array("recursive", target) is None
    np.testing.assert_array_equal(
        used.calculate("recursive", target), fresh.calculate("recursive", target)
    )
    assert used.get_array("recursive", target) is None
    assert used.tracer.stack == []
    assert used._calculations_in_flight == 0


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("operation", ["add", "divide"])
def test_a_cut_option_keeps_an_input_set_meanwhile_at_its_requested_period(
    traced, operation
):
    definition_period = periods.MONTH if operation == "add" else periods.YEAR
    target = "2020" if operation == "add" else "2020-01"

    class recursive(Variable):
        value_type = float
        entity = Person
        quantity_type = QuantityType.FLOW
        set_input = None
        label = "Cut recurrence that supplies the requested option value"

        def formula(person, period):
            previous = period.offset(-1, definition_period)
            result = person("recursive", previous) + 1
            if period.start.year == 2020 and period.start.month == 1:
                person.simulation.set_input("recursive", target, np.array([120.0]))
            return result

    recursive.definition_period = definition_period
    simulation = build([recursive], max_spiral_loops=1)
    simulation.trace = traced
    method = (
        simulation.calculate_add if operation == "add" else simulation.calculate_divide
    )

    result = method("recursive", target)

    # Suppressing an incomplete aggregate's cache write must retain the
    # input-first rule: an input supplied for that aggregate while it ran
    # supplies its answer, even though the computed terms were cut.
    np.testing.assert_array_equal(result, [120.0])
    np.testing.assert_array_equal(simulation.get_array("recursive", target), [120.0])
    assert not simulation.get_holder("recursive").is_derived(periods.period(target))
    assert simulation.tracer.stack == []
    assert simulation._calculations_in_flight == 0
