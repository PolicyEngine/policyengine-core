"""A stale caller must not disable caching for unrelated nested formulas."""

from collections import Counter
import contextvars
import threading

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.simulations import SimulationBuilder
import policyengine_core.simulations.simulation as simulation_module
from policyengine_core.variables import Variable


def _variable(name, definition_period, formula=None):
    attributes = {
        "value_type": float,
        "entity": entities.Person,
        "definition_period": definition_period,
        "label": name,
    }
    if formula is not None:
        attributes["formula"] = formula
    return type(name, (Variable,), attributes)


def _simulation(definition_period, formulas):
    system = CountryTaxBenefitSystem()
    system.add_variables(
        _variable("sf_source", definition_period),
        *[
            _variable(name, definition_period, formula)
            for name, formula in formulas.items()
        ],
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    input_periods = (
        ["2020"]
        if definition_period == periods.YEAR
        else [f"2020-{month:02d}" for month in range(1, 13)]
    )
    for input_period in input_periods:
        simulation.set_input("sf_source", input_period, np.array([1.0]))
    return simulation


def _in_thread(call):
    returned = []
    failures = []

    def target():
        try:
            returned.append(call())
        except BaseException as error:
            failures.append(error)

    context = contextvars.Context()
    thread = threading.Thread(target=context.run, args=(target,))
    thread.start()
    thread.join()
    if failures:
        raise failures[0]
    return returned[0]


@pytest.mark.parametrize("definition_period", [periods.YEAR, periods.MONTH])
def test_unrelated_suffix_caches_while_its_stale_caller_retries(definition_period):
    calls = Counter()

    def doubled(person, period):
        return person("sf_source", period) * 2

    def reader(person, period):
        calls["reader"] += 1
        branch = person.simulation.get_branch("worker")
        earlier = branch.calculate("sf_doubled", period)
        if np.any(earlier != 6):
            branch.set_input("sf_source", period, np.array([3.0]))
        return earlier

    def independent(person, period):
        calls["independent"] += 1
        return person("sf_source", period) + 10

    def outer(person, period):
        calls["outer"] += 1
        earlier = person("sf_reader", period)
        # The reader made the caller stale. This value only uses the
        # caller's own unchanged input, so it can still be kept.
        return earlier + person("sf_independent", period)

    simulation = _simulation(
        definition_period,
        {
            "sf_doubled": doubled,
            "sf_reader": reader,
            "sf_independent": independent,
            "sf_outer": outer,
        },
    )
    count = 1 if definition_period == periods.YEAR else 12
    calculate = (
        simulation.calculate
        if definition_period == periods.YEAR
        else simulation.calculate_add
    )
    result = calculate("sf_outer", "2020")
    # A new calculation with the final branch input first returns 6 + 11
    # per period. The dependent reader and caller must both run again.
    np.testing.assert_array_equal(result, [count * 17.0])
    assert calls == {"outer": count * 2, "reader": count * 2, "independent": count}
    np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)

    completed = calls.copy()
    np.testing.assert_array_equal(calculate("sf_outer", "2020"), result)
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
    assert calls == completed


def test_ordered_local_credits_cache_shared_dependencies_during_a_stale_attempt():
    calls = Counter()

    def credit_formula(index):
        def formula(person, period):
            calls[f"credit_{index}"] += 1
            return person("sf_source", period) + sum(
                person(f"sf_credit_{previous}", period) for previous in range(index)
            )

        return formula

    def outer(person, period):
        calls["outer"] += 1
        worker = person.simulation.get_branch("worker")
        previous = worker.calculate("sf_source", period)
        if np.any(previous != 3):
            worker.set_input("sf_source", period, np.array([3.0]))
        # Every credit reads all its predecessors. Reusing those local
        # values keeps this dependency graph from expanding on stale retries.
        return previous + person("sf_credit_5", period)

    simulation = _simulation(
        periods.YEAR,
        {
            **{f"sf_credit_{index}": credit_formula(index) for index in range(6)},
            "sf_outer": outer,
        },
    )
    result = simulation.calculate("sf_outer", "2020")
    np.testing.assert_array_equal(result, [35.0])
    assert calls == {"outer": 2, **{f"credit_{index}": 1 for index in range(6)}}
    np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)
    completed = calls.copy()
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
    np.testing.assert_array_equal(simulation.calculate("sf_credit_5", "2020"), [32.0])
    assert calls == completed


@pytest.mark.parametrize("depth", [1, 6, 12, 20])
@pytest.mark.parametrize("mediated_foreign_read", [False, True])
@pytest.mark.parametrize("replace_after_credits", [False, True])
def test_foreign_read_credits_take_linear_work_during_a_stale_attempt(
    depth,
    mediated_foreign_read,
    replace_after_credits,
):
    calls = Counter()

    def foreign_source(person, period):
        calls["foreign_source"] += 1
        return person.simulation.get_branch("worker").calculate("sf_source", period)

    def credit_formula(index):
        def formula(person, period):
            calls[f"credit_{index}"] += 1
            source = (
                person("sf_foreign_source", period)
                if mediated_foreign_read
                else person.simulation.get_branch("worker").calculate(
                    "sf_source", period
                )
            )
            return source + sum(
                person(f"sf_credit_{previous}", period) for previous in range(index)
            )

        return formula

    def outer(person, period):
        calls["outer"] += 1
        worker = person.simulation.get_branch("worker")
        previous = worker.calculate("sf_source", period)
        first = np.any(previous == 1)
        if first:
            worker.set_input(
                "sf_source",
                period,
                np.array([2.0 if replace_after_credits else 3.0]),
            )
        # Every credit reads all its predecessors. Refusing all foreign-read
        # reuse in this stale attempt expands depth n into 2**(n - 1) calls.
        result = person(f"sf_credit_{depth - 1}", period)
        if first and replace_after_credits:
            np.testing.assert_array_equal(result, [2.0 * 2 ** (depth - 1)])
            worker.set_input("sf_source", period, np.array([3.0]))
            # Reused values must retain their foreign dependencies, including
            # dependencies reached through a locally calculated source.
            result = person(f"sf_credit_{depth - 1}", period)
        np.testing.assert_array_equal(result, [3.0 * 2 ** (depth - 1)])
        return result

    simulation = _simulation(
        periods.YEAR,
        {
            "sf_foreign_source": foreign_source,
            **{f"sf_credit_{index}": credit_formula(index) for index in range(depth)},
            "sf_outer": outer,
        },
    )
    expected = [3.0 * 2 ** (depth - 1)]
    result = simulation.calculate("sf_outer", "2020")
    np.testing.assert_array_equal(result, expected)
    assert calls["outer"] == 2
    # At most one evaluation per credit and settled input state in each
    # attempt: the stale graph, an optional changed graph, and the clean retry.
    graphs = 3 if replace_after_credits else 2
    credit_calls = sum(calls[f"credit_{index}"] for index in range(depth))
    assert credit_calls <= graphs * depth
    assert max(calls[f"credit_{index}"] for index in range(depth)) <= graphs
    assert calls["foreign_source"] <= (graphs if mediated_foreign_read else 0)
    np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)

    completed = calls.copy()
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), expected)
    np.testing.assert_array_equal(
        simulation.calculate(f"sf_credit_{depth - 1}", "2020"), expected
    )
    assert calls == completed


@pytest.mark.parametrize("mutation_in_thread", [False, True])
@pytest.mark.parametrize("raw_cache_write", [False, True])
def test_stale_attempt_reuse_observes_raw_holder_changes_between_siblings(
    mutation_in_thread,
    raw_cache_write,
):
    calls = Counter()
    expected = [4.0 if raw_cache_write else 1.0]

    def suffix(person, period):
        calls["suffix"] += 1
        return person.simulation.get_branch("worker").calculate("sf_source", period)

    def outer(person, period):
        calls["outer"] += 1
        worker = person.simulation.get_branch("worker")
        worker.calculate("sf_source", period)
        worker.set_input("sf_source", period, np.array([2.0]))
        np.testing.assert_array_equal(person("sf_suffix", period), [2.0])
        counters = worker._input_epoch, worker._inputs_set

        def change():
            holder = worker.get_holder("sf_source")
            if raw_cache_write:
                holder.put_in_cache(np.array([4.0]), period, worker.branch_name)
            else:
                # Removing only the worker override exposes its inherited 1.
                holder.delete_arrays(period, worker.branch_name)

        if mutation_in_thread:
            # The memo-producing child has exited. Context-free activity
            # must invalidate its reuse even with no restricted frame open.
            assert not simulation_module._restricted_frames
            _in_thread(change)
        else:
            change()
        assert (worker._input_epoch, worker._inputs_set) == counters
        result = person("sf_suffix", period)
        np.testing.assert_array_equal(result, expected)
        return result

    simulation = _simulation(periods.YEAR, {"sf_suffix": suffix, "sf_outer": outer})
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), expected)
    # Context-free effects can exhaust the retry budget; every attempt must
    # nevertheless recalculate the suffix after the counter-preserving change.
    assert 2 <= calls["outer"] <= simulation_module._RERUNS_AFTER_INPUT_CHANGE + 1
    assert calls["suffix"] == 2 * calls["outer"]


def test_stale_attempt_reuse_revalidates_foreign_branch_registration():
    calls = Counter()
    replacement = None

    def suffix(person, period):
        calls["suffix"] += 1
        return person.simulation.get_branch("worker").calculate("sf_source", period)

    def outer(person, period):
        calls["outer"] += 1
        trigger = person.simulation.get_branch("trigger")
        previous = trigger.calculate("sf_source", period)
        first = np.any(previous == 1)
        if first:
            trigger.set_input("sf_source", period, np.array([3.0]))
            np.testing.assert_array_equal(person("sf_suffix", period), [1.0])
            original = person.simulation.get_branch("worker")
            assert (original._input_epoch, original._inputs_set) == (
                replacement._input_epoch,
                replacement._inputs_set,
            )
            # Replacing a registration does not change the old snapshot's
            # counters. A memoized named read must still follow the new branch.
            person.simulation.branches["worker"] = replacement
        result = person("sf_suffix", period)
        np.testing.assert_array_equal(result, [4.0])
        return result

    simulation = _simulation(periods.YEAR, {"sf_suffix": suffix, "sf_outer": outer})
    original = simulation.get_branch("worker")
    replacement = original.clone()
    replacement.get_holder("sf_source").put_in_cache(
        np.array([4.0]), periods.period("2020"), replacement.branch_name
    )
    assert (original._input_epoch, original._inputs_set) == (
        replacement._input_epoch,
        replacement._inputs_set,
    )

    result = simulation.calculate("sf_outer", "2020")
    np.testing.assert_array_equal(result, [4.0])
    assert calls == {"outer": 2, "suffix": 3}
    np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)
    completed = calls.copy()
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
    assert calls == completed


def test_stale_attempt_reuse_keeps_direct_foreign_calculations_in_their_simulation():
    calls = Counter()

    def value(person, period):
        calls[f"value_{person.simulation.branch_name}"] += 1
        return person("sf_source", period) * 10

    def suffix(person, period):
        calls["suffix"] += 1
        # Custom input handlers can call _calculate without opening a frame
        # in that simulation. Its result still belongs to the foreign worker.
        return person.simulation.get_branch("worker")._calculate("sf_value", period)

    def outer(person, period):
        calls["outer"] += 1
        worker = person.simulation.get_branch("worker")
        previous = worker.calculate("sf_source", period)
        if np.any(previous != 3):
            worker.set_input("sf_source", period, np.array([3.0]))
        foreign = person("sf_suffix", period)
        local = person("sf_value", period)
        # Assert inside the stale attempt so its later retry cannot hide a
        # worker result incorrectly memoized under the root's variable name.
        np.testing.assert_array_equal(foreign, [30.0])
        np.testing.assert_array_equal(local, [10.0])
        return foreign + local

    simulation = _simulation(
        periods.YEAR,
        {"sf_value": value, "sf_suffix": suffix, "sf_outer": outer},
    )
    simulation.get_branch("worker")
    result = simulation.calculate("sf_outer", "2020")
    np.testing.assert_array_equal(result, [40.0])
    assert calls == {"outer": 2, "suffix": 2, "value_worker": 2, "value_default": 1}
    completed = calls.copy()
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
    assert calls == completed


@pytest.mark.parametrize("suffix_branch", ["worker", "other"])
def test_foreign_suffix_is_not_cached_before_its_stale_caller_settles(suffix_branch):
    calls = Counter()

    def suffix(person, period):
        calls["suffix"] += 1
        return person.simulation.get_branch(suffix_branch).calculate(
            "sf_source", period
        )

    def outer(person, period):
        calls["outer"] += 1
        worker = person.simulation.get_branch("worker")
        previous = worker.calculate("sf_source", period)
        first = np.any(previous == 1)
        if first:
            worker.set_input("sf_source", period, np.array([2.0]))
        # A new frame for this suffix is locally clean, but its caller
        # already read the replaced worker input. Keeping the suffix now
        # would hide the later mutation from the caller's next attempt.
        result = person("sf_suffix", period)
        if first:
            person.simulation.get_branch(suffix_branch).set_input(
                "sf_source", period, np.array([3.0])
            )
            worker.set_input("sf_source", period, np.array([3.0]))
        return result

    simulation = _simulation(periods.YEAR, {"sf_suffix": suffix, "sf_outer": outer})
    result = simulation.calculate("sf_outer", "2020")
    np.testing.assert_array_equal(result, [3.0])
    assert calls == {"outer": 2, "suffix": 2}
    np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)

    completed = calls.copy()
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
    assert calls == completed


@pytest.mark.parametrize("write_in_thread", [False, True])
@pytest.mark.parametrize("worker_family", ["own", "unrelated"])
def test_foreign_writing_suffix_repeats_its_effects_while_independent_work_caches(
    write_in_thread,
    worker_family,
):
    calls = Counter()
    unrelated_worker = (
        _simulation(periods.YEAR, {}).get_branch("worker")
        if worker_family == "unrelated"
        else None
    )

    def get_worker(simulation):
        return (
            unrelated_worker
            if unrelated_worker is not None
            else simulation.get_branch("worker")
        )

    def independent(person, period):
        calls["independent"] += 1
        return person("sf_source", period) + 10

    def suffix(person, period):
        calls["suffix"] += 1
        worker = get_worker(person.simulation)

        def change():
            worker.set_input("sf_source", period, np.array([3.0]))

        if write_in_thread:
            _in_thread(change)
        else:
            change()
        # This formula writes a foreign input without reading a foreign
        # value. Its effect must still repeat on the caller's retry, while
        # the independent calculation beneath it can safely stay cached.
        return person("sf_independent", period)

    def outer(person, period):
        calls["outer"] += 1
        worker = get_worker(person.simulation)
        previous = worker.calculate("sf_source", period)
        worker.set_input("sf_source", period, np.array([2.0]))
        return previous + person("sf_suffix", period)

    simulation = _simulation(
        periods.YEAR,
        {
            "sf_independent": independent,
            "sf_suffix": suffix,
            "sf_outer": outer,
        },
    )
    attempts = (
        simulation_module._RERUNS_AFTER_INPUT_CHANGE + 1 if write_in_thread else 3
    )
    result = simulation.calculate("sf_outer", "2020")
    np.testing.assert_array_equal(result, [14.0])
    assert calls == {"outer": attempts, "suffix": attempts, "independent": 1}
    np.testing.assert_array_equal(
        get_worker(simulation).calculate("sf_source", "2020"), [3.0]
    )
    if write_in_thread:
        # A context-free write cannot certify convergence even though each
        # transition ends at the same input. The original shared-frame rule
        # exhausts the budget and leaves the result unkept in this case.
        assert simulation.get_array("sf_outer", "2020") is None
        np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
        assert calls == {
            "outer": attempts * 2,
            "suffix": attempts * 2,
            "independent": 1,
        }
    else:
        # The first attempt returns the old input; the next two repeat the
        # same complete 2 -> 3 transition and result and prove a fixed point.
        np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)
        completed = calls.copy()
        np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
        assert calls == completed


def test_branch_creating_suffix_recreates_registration_while_independent_work_caches():
    def run(input_first):
        calls = Counter()
        created = []

        def independent(person, period):
            calls["independent"] += 1
            return person("sf_source", period) + 10

        def suffix(person, period):
            calls["suffix"] += 1
            # Creating a branch is the only foreign effect here: no
            # foreign value is read and no input is changed by this suffix.
            created.append(person.simulation.get_branch("scratch"))
            return person("sf_independent", period)

        def outer(person, period):
            calls["outer"] += 1
            worker = person.simulation.get_branch("worker")
            previous = worker.calculate("sf_source", period)
            worker.set_input("sf_source", period, np.array([3.0]))
            result = previous + person("sf_suffix", period)
            # Each attempt requires the suffix's creation, even after an
            # earlier attempt removed its registration. A cached suffix
            # would omit that effect and leave no branch to remove here.
            assert person.simulation.branches.pop("scratch") is created[-1]
            return result

        simulation = _simulation(
            periods.YEAR,
            {
                "sf_independent": independent,
                "sf_suffix": suffix,
                "sf_outer": outer,
            },
        )
        if input_first:
            simulation.get_branch("worker").set_input(
                "sf_source", "2020", np.array([3.0])
            )
        result = simulation.calculate("sf_outer", "2020")
        attempts = 1 if input_first else 2
        assert calls == {"outer": attempts, "suffix": attempts, "independent": 1}
        assert len(created) == attempts
        if not input_first:
            assert created[0] is not created[1]
        np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)
        completed = calls.copy()
        np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
        assert calls == completed
        return result

    np.testing.assert_array_equal(run(input_first=False), run(input_first=True))


@pytest.mark.parametrize("prime_fast_cache", [False, True])
def test_unrelated_thread_read_suffix_is_read_again_after_its_caller_changes_input(
    prime_fast_cache,
):
    calls = Counter()
    unrelated_worker = _simulation(
        periods.YEAR,
        {"sf_doubled": lambda person, period: person("sf_source", period) * 2},
    ).get_branch("worker")
    if prime_fast_cache:
        unrelated_worker.calculate("sf_doubled", "2020")

    def suffix(person, period):
        calls["suffix"] += 1
        # A context-free read from another family cannot reach this frame
        # through ancestry, so it must still refuse the optional cache.
        return _in_thread(lambda: unrelated_worker.calculate("sf_doubled", period))

    def outer(person, period):
        calls["outer"] += 1
        worker = person.simulation.get_branch("worker")
        previous = worker.calculate("sf_source", period)
        first = np.any(previous == 1)
        if first:
            worker.set_input("sf_source", period, np.array([2.0]))
        result = person("sf_suffix", period)
        if first:
            unrelated_worker.set_input("sf_source", period, np.array([3.0]))
            worker.set_input("sf_source", period, np.array([3.0]))
        return result

    simulation = _simulation(periods.YEAR, {"sf_suffix": suffix, "sf_outer": outer})
    result = simulation.calculate("sf_outer", "2020")
    np.testing.assert_array_equal(result, [6.0])
    assert calls == {"outer": 2, "suffix": 2}
    np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)
    completed = calls.copy()
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
    assert calls == completed


@pytest.mark.parametrize("raw_cache_write", [False, True])
def test_unrelated_unread_thread_mutation_does_not_prevent_caller_convergence(
    raw_cache_write,
):
    calls = Counter()
    unrelated_worker = _simulation(periods.YEAR, {}).get_branch("worker")

    def independent(person, period):
        calls["independent"] += 1
        return person("sf_source", period) + 10

    def suffix(person, period):
        calls["suffix"] += 1

        def change():
            if raw_cache_write:
                unrelated_worker.get_holder("sf_source").put_in_cache(
                    np.array([5.0]), period, unrelated_worker.branch_name
                )
            else:
                unrelated_worker.delete_arrays("sf_source")
                unrelated_worker.set_input("sf_source", period, np.array([5.0]))

        _in_thread(change)
        return person("sf_independent", period)

    def outer(person, period):
        calls["outer"] += 1
        worker = person.simulation.get_branch("worker")
        previous = worker.calculate("sf_source", period)
        worker.set_input("sf_source", period, np.array([2.0]))
        result = person("sf_suffix", period)
        worker.set_input("sf_source", period, np.array([3.0]))
        return previous + result

    simulation = _simulation(
        periods.YEAR,
        {
            "sf_independent": independent,
            "sf_suffix": suffix,
            "sf_outer": outer,
        },
    )
    result = simulation.calculate("sf_outer", "2020")
    np.testing.assert_array_equal(result, [14.0])
    # The optional suffix cache cannot skip its threaded effects, but the
    # caller's observed 2 -> 3 transition still settles on the third attempt.
    # Unread changes in another family do not taint that convergence proof.
    assert calls == {"outer": 3, "suffix": 3, "independent": 1}
    if raw_cache_write:
        # A raw cache write is observable activity, but is still excluded
        # from the record of supplied inputs used for replay and export.
        assert (
            "sf_source",
            unrelated_worker.branch_name,
            periods.period("2020"),
        ) not in unrelated_worker._user_input_keys
    np.testing.assert_array_equal(simulation.get_array("sf_outer", "2020"), result)
    completed = calls.copy()
    np.testing.assert_array_equal(simulation.calculate("sf_outer", "2020"), result)
    assert calls == completed
