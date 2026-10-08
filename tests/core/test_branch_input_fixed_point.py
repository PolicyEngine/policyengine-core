"""Repeated branch resets settle without forgetting real input changes.

The fixed-point result must match a fresh branch given its override before
calculating. Equal results alone do not prove convergence: the branch's input
mutations must also settle, and changes outside the formula's context cannot
justify keeping a stale result.
"""

from __future__ import annotations

import contextvars
import threading

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.experimental import MemoryConfig
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


def _in_thread(call):
    values = []
    errors = []

    def target():
        try:
            values.append(call())
        except BaseException as error:
            errors.append(error)

    context = contextvars.Context()
    thread = threading.Thread(target=context.run, args=(target,))
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return values[0]


def _simulation(
    values,
    definition_period=periods.YEAR,
    delete_derived=False,
    alternating=False,
    own_input=False,
    thread_operation=None,
    masked_inputs=False,
    recreate_branch=False,
    cache_mode="memory",
):
    values = np.asarray(values, dtype=np.float32)
    calls = []

    def base(person, period):
        source = "fp_thread_source" if thread_operation == "before_read" else "fp_x"
        return np.abs(person(source, period)) / 2 + 5

    def cap(person, period):
        return np.full(person.count, 1000.0)

    def capped(person, period):
        return np.minimum(person("fp_x", period), person("fp_cap", period))

    def outer(person, period):
        calls.append(str(period))
        simulation = person.simulation
        # The branch inherits a record of the input it will replace.
        person("fp_cap", period)
        branch = simulation.get_branch("persistent")
        branch.delete_arrays("fp_cap")
        if delete_derived:
            branch.delete_arrays("fp_capped")
        if thread_operation == "before_read":
            _in_thread(lambda: branch.set_input("fp_thread_source", period, values + 1))

        def read_base():
            return branch.calculate("fp_base", period)

        maximum = _in_thread(read_base) if thread_operation == "read" else read_base()
        if alternating:
            maximum = np.full(person.count, 1.0 + len(calls) % 2)
        if masked_inputs:
            maximum = np.ma.array(np.ones(person.count), mask=bool(len(calls) % 2))

        def change():
            return branch.set_input("fp_cap", period, maximum)

        if thread_operation == "change":
            _in_thread(change)
        else:
            change()
        result = branch.calculate("fp_capped", period)
        if masked_inputs:
            result = np.asarray(result) * 0
        if recreate_branch:
            del simulation.branches["persistent"]
        if own_input:
            simulation.set_input("fp_outer", period, np.full(person.count, 9.0))
        return result

    system = CountryTaxBenefitSystem()
    system.add_variables(
        _variable("fp_x", definition_period),
        _variable("fp_thread_source", definition_period),
        _variable("fp_base", definition_period, base),
        _variable("fp_cap", definition_period, cap),
        _variable("fp_capped", definition_period, capped),
        _variable("fp_outer", definition_period, outer),
    )
    simulation = SimulationBuilder().build_default_simulation(system, count=len(values))
    if cache_mode == "disk":
        simulation.memory_config = MemoryConfig(0)
    elif cache_mode == "drop":
        simulation.memory_config = MemoryConfig(1, variables_to_drop=["fp_outer"])
    elif cache_mode == "blacklist":
        system.cache_blacklist = ["fp_outer"]
        simulation.opt_out_cache = True
    input_periods = (
        ["2020"]
        if definition_period == periods.YEAR
        else [f"2020-{month:02d}" for month in range(1, 13)]
    )
    for input_period in input_periods:
        simulation.set_input("fp_x", input_period, values)
        if thread_operation == "before_read":
            # Its history follows cap's, so writing it preserves the cap
            # dependency that the formula's later contextual set invalidates.
            simulation.calculate("fp_cap", input_period)
            simulation.set_input("fp_thread_source", input_period, values)
    return simulation, calls


def _input_first_result(values):
    fresh, _ = _simulation(values)
    maximum = fresh.calculate("fp_base", "2020")
    branch = fresh.get_branch("input_first")
    branch.set_input("fp_cap", "2020", maximum)
    return branch.calculate("fp_capped", "2020")


def _assert_settled(simulation, calls, expected, kept_in_holder=True):
    result = simulation.calculate("fp_outer", "2020")
    np.testing.assert_array_equal(result, expected)
    assert 1 < len(calls) <= 3
    stored = simulation.get_array("fp_outer", "2020")
    if kept_in_holder:
        np.testing.assert_array_equal(stored, expected)
    else:
        assert stored is None
    completed = len(calls)
    np.testing.assert_array_equal(simulation.calculate("fp_outer", "2020"), expected)
    assert len(calls) == completed


@pytest.mark.parametrize("cache_mode", ["disk", "drop", "blacklist"])
def test_fixed_point_respects_holder_storage_restrictions(cache_mode):
    values = [0.0, 40.0]
    simulation, calls = _simulation(values, cache_mode=cache_mode)
    _assert_settled(
        simulation,
        calls,
        _input_first_result(values),
        kept_in_holder=cache_mode == "disk",
    )


@pytest.mark.parametrize("delete_derived", [False, True])
@pytest.mark.parametrize("recreate_branch", [False, True])
def test_branch_delete_read_set_reaches_a_cached_fixed_point(
    delete_derived, recreate_branch
):
    values = [-20.0, 0.0, 40.0]
    simulation, calls = _simulation(
        values, delete_derived=delete_derived, recreate_branch=recreate_branch
    )
    _assert_settled(simulation, calls, _input_first_result(values))


def test_repeated_result_with_alternating_inputs_is_not_a_fixed_point():
    # Both caps produce zero, but the mutations differ on every attempt.
    simulation, calls = _simulation([0.0], alternating=True)
    budget = simulation_module._RERUNS_AFTER_INPUT_CHANGE + 1
    for calculation in (1, 2):
        assert simulation.calculate("fp_outer", "2020").tolist() == [0.0]
        assert len(calls) == calculation * budget
        assert simulation.get_array("fp_outer", "2020") is None


def test_masked_inputs_cannot_certify_a_fixed_point_from_backing_bytes():
    simulation, calls = _simulation([0.0], masked_inputs=True)
    assert simulation.calculate("fp_outer", "2020").tolist() == [0.0]
    assert len(calls) == simulation_module._RERUNS_AFTER_INPUT_CHANGE + 1
    assert simulation.get_array("fp_outer", "2020") is None


@pytest.mark.parametrize("thread_operation", ["read", "change"])
def test_untracked_thread_operation_cannot_certify_a_fixed_point(thread_operation):
    simulation, calls = _simulation([40.0], thread_operation=thread_operation)
    np.testing.assert_array_equal(
        simulation.calculate("fp_outer", "2020"), _input_first_result([40.0])
    )
    assert len(calls) == simulation_module._RERUNS_AFTER_INPUT_CHANGE + 1
    assert simulation.get_array("fp_outer", "2020") is None


def test_untracked_thread_write_before_first_read_cannot_certify_a_fixed_point():
    # The write's epoch precedes every read, so checking intervening drops
    # alone cannot establish that the formula observed every input mutation.
    simulation, calls = _simulation([40.0], thread_operation="before_read")
    np.testing.assert_array_equal(
        simulation.calculate("fp_outer", "2020"), _input_first_result([41.0])
    )
    assert len(calls) == simulation_module._RERUNS_AFTER_INPUT_CHANGE + 1
    assert simulation.get_array("fp_outer", "2020") is None


def test_distinct_clones_with_the_same_branch_path_do_not_prove_a_fixed_point():
    # A direct branch clone retains its parent's branch name and ancestry.
    # One indistinguishable set(path, x, 0) per run changes a different clone:
    # the first two results equal 1, but the third becomes 0.
    clones = []
    calls = []

    def outer(person, period):
        calls.append(str(period))
        previous = [clone.calculate("fp_clone_x", period) for clone in clones]
        largest = int(np.argmax([value[0] for value in previous]))
        clones[largest].set_input("fp_clone_x", period, np.zeros(person.count))
        return clones[0].calculate("fp_clone_x", period)

    system = CountryTaxBenefitSystem()
    system.add_variables(
        _variable("fp_clone_x", periods.YEAR),
        _variable("fp_clone_outer", periods.YEAR, outer),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.set_input("fp_clone_x", "2020", np.array([0.0]))
    original = simulation.get_branch("same_path")
    clones.extend([original, original.clone(), original.clone()])
    for clone, value in zip(clones, (1.0, 2.0, 3.0)):
        clone.set_input("fp_clone_x", "2020", np.array([value]))

    assert simulation.calculate("fp_clone_outer", "2020").tolist() == [0.0]
    assert len(calls) == 4
    assert simulation.get_array("fp_clone_outer", "2020").tolist() == [0.0]
    assert simulation.calculate("fp_clone_outer", "2020").tolist() == [0.0]
    assert len(calls) == 4


def test_rotating_same_path_clones_do_not_prove_a_fixed_point_between_attempts():
    # Only one clone is read and changed per attempt. The next clone comes
    # from an input, so the first two identical set(path, x, 0) transitions
    # and results must not hide the third clone's different state.
    clones = []
    calls = []

    def outer(person, period):
        calls.append(str(period))
        simulation = person.simulation
        branch = simulation.get_branch("same_path")
        previous = branch.calculate("fp_rotate_x", period)
        next_clone = int(branch.calculate("fp_rotate_next", period)[0])
        branch.set_input("fp_rotate_x", period, np.zeros(person.count))
        simulation.branches["same_path"] = clones[next_clone]
        return np.minimum(previous, 1)

    system = CountryTaxBenefitSystem()
    system.add_variables(
        _variable("fp_rotate_x", periods.YEAR),
        _variable("fp_rotate_next", periods.YEAR),
        _variable("fp_rotate_outer", periods.YEAR, outer),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.set_input("fp_rotate_x", "2020", np.array([0.0]))
    original = simulation.get_branch("same_path")
    clones.extend([original, original.clone(), original.clone()])
    for next_clone, (clone, value) in enumerate(zip(clones, (1.0, 2.0, 0.0)), 1):
        clone.set_input("fp_rotate_x", "2020", np.array([value]))
        clone.set_input("fp_rotate_next", "2020", np.array([next_clone % 3]))

    assert simulation.calculate("fp_rotate_outer", "2020").tolist() == [0.0]
    assert len(calls) == 3
    assert simulation.get_array("fp_rotate_outer", "2020").tolist() == [0.0]
    assert simulation.calculate("fp_rotate_outer", "2020").tolist() == [0.0]
    assert len(calls) == 3


def test_own_period_input_wins_over_a_branch_reset_formula_result():
    simulation, calls = _simulation([40.0], own_input=True)
    assert simulation.calculate("fp_outer", "2020").tolist() == [9.0]
    holder = simulation.get_holder("fp_outer")
    assert holder._is_input(periods.period("2020"), "default")
    assert holder.get_array("2020").tolist() == [9.0]
    completed = len(calls)
    assert simulation.calculate("fp_outer", "2020").tolist() == [9.0]
    assert len(calls) == completed


def test_calculate_add_keeps_a_sum_after_persistent_branch_resets_settle():
    values = [0.0, 40.0]
    simulation, calls = _simulation(values, definition_period=periods.MONTH)
    expected = 12 * _input_first_result(values)
    np.testing.assert_array_equal(
        simulation.calculate_add("fp_outer", "2020"), expected
    )
    assert 12 < len(calls) <= 36
    np.testing.assert_array_equal(simulation.get_array("fp_outer", "2020"), expected)
    completed = len(calls)
    np.testing.assert_array_equal(simulation.calculate("fp_outer", "2020"), expected)
    assert len(calls) == completed
