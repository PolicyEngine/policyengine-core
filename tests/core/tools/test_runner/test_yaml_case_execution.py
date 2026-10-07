from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, field

import pytest

from policyengine_core.tools.yaml_case_execution import (
    YamlCaseExecution,
    YamlCaseExecutionError,
)


@dataclass
class FakeTracer:
    receipts: list[str] = field(default_factory=list)


@dataclass
class FakeSimulation:
    name: str
    tracer: FakeTracer = field(default_factory=FakeTracer)


class FakeSystem:
    pass


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("absent-reference", id="absent-reference"),
        pytest.param("none-reference", id="none-reference"),
        pytest.param("existing-reference", id="existing-reference"),
        pytest.param("builder-sets-reference", id="builder-sets-reference"),
        pytest.param("builder-leaves-reference", id="builder-leaves-reference"),
        pytest.param("simulation-is-returned", id="simulation-is-returned"),
        pytest.param("active-state", id="active-state"),
        pytest.param("closed-state", id="closed-state"),
    ],
)
def test_successful_execution_obeys_lifecycle(scenario: str) -> None:
    system = FakeSystem()
    previous = FakeSimulation("previous")
    if scenario == "none-reference":
        system.simulation = None
    elif scenario in {"existing-reference", "builder-leaves-reference"}:
        system.simulation = previous
    simulation = FakeSimulation("case")

    def build():
        if scenario == "builder-sets-reference":
            system.simulation = simulation
        return simulation

    execution = YamlCaseExecution(system, build)
    with execution as active:
        assert active is simulation
        assert execution.simulation is simulation
        assert execution.active is True
        assert execution.closed is False

    assert execution.active is False
    assert execution.closed is True
    if scenario == "absent-reference":
        assert not hasattr(system, "simulation")
    elif scenario in {"existing-reference", "builder-leaves-reference"}:
        assert system.simulation is previous
    elif scenario == "none-reference":
        assert system.simulation is None


@pytest.mark.parametrize(
    ("exception_type", "sets_reference"),
    [
        pytest.param(ValueError, False, id="value-before-reference"),
        pytest.param(ValueError, True, id="value-after-reference"),
        pytest.param(TypeError, False, id="type-before-reference"),
        pytest.param(TypeError, True, id="type-after-reference"),
        pytest.param(KeyError, False, id="key-before-reference"),
        pytest.param(KeyError, True, id="key-after-reference"),
        pytest.param(RuntimeError, False, id="runtime-before-reference"),
        pytest.param(RuntimeError, True, id="runtime-after-reference"),
    ],
)
def test_build_failure_restores_state(exception_type, sets_reference: bool) -> None:
    system = FakeSystem()
    previous = FakeSimulation("previous")
    system.simulation = previous
    partial = FakeSimulation("partial")

    def build():
        if sets_reference:
            system.simulation = partial
        raise exception_type("failure")

    execution = YamlCaseExecution(system, build)

    with pytest.raises(exception_type):
        with execution:
            pass

    assert system.simulation is previous
    assert execution.active is False
    assert execution.closed is True


@pytest.mark.parametrize(
    "exception_type",
    [
        pytest.param(AssertionError, id="assertion"),
        pytest.param(ValueError, id="value"),
        pytest.param(TypeError, id="type"),
        pytest.param(KeyError, id="key"),
        pytest.param(RuntimeError, id="runtime"),
        pytest.param(LookupError, id="lookup"),
        pytest.param(ArithmeticError, id="arithmetic"),
        pytest.param(EOFError, id="end-of-file"),
    ],
)
def test_body_failure_is_not_suppressed_and_still_cleans_up(exception_type) -> None:
    system = FakeSystem()
    simulation = FakeSimulation("case")
    execution = YamlCaseExecution(system, lambda: simulation)

    with pytest.raises(exception_type):
        with execution:
            system.simulation = simulation
            raise exception_type("failure")

    assert not hasattr(system, "simulation")
    assert execution.closed is True


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("read-before-enter", id="read-before-enter"),
        pytest.param("read-after-close", id="read-after-close"),
        pytest.param("enter-twice", id="enter-twice"),
        pytest.param("reenter-after-close", id="reenter-after-close"),
        pytest.param("close-before-enter", id="close-before-enter"),
        pytest.param("none-simulation", id="none-simulation"),
        pytest.param("none-system", id="none-system"),
        pytest.param("non-callable-builder", id="non-callable-builder"),
    ],
)
def test_invalid_lifecycle_operations_are_prevented(scenario: str) -> None:
    system = None if scenario == "none-system" else FakeSystem()
    builder = (
        None if scenario == "non-callable-builder" else lambda: FakeSimulation("case")
    )
    if scenario == "none-simulation":

        def builder():
            return None

    if scenario in {"none-system", "non-callable-builder"}:
        with pytest.raises(YamlCaseExecutionError):
            YamlCaseExecution(system, builder)
        return

    execution = YamlCaseExecution(system, builder)
    if scenario == "read-before-enter":
        with pytest.raises(YamlCaseExecutionError):
            _ = execution.simulation
    elif scenario == "close-before-enter":
        execution.close()
        with pytest.raises(YamlCaseExecutionError):
            execution.__enter__()
    elif scenario == "none-simulation":
        with pytest.raises(YamlCaseExecutionError):
            execution.__enter__()
        assert execution.closed is True
    else:
        execution.__enter__()
        if scenario == "enter-twice":
            with pytest.raises(YamlCaseExecutionError):
                execution.__enter__()
            execution.close()
        else:
            execution.close()
            if scenario == "read-after-close":
                with pytest.raises(YamlCaseExecutionError):
                    _ = execution.simulation
            elif scenario == "reenter-after-close":
                with pytest.raises(YamlCaseExecutionError):
                    execution.__enter__()


@pytest.mark.parametrize(
    "retained_object",
    [
        pytest.param("simulation", id="simulation"),
        pytest.param("tracer", id="tracer"),
        pytest.param("receipt", id="receipt"),
        pytest.param("builder-closure", id="builder-closure"),
        pytest.param("simulation-cycle", id="simulation-cycle"),
        pytest.param("tracer-cycle", id="tracer-cycle"),
        pytest.param("failed-simulation", id="failed-simulation"),
        pytest.param("failed-tracer", id="failed-tracer"),
    ],
)
def test_execution_does_not_retain_case_objects(retained_object: str) -> None:
    system = FakeSystem()
    simulation = FakeSimulation("case")
    receipt = FakeSimulation("receipt")
    simulation.tracer.receipts.append(receipt)
    if retained_object == "simulation-cycle":
        simulation.cycle = simulation
    if retained_object == "tracer-cycle":
        simulation.tracer.cycle = simulation.tracer
    references = {
        "simulation": weakref.ref(simulation),
        "tracer": weakref.ref(simulation.tracer),
        "receipt": weakref.ref(receipt),
        "builder-closure": weakref.ref(simulation),
        "simulation-cycle": weakref.ref(simulation),
        "tracer-cycle": weakref.ref(simulation.tracer),
        "failed-simulation": weakref.ref(simulation),
        "failed-tracer": weakref.ref(simulation.tracer),
    }

    def build():
        system.simulation = simulation
        if retained_object.startswith("failed"):
            raise RuntimeError("failure")
        return simulation

    execution = YamlCaseExecution(system, build)
    if retained_object.startswith("failed"):
        with pytest.raises(RuntimeError):
            execution.__enter__()
    else:
        with execution:
            pass

    del receipt
    del simulation
    del build
    gc.collect()

    assert references[retained_object]() is None


@pytest.mark.parametrize(
    "previous_state",
    [
        pytest.param("absent", id="absent"),
        pytest.param("none", id="none"),
        pytest.param("object", id="object"),
        pytest.param("false", id="false"),
        pytest.param("zero", id="zero"),
        pytest.param("empty-list", id="empty-list"),
        pytest.param("empty-dict", id="empty-dict"),
        pytest.param("empty-string", id="empty-string"),
    ],
)
def test_previous_reference_is_restored_exactly(previous_state: str) -> None:
    system = FakeSystem()
    previous_values = {
        "none": None,
        "object": FakeSimulation("previous"),
        "false": False,
        "zero": 0,
        "empty-list": [],
        "empty-dict": {},
        "empty-string": "",
    }
    if previous_state != "absent":
        system.simulation = previous_values[previous_state]
    simulation = FakeSimulation("case")

    with YamlCaseExecution(system, lambda: simulation):
        system.simulation = simulation

    if previous_state == "absent":
        assert not hasattr(system, "simulation")
    else:
        assert system.simulation is previous_values[previous_state]
