from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

SystemT = TypeVar("SystemT")
SimulationT = TypeVar("SimulationT")

_ABSENT = object()


class YamlCaseExecutionError(RuntimeError):
    """Raised when a YAML case execution violates its lifecycle."""


class YamlCaseExecution(Generic[SystemT, SimulationT]):
    """Own the temporary objects and backreference for one YAML case."""

    def __init__(
        self,
        policy_system: SystemT,
        simulation_builder: Callable[[], SimulationT],
    ) -> None:
        if policy_system is None:
            raise YamlCaseExecutionError("a policy system is required")
        if not callable(simulation_builder):
            raise YamlCaseExecutionError("a callable simulation builder is required")
        self._policy_system: SystemT | None = policy_system
        self._simulation_builder: Callable[[], SimulationT] | None = simulation_builder
        self._simulation: SimulationT | None = None
        self._previous_simulation: Any = _ABSENT
        self._active = False
        self._closed = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def simulation(self) -> SimulationT:
        if not self._active or self._simulation is None:
            raise YamlCaseExecutionError("the YAML case is not active")
        return self._simulation

    def __enter__(self) -> SimulationT:
        if self._closed:
            raise YamlCaseExecutionError("the YAML case execution is closed")
        if self._active:
            raise YamlCaseExecutionError("the YAML case execution is already active")
        if self._policy_system is None or self._simulation_builder is None:
            raise YamlCaseExecutionError("the YAML case execution has no resources")

        self._previous_simulation = getattr(
            self._policy_system,
            "simulation",
            _ABSENT,
        )
        try:
            simulation = self._simulation_builder()
            if simulation is None:
                raise YamlCaseExecutionError(
                    "the simulation builder returned None",
                )
        except BaseException:
            self.close()
            raise

        self._simulation = simulation
        self._active = True
        return simulation

    def close(self) -> None:
        if self._closed:
            return
        system = self._policy_system
        if system is not None and self._previous_simulation is not _ABSENT:
            setattr(system, "simulation", self._previous_simulation)
        elif system is not None and hasattr(system, "simulation"):
            delattr(system, "simulation")
        self._simulation = None
        self._simulation_builder = None
        self._policy_system = None
        self._active = False
        self._closed = True

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
