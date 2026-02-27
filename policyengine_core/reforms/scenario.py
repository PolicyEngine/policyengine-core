from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from policyengine_core.simulations import Simulation


class Scenario(BaseModel):
    parameter_changes: (
        dict[str, dict[str | int, float | int | bool]] | None
    ) = None
    """A dictionary mapping parameter names to their time-period-specific changes."""
    modifier_function: Callable[["Simulation"], None] | None = None
    """A function that modifies the simulation in some way, e.g., by applying a tax policy change."""

    def __init__(self):
        # Validate parameter changes
        pass
        # Validate modifier function?
        pass
