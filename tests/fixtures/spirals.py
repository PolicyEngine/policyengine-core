"""Variables that read themselves, or each other, at other periods.

Shared by ``tests/core/test_spiral_order.py`` and its Hypothesis property
module, which pytest must be able to skip on its own when Hypothesis is not
installed.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable


def make_recurrence(
    name: str,
    terms: Iterable[Tuple[str, int]],
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> type:
    """A yearly variable equal to 1 plus other variables at lagged years.

    Each of ``terms`` is ``(variable, lag)``: the variable's value ``lag``
    years earlier (later when negative). ``start`` dates the formula
    (``formula_<start>``), so that earlier years have none, and ``end`` is the
    last year the variable has a formula for.
    """
    terms = tuple(terms)

    def formula(person, period):
        value = person.filled_array(1.0)
        for variable, lag in terms:
            value = value + person(variable, period.offset(-lag, "year"))
        return value

    attributes = dict(
        value_type=float,
        entity=Person,
        definition_period=periods.YEAR,
        label=f"Recurrence {name}",
    )
    attributes["formula" if start is None else f"formula_{start}"] = formula
    if end is not None:
        attributes["end"] = f"{end}-12-31"
    return type(name, (Variable,), attributes)


def build(
    variables: Iterable[type],
    inputs: Optional[Dict[Tuple[str, int], float]] = None,
    max_spiral_loops: Optional[int] = None,
):
    """One-person simulation of ``variables`` with ``inputs``.

    ``inputs`` maps ``(variable, year)`` to a value.
    """
    system = CountryTaxBenefitSystem()
    system.add_variables(*variables)
    simulation = SimulationBuilder().build_default_simulation(system, count=1)
    for (variable, year), value in (inputs or {}).items():
        simulation.set_input(variable, str(year), np.array([value]))
    if max_spiral_loops is not None:
        simulation.max_spiral_loops = max_spiral_loops
    return simulation


def run(simulation, variable: str, year: int):
    """The value as a list, or the name of the error the calculation raises."""
    try:
        return simulation.calculate(variable, str(year)).tolist()
    except Exception as error:
        return type(error).__name__


def stored_years(simulation, variable: str, branch_name: str = "default"):
    """The years ``variable`` has a value stored for under ``branch_name``."""
    return sorted(
        period.start.year
        for branch, period in simulation.get_holder(variable).get_known_branch_periods()
        if branch == branch_name
    )
