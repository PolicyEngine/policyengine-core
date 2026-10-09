"""One-variable simulations for testing what ADD and DIVIDE results cache.

Shared by ``tests/core/test_option_result_cache.py`` and its Hypothesis
property module, which pytest must be able to skip on its own when
Hypothesis is not installed.
"""

from typing import Any, Dict, Optional

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import QuantityType, Variable

PROBE = "probe"


def period_code(period) -> float:
    """A value that differs for every month and year, for probe formulas."""
    start = period.start
    if period.unit == periods.YEAR:
        return 1000.0 + start.year % 100
    return float(100 * (start.year % 100) + start.month)


def make_probe(
    definition_period: str,
    quantity_type: str,
    value_type: type = float,
    with_formula: bool = False,
    set_input: Optional[Any] = "default",
) -> type:
    """A Person variable named ``probe`` with the given time behaviour.

    With ``with_formula``, its formula returns ``period_code(period)``, so
    sums over months and twelfths of years are easy to tell apart from
    single-period values. ``set_input=None`` stores inputs at the period
    given instead of spreading them over the variable's own periods.
    """
    attributes = dict(
        value_type=value_type,
        entity=Person,
        definition_period=definition_period,
        quantity_type=quantity_type,
        label="Option cache probe",
    )
    if set_input is None:
        attributes["set_input"] = None
    if with_formula:

        def formula(person, period):
            return person.filled_array(period_code(period))

        attributes["formula"] = formula
    return type(PROBE, (Variable,), attributes)


def build(
    probe: type,
    inputs: Optional[Dict[str, Any]] = None,
    carry_over: bool = False,
):
    """One-person simulation of ``probe`` with ``inputs`` ({period: value})."""
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = carry_over
    system.add_variable(probe)
    simulation = SimulationBuilder().build_default_simulation(system, count=1)
    for period, value in (inputs or {}).items():
        simulation.set_input(PROBE, period, np.array([value]))
    return simulation


def run(simulation, operation: str, period: str):
    """Run ``calculate``, ``add`` or ``divide`` on ``probe``.

    Returns the value as a list, or the error's type name, so that results of
    fresh and reused simulations compare equal exactly when both return the
    same values or both raise the same kind of error.
    """
    method = {
        "calculate": simulation.calculate,
        "add": simulation.calculate_add,
        "divide": simulation.calculate_divide,
    }[operation]
    try:
        return np.asarray(method(PROBE, period)).astype(float).tolist()
    except ValueError as error:
        return type(error).__name__


STOCK = QuantityType.STOCK
FLOW = QuantityType.FLOW
MONTH = periods.MONTH
YEAR = periods.YEAR
DAY = periods.DAY
