"""A synthetic system for derived values and branches inside formulas."""

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.holders import set_input_divide_by_period
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable


def _yearly(name, formula=None, **attributes):
    namespace = dict(
        value_type=float,
        entity=entities.Person,
        definition_period=periods.YEAR,
        label=name,
        **attributes,
    )
    if formula is not None:
        namespace["formula"] = formula
    return type(name, (Variable,), namespace)


def _inner_branch_formula(person, period):
    """Calculate a value in a branch that exists only while this formula runs."""
    simulation = person.simulation
    inner = simulation.get_branch("inner")
    try:
        inner.set_input("p_switch", period, np.ones(person.count))
        inner_value = inner.calculate("p_inner_only", period)
    finally:
        del simulation.branches["inner"]
    return inner_value + person("p_sum", period)


def _monthly_formula(person, period):
    return (
        person("p_a", period)
        + person("p_b", period.this_year) / 12
        + person("p_m", period)
    )


def _monthly(name, formula=None, **attributes):
    namespace = dict(
        value_type=float,
        entity=entities.Person,
        definition_period=periods.MONTH,
        label=name,
        **attributes,
    )
    if formula is not None:
        namespace["formula"] = formula
    return type(name, (Variable,), namespace)


SYNTHETIC_VARIABLES = [
    # Inputs. p_up is uprated by a parameter that changes every year from
    # 2012 to 2015; p_c is given for 2012 only, and carried over where the
    # system carries inputs over.
    _yearly("p_a"),
    _yearly("p_b"),
    _yearly("p_up", uprating="taxes.income_tax_rate"),
    _yearly("p_c"),
    _yearly(
        "p_sum",
        lambda person, period: person("p_a", period) + 2 * person("p_b", period),
    ),
    _yearly(
        "p_prod",
        lambda person, period: (
            person("p_sum", period) * (1 + person("p_up", period) / 100)
        ),
    ),
    _yearly(
        "p_cond",
        lambda person, period: np.where(
            person("p_sum", period) > 60,
            person("p_prod", period),
            person("p_c", period),
        ),
    ),
    _yearly(
        "p_lag",
        lambda person, period: (
            person("p_sum", period.last_year) + person("p_a", period)
        ),
    ),
    # A monthly input; an input for a year is divided between its months.
    _monthly("p_m", set_input=set_input_divide_by_period),
    _monthly("p_month", _monthly_formula),
    _yearly("p_months", lambda person, period: person("p_month", period)),
    _yearly("p_switch", lambda person, period: np.zeros(person.count)),
    _yearly(
        "p_inner_only",
        lambda person, period: 3 * person("p_a", period) + person("p_switch", period),
    ),
    _yearly("p_inner", _inner_branch_formula),
    # Reads itself a year earlier, back until core's spiral detection gives
    # up and returns the default.
    _yearly(
        "p_spiral", lambda person, period: person("p_spiral", period.last_year) + 1
    ),
]


def _synthetic_system(carry_over):
    class System(CountryTaxBenefitSystem):
        auto_carry_over_input_variables = carry_over

    system = System()
    system.add_variables(*SYNTHETIC_VARIABLES)
    return system


# Carry-over is left out of the property test's random programs: core's
# carry-over itself depends on calculation order (a later period calculated
# first stops an earlier input being carried into the years between), in any
# simulation.
SYNTHETIC_SYSTEM = _synthetic_system(carry_over=False)
CARRY_OVER_SYSTEM = _synthetic_system(carry_over=True)

PEOPLE = 3
YEARS = ["2012", "2013", "2014", "2015"]


def synthetic_simulation(inputs, memory_config=None, system=SYNTHETIC_SYSTEM):
    """A simulation of ``PEOPLE`` people given ``inputs`` before anything else."""
    simulation = SimulationBuilder().build_default_simulation(system, count=PEOPLE)
    if memory_config is not None:
        simulation.memory_config = memory_config
        # Holders read the memory configuration when created; nothing is
        # stored yet, so create them again. Create them all here, so every
        # branch's holders are copies of these: a holder first created in a
        # branch would own (and remove on deletion) a directory the whole
        # family's disk storage shares.
        for population in simulation.populations.values():
            population._holders = {}
        for variable in system.variables:
            simulation.get_holder(variable)
    for (variable, period), values in sorted(
        inputs.items(),
        key=lambda item: periods.key_period_size(periods.period(item[0][1])),
    ):
        simulation.set_input(variable, period, np.asarray(values, dtype=float))
    return simulation


ROOT_INPUTS = {
    **{("p_a", year): (10.0 * i, 20.0, 35.0 + i) for i, year in enumerate(YEARS)},
    **{("p_b", year): (5.0, 15.0 + i, 25.0) for i, year in enumerate(YEARS)},
    ("p_up", "2012"): (100.0, 200.0, 300.0),
    ("p_c", "2012"): (7.0, 8.0, 9.0),
    **{
        ("p_m", f"{year}-{month:02d}"): (1.0 * month, 2.0, 0.5 * month)
        for year in (2013, 2014)
        for month in range(1, 13)
    },
}
