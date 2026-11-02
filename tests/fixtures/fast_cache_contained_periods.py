"""Variables and builders shared by the fast-cache containment tests."""

from __future__ import annotations

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.holders import set_input_divide_by_period
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable


class monthly_input(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly input"


class monthly_input_split(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly input an annual value is split over"
    set_input = set_input_divide_by_period


class monthly_formula(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly formula of the monthly input"

    def formula(person, period, parameters):
        return person("monthly_input", period) + 1


class yearly_input(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly input"


class eternal_input(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.ETERNITY
    label = "Eternal input"


class eternal_formula(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.ETERNITY
    label = "Eternal formula of the eternal input"

    def formula(person, period, parameters):
        return person("eternal_input", period) * 2


VARIABLES = (
    monthly_input,
    monthly_input_split,
    monthly_formula,
    yearly_input,
    eternal_input,
    eternal_formula,
)


def build_simulation(*, carry_over: bool = True):
    """One person, with the variables above."""
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = carry_over
    simulation = SimulationBuilder().build_default_simulation(system, count=1)
    system.add_variables(*VARIABLES)
    return simulation
