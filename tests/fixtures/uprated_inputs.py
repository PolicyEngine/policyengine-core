"""A small tax-benefit system for tests of inputs, uprating and caches.

Two people, each in their own group entities. The variables cover the
storage paths the dump/restore and fast-cache tests need: uprated int and
float inputs, an input with a ``set_input`` helper (annual values split into
months), ETERNITY inputs and formulas, formulas reading other variables, and
an uprated input masked by ``defined_for``.
"""

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.parameters import Parameter
from policyengine_core.reforms import Reform
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable

PERSON_COUNT = 2


class uprated_count(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = "index"
    label = "Uprated integer input"


class uprated_amount(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = "index"
    label = "Uprated float input"


class doubled_amount(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Twice the uprated float input"

    def formula(person, period):
        return 2 * person("uprated_amount", period)


class monthly_amount(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly input; an annual input is split into twelve months"


class monthly_doubled(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Twice the monthly input"

    def formula(person, period):
        return 2 * person("monthly_amount", period)


class eternal_code(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.ETERNITY
    label = "ETERNITY input"


class eternal_code_plus_one(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.ETERNITY
    label = "ETERNITY formula"

    def formula(person, period):
        return person("eternal_code", period) + 1


class eligible(Variable):
    value_type = bool
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Eligibility"


class masked_amount(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    uprating = "index"
    defined_for = "eligible"
    label = "Uprated float input, kept only where eligible"


VARIABLES = (
    uprated_count,
    uprated_amount,
    doubled_amount,
    monthly_amount,
    monthly_doubled,
    eternal_code,
    eternal_code_plus_one,
    eligible,
    masked_amount,
)

# Value type of each input variable, for building input arrays.
INPUT_VALUE_TYPES = {
    "uprated_count": int,
    "uprated_amount": float,
    "monthly_amount": float,
    "eternal_code": int,
    "eligible": bool,
    "masked_amount": float,
}


class NoOp(Reform):
    """A reform that changes nothing: ``apply_reform`` then only drops caches."""

    def apply(self):
        pass


def build_system() -> CountryTaxBenefitSystem:
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = True
    system.parameters.add_child(
        "index",
        Parameter(
            "index",
            data={
                "values": {
                    f"{year}-01-01": 100 * 1.037 ** (year - 2010)
                    for year in range(2010, 2021)
                }
            },
        ),
    )
    system.add_variables(*VARIABLES)
    return system


def build_simulation(system, inputs=None):
    """A two-person simulation with ``inputs`` set in order.

    ``inputs`` is a list of ``(variable, period, values)``.
    """
    simulation = SimulationBuilder().build_default_simulation(
        system, count=PERSON_COUNT
    )
    for variable, period, values in inputs or ():
        simulation.set_input(variable, period, values)
    return simulation


def input_values(variable: str, seed: int):
    """Two input values for ``variable``, varied by ``seed``."""
    value_type = INPUT_VALUE_TYPES[variable]
    if value_type is bool:
        return [seed % 2 == 1, seed % 3 != 0]
    if value_type is int:
        return [1001 + 37 * seed, 77 + seed]
    return [1000.5 + 37.25 * seed, 12.75 + seed]
