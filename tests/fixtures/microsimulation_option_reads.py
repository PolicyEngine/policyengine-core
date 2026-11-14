"""A small system whose formulas read variables with ADD and DIVIDE.

Shared by ``tests/core/test_microsimulation_option_reads.py`` and its
Hypothesis property module, which pytest must be able to skip on its own when
Hypothesis is not installed.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from policyengine_core.country_template import (
    CountryTaxBenefitSystem,
    Microsimulation,
    Simulation,
)
from policyengine_core.country_template.entities import Household, Person
from policyengine_core.data import Dataset
from policyengine_core.enums import Enum
from policyengine_core.model_api import ADD, DIVIDE, MONTH, YEAR, Variable, add
from policyengine_core.simulations import Microsimulation as CoreMicrosimulation

DATASET_YEAR = "2022"
MONTHS = [f"{DATASET_YEAR}-{month:02d}" for month in range(1, 13)]


# Inputs, and variables with formulas that the readers below read.


class monthly_pay(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Monthly pay"


class monthly_take_home(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Monthly take-home pay"

    def formula(person, period, parameters):
        return person("monthly_pay", period) * 0.75


class yearly_grant(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Yearly grant"


class household_monthly_fee(Variable):
    value_type = float
    entity = Household
    definition_period = MONTH
    label = "Household monthly fee"


class household_yearly_grant(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household yearly grant"


class household_yearly_allowance(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household yearly allowance"

    def formula(household, period, parameters):
        return household("household_yearly_grant", period) * 2


class Tenure(Enum):
    OWNER = "Owner"
    RENTER = "Renter"


class yearly_tenure(Variable):
    value_type = Enum
    possible_values = Tenure
    default_value = Tenure.RENTER
    entity = Person
    definition_period = YEAR
    label = "Yearly tenure"


# Readers: formulas that read with ADD or DIVIDE, as 21 policyengine-us
# formulas do (``spm_unit("snap", period, options=[ADD]) > 0`` in
# mt_liheap_categorical_eligibility, for one).


class pay_added(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Pay summed over the year"

    def formula(person, period, parameters):
        return person("monthly_pay", period, options=[ADD])


class take_home_added(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Take-home pay summed over the year"

    def formula(person, period, parameters):
        return person("monthly_take_home", period, options=[ADD])


class grant_divided(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Monthly share of the yearly grant"

    def formula(person, period, parameters):
        return person("yearly_grant", period, options=[DIVIDE])


class fee_added(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household fees summed over the year"

    def formula(household, period, parameters):
        return household("household_monthly_fee", period, options=[ADD])


class pays_fee(Variable):
    value_type = bool
    entity = Household
    definition_period = YEAR
    label = "Household paid a fee in the year"

    def formula(household, period, parameters):
        return household("household_monthly_fee", period, options=[ADD]) > 0


class allowance_divided(Variable):
    value_type = float
    entity = Household
    definition_period = MONTH
    label = "Monthly share of the household allowance"

    def formula(household, period, parameters):
        return household("household_yearly_allowance", period, options=[DIVIDE])


class members_pay_added(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Members' pay summed over the year"

    def formula(household, period, parameters):
        return household("monthly_pay", period, options=[ADD])


class members_grant_divided(Variable):
    value_type = float
    entity = Household
    definition_period = MONTH
    label = "Monthly share of members' yearly grants"

    def formula(household, period, parameters):
        return household("yearly_grant", period, options=[DIVIDE])


class fee_and_pay_added(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household fees and members' pay summed over the year"

    def formula(household, period, parameters):
        return add(
            household, period, ["household_monthly_fee", "monthly_pay"], options=[ADD]
        )


class fee_or_all_paid(Variable):
    value_type = bool
    entity = Household
    definition_period = YEAR
    label = "Household paid a fee or every member was paid"

    def formula(household, period, parameters):
        # policyengine-us's wi_wheap_categorically_eligible has this shape: it
        # projects a group's ADD read to its members and combines the result
        # with a person value.
        person = household.members
        paid = add(person, period, ["monthly_pay"]) > 0
        pays = household("household_monthly_fee", period, options=[ADD]) > 0
        return household.all(paid | household.project(pays))


SOURCES = [
    monthly_pay,
    monthly_take_home,
    yearly_grant,
    household_monthly_fee,
    household_yearly_grant,
    household_yearly_allowance,
]
READERS = [
    pay_added,
    take_home_added,
    grant_divided,
    fee_added,
    pays_fee,
    allowance_divided,
    members_pay_added,
    members_grant_divided,
    fee_and_pay_added,
    fee_or_all_paid,
]


@lru_cache(maxsize=None)
def tax_benefit_system() -> CountryTaxBenefitSystem:
    system = CountryTaxBenefitSystem()
    for variable in SOURCES + READERS + [yearly_tenure]:
        system.add_variable(variable)
    return system


@dataclass(frozen=True)
class Inputs:
    """People in the order the data lists them, and each input's values.

    ``household_of_person`` gives each person's household. Households take the
    numbers 0, 1, ... and each has at least one member; the data can list a
    household's members anywhere. ``monthly_pay`` and
    ``household_monthly_fee`` hold 12 values (January to December) for each
    person or household.
    """

    household_of_person: Tuple[int, ...]
    household_weight: Tuple[float, ...]
    monthly_pay: Tuple[Tuple[float, ...], ...]
    yearly_grant: Tuple[float, ...]
    household_monthly_fee: Tuple[Tuple[float, ...], ...]
    household_yearly_grant: Tuple[float, ...]
    yearly_tenure: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        # A dataset declares a household only through its members' rows.
        if set(self.household_of_person) != set(range(len(self.household_weight))):
            raise ValueError("Every household needs a weight and a member.")


def dataframe(inputs: Inputs) -> pd.DataFrame:
    """One row per person. Each member's row repeats its household's values."""
    household = np.array(inputs.household_of_person)
    first_member = np.zeros(len(household), dtype=bool)
    first_member[np.unique(household, return_index=True)[1]] = True
    pay = np.array(inputs.monthly_pay, dtype=float)
    fee = np.array(inputs.household_monthly_fee, dtype=float)
    data = {
        f"person_id__{DATASET_YEAR}": np.arange(len(household)),
        f"person_household_id__{DATASET_YEAR}": household,
        f"person_household_role__{DATASET_YEAR}": np.where(
            first_member, "parent", "child"
        ),
        f"household_weight__{DATASET_YEAR}": np.array(inputs.household_weight)[
            household
        ],
        f"yearly_grant__{DATASET_YEAR}": np.array(inputs.yearly_grant),
        f"household_yearly_grant__{DATASET_YEAR}": np.array(
            inputs.household_yearly_grant
        )[household],
    }
    for index, month in enumerate(MONTHS):
        data[f"monthly_pay__{month}"] = pay[:, index]
        data[f"household_monthly_fee__{month}"] = fee[household, index]
    if inputs.yearly_tenure is not None:
        data[f"yearly_tenure__{DATASET_YEAR}"] = np.array(inputs.yearly_tenure)
    return pd.DataFrame(data)


def simulations(inputs: Inputs) -> Tuple[Simulation, Microsimulation]:
    """A Simulation and a Microsimulation, each loaded from ``inputs``."""
    frame = dataframe(inputs)
    return tuple(
        simulation_class(
            tax_benefit_system=tax_benefit_system(),
            dataset=Dataset.from_dataframe(frame, DATASET_YEAR),
        )
        for simulation_class in (Simulation, Microsimulation)
    )


def unweighted(simulation) -> dict:
    """Keyword arguments that make ``simulation`` return unweighted values."""
    if isinstance(simulation, CoreMicrosimulation):
        return {"use_weights": False}
    return {}


def outcome(read: Callable):
    """What ``read()`` returns, as a list, or the name of the error it raises."""
    try:
        return np.asarray(read()).tolist()
    except Exception as error:
        return type(error).__name__


def run(simulation, request: tuple):
    """Run one read ``request`` on ``simulation`` and return its outcome.

    A request is one of:

    - ``("calculate", variable, period)``;
    - ``("population", entity_key, variable, period, option)``, the call a
      formula makes;
    - ``("add" or "divide", variable, period, decode_enums)``.

    Reads in a Microsimulation are unweighted.
    """
    kind, *arguments = request
    if kind == "calculate":
        variable, period = arguments
        return outcome(
            lambda: simulation.calculate(variable, period, **unweighted(simulation))
        )
    if kind == "population":
        entity_key, variable, period, option = arguments
        population = simulation.populations[entity_key]
        return outcome(lambda: population(variable, period, options=[option]))
    variable, period, decode_enums = arguments
    method = {"add": simulation.calculate_add, "divide": simulation.calculate_divide}[
        kind
    ]
    return outcome(
        lambda: method(
            variable, period, decode_enums=decode_enums, **unweighted(simulation)
        )
    )


def expected(inputs: Inputs, reader: str) -> np.ndarray:
    """``reader`` worked out from ``inputs`` with NumPy, without Core.

    Monthly readers return the same value for every month of the year.
    """
    household = np.array(inputs.household_of_person)
    households = len(inputs.household_weight)
    pay = np.array(inputs.monthly_pay, dtype=float)
    grant = np.array(inputs.yearly_grant, dtype=float)
    fee = np.array(inputs.household_monthly_fee, dtype=float)
    household_grant = np.array(inputs.household_yearly_grant, dtype=float)

    def members(values):
        return np.bincount(household, weights=values, minlength=households)

    paid = pay.sum(axis=1) > 0
    pays = fee.sum(axis=1) > 0
    return {
        "pay_added": pay.sum(axis=1),
        "take_home_added": (0.75 * pay).sum(axis=1),
        "grant_divided": grant / 12,
        "fee_added": fee.sum(axis=1),
        "pays_fee": fee.sum(axis=1) > 0,
        "allowance_divided": 2 * household_grant / 12,
        "members_pay_added": members(pay.sum(axis=1)),
        "members_grant_divided": members(grant) / 12,
        "fee_and_pay_added": fee.sum(axis=1) + members(pay.sum(axis=1)),
        # ``all`` over each household's members: no member fails the test.
        "fee_or_all_paid": members(~(paid | pays[household]) * 1.0) == 0,
    }[reader]
