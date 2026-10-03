"""A small dataset-backed simulation for testing what ``subsample`` keeps.

Shared by ``tests/core/test_subsample_inputs_only.py`` and its Hypothesis
property module, which pytest must be able to skip on its own when
Hypothesis is not installed.
"""

import numpy as np
import pandas as pd

from policyengine_core.country_template import CountryTaxBenefitSystem, Simulation
from policyengine_core.country_template.entities import Household, Person
from policyengine_core.data import Dataset
from policyengine_core.periods import YEAR
from policyengine_core.variables import Variable

DATASET_YEAR = "2022"
HOUSEHOLD_OF_PERSON = [1, 1, 2, 2, 2, 3, 4, 4, 5, 6]
BASE = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
WEIGHTS = {1: 100.0, 2: 200.0, 3: 300.0, 4: 400.0, 5: 500.0, 6: 600.0}


class base(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Input amount"


class doubled(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Twice the input amount"

    def formula(person, period):
        return 2 * person("base", period)


class ended(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Seven, until its formula ends with the dataset year"
    end = f"{DATASET_YEAR}-12-31"

    def formula(person, period):
        return person.filled_array(7.0)


class household_total(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household sum of the doubled amount"

    def formula(household, period):
        return household.sum(household.members("doubled", period))


class overridden(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "A formula variable the dataset gives values for"

    def formula(person, period):
        return person.filled_array(-1.0)


FORMULA_VARIABLES = ["doubled", "ended", "household_total", "person_weight"]
OVERRIDDEN = [float(index) for index in range(len(BASE))]


def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"person_id__{DATASET_YEAR}": list(range(1, len(BASE) + 1)),
            f"household_id__{DATASET_YEAR}": HOUSEHOLD_OF_PERSON,
            f"person_household_id__{DATASET_YEAR}": HOUSEHOLD_OF_PERSON,
            f"household_weight__{DATASET_YEAR}": [
                WEIGHTS[household] for household in HOUSEHOLD_OF_PERSON
            ],
            f"base__{DATASET_YEAR}": BASE,
            f"overridden__{DATASET_YEAR}": OVERRIDDEN,
        }
    )


def build(carry_over: bool = True) -> Simulation:
    """A ten-person, six-household simulation loaded from ``dataframe()``."""
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = carry_over
    system.add_variables(base, doubled, ended, household_total, overridden)
    return Simulation(
        tax_benefit_system=system,
        dataset=Dataset.from_dataframe(dataframe(), DATASET_YEAR),
    )


def stored(simulation) -> dict:
    """Every stored value: ``{(variable, branch, period): values}``."""
    values = {}
    for population in simulation.populations.values():
        for name, holder in population._holders.items():
            for branch, period in holder.get_known_branch_periods():
                array = holder.get_array(period, branch)
                values[(name, branch, str(period))] = np.asarray(array).tolist()
    return values
