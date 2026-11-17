"""A small system for checking what calls made inside a calculation return.

Used by ``tests/core/test_microsimulation_internal_calls.py``. It adds to the
ADD and DIVIDE fixture (``microsimulation_option_reads``):

- variables that core reads for itself: one with ``adds`` and ``subtracts``,
  one with ``defined_for``;
- probe formulas. Each makes one kind of call from inside a calculation and
  stores what the call returns in ``RECEIVED``, under a label.
"""

import contextvars
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict

import numpy as np

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.data import Dataset
from policyengine_core.model_api import ADD, DIVIDE, MONTH, YEAR, Variable
from policyengine_core.reforms import Reform
from policyengine_core.simulations import Microsimulation, Simulation
from tests.fixtures.microsimulation_option_reads import (
    DATASET_YEAR,
    READERS,
    SOURCES,
    Inputs,
    Tenure,
    dataframe,
    yearly_tenure,
)

EXAMPLE = Inputs(
    household_of_person=(0, 0, 1),
    household_weight=(10.0, 20.0),
    monthly_pay=(
        tuple(1.0 * month for month in range(1, 13)),
        tuple(2.0 * month for month in range(1, 13)),
        tuple(5.0 * month for month in range(1, 13)),
    ),
    yearly_grant=(120.0, 240.0, 0.0),
    household_monthly_fee=((5.0,) * 12, (0.0,) * 12),
    household_yearly_grant=(12.0, 36.0),
    yearly_tenure=("OWNER", "RENTER", "OWNER"),
)

# What the probe formulas received, by label. Tests clear it before each run.
RECEIVED: Dict[str, object] = {}


# Variables core reads for itself, with no formula of their own.


class pay_and_grant(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Pay summed over the year, plus the grant, less the take-home pay"
    # ``monthly_pay`` and ``monthly_take_home`` are monthly, so each is read
    # over the year by summing its months.
    adds = ["monthly_pay", "yearly_grant"]
    subtracts = ["monthly_take_home"]


class pay_if_granted(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Pay summed over the year, for people with a grant"
    defined_for = "yearly_grant"

    def formula(person, period, parameters):
        return person("monthly_pay", period, options=[ADD])


# Formulas that read an Enum with ADD or DIVIDE (core issue #658).


class owner_by_add(Variable):
    value_type = bool
    entity = Person
    definition_period = YEAR
    label = "Owner, from a read of the tenure with ADD"

    def formula(person, period, parameters):
        return person("yearly_tenure", period, options=[ADD]) == Tenure.OWNER


class owner_by_divide(Variable):
    value_type = bool
    entity = Person
    definition_period = YEAR
    label = "Owner, from a read of the tenure with DIVIDE"

    def formula(person, period, parameters):
        return person("yearly_tenure", period, options=[DIVIDE]) == Tenure.OWNER


class owner_twelfth(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "A twelfth of the tenure code, from a read with DIVIDE"

    def formula(person, period, parameters):
        return np.asarray(person("yearly_tenure", period, options=[DIVIDE]))


# Probe formulas.


def _zeros(person):
    return np.zeros(person.count)


def _reads_own_simulation(person, period, parameters):
    simulation = person.simulation
    RECEIVED["population read"] = person("yearly_grant", period)
    RECEIVED["population enum read"] = person("yearly_tenure", period)
    RECEIVED["calculate"] = simulation.calculate("yearly_grant", period)
    RECEIVED["calculate enum"] = simulation.calculate("yearly_tenure", period)
    # Already stored by the read above: Simulation.calculate returns it from
    # its fast cache.
    RECEIVED["calculate again"] = simulation.calculate("yearly_grant", period)
    RECEIVED["calculate_add"] = simulation.calculate_add("monthly_pay", period)
    RECEIVED["calculate_divide"] = simulation.calculate_divide("yearly_grant", period)
    RECEIVED["calculate_divide month"] = simulation.calculate_divide(
        "yearly_grant", period.first_month
    )
    RECEIVED["calculate use_weights=True"] = simulation.calculate(
        "yearly_grant", period, use_weights=True
    )
    RECEIVED["calculate decode_enums=True"] = simulation.calculate(
        "yearly_tenure", period, decode_enums=True
    )
    RECEIVED["calculate_add use_weights=True"] = simulation.calculate_add(
        "monthly_pay", period, use_weights=True
    )
    RECEIVED["calculate_divide use_weights=True"] = simulation.calculate_divide(
        "yearly_grant", period, use_weights=True
    )
    return _zeros(person)


def _reads_a_branch(person, period, parameters):
    branch = person.simulation.get_branch("probe_branch")
    RECEIVED["branch calculate"] = branch.calculate("yearly_grant", period)
    RECEIVED["branch calculate enum"] = branch.calculate("yearly_tenure", period)
    return _zeros(person)


def _reads_its_baseline(person, period, parameters):
    RECEIVED["baseline calculate"] = person.simulation.baseline.calculate(
        "yearly_grant", period
    )
    return _zeros(person)


def _builds_a_microsimulation(person, period, parameters):
    inner = ProbeMicrosimulation(dataset=example_dataset())
    RECEIVED["new microsimulation calculate"] = inner.calculate("yearly_grant", period)
    return _zeros(person)


def _reads_from_worker_threads(person, period, parameters):
    simulation = person.simulation
    with ThreadPoolExecutor(max_workers=1) as pool:
        RECEIVED["thread calculate"] = pool.submit(
            simulation.calculate, "yearly_grant", period
        ).result()
        context = contextvars.copy_context()
        RECEIVED["thread calculate, copied context"] = pool.submit(
            context.run, simulation.calculate, "yearly_grant", period
        ).result()
    return _zeros(person)


def _keeps_its_context(person, period, parameters):
    # A test runs a calculation in this context after the formula returns.
    RECEIVED["context"] = contextvars.copy_context()
    return _zeros(person)


def _raises(person, period, parameters):
    person.simulation.calculate("yearly_grant", period.this_year)
    raise RuntimeError("probe error")


class ProbeInterrupt(BaseException):
    """Raised by a probe formula: not an ``Exception``, as ``KeyboardInterrupt``."""


def _raises_base_exception(person, period, parameters):
    person.simulation.calculate("yearly_grant", period)
    raise ProbeInterrupt()


def _catches_a_branch_error(person, period, parameters):
    branch = person.simulation.get_branch("probe_error_branch")
    try:
        branch.calculate("probe_raises", period)
    except RuntimeError:
        pass
    RECEIVED["calculate after a caught branch error"] = person.simulation.calculate(
        "yearly_grant", period
    )
    return _zeros(person)


def _sets_an_input(person, period, parameters):
    # Setting an input drops values computed from it, so the outermost
    # calculation runs again.
    simulation = person.simulation
    simulation.calculate("pay_and_grant", period)
    simulation.set_input("yearly_grant", period, np.full(person.count, 1234.0))
    RECEIVED["calculate after setting an input"] = simulation.calculate(
        "pay_and_grant", period
    )
    return simulation.calculate("pay_and_grant", period)


def _probe(name, formula, definition_period=YEAR):
    return type(
        name,
        (Variable,),
        {
            "value_type": float,
            "entity": Person,
            "definition_period": definition_period,
            "label": name,
            "formula": formula,
        },
    )


PROBES = [
    _probe("probe_reads_own_simulation", _reads_own_simulation),
    _probe("probe_reads_a_branch", _reads_a_branch),
    _probe("probe_reads_its_baseline", _reads_its_baseline),
    _probe("probe_builds_a_microsimulation", _builds_a_microsimulation),
    _probe("probe_reads_from_worker_threads", _reads_from_worker_threads),
    _probe("probe_keeps_its_context", _keeps_its_context),
    _probe("probe_raises", _raises),
    _probe("probe_monthly_raises", _raises, definition_period=MONTH),
    _probe("probe_raises_base_exception", _raises_base_exception),
    _probe("probe_catches_a_branch_error", _catches_a_branch_error),
    _probe("probe_sets_an_input", _sets_an_input),
]


class ProbeSystem(CountryTaxBenefitSystem):
    """The country template's system with this module's variables added.

    A simulation with a reform builds the reform's system, and its baseline's,
    from its ``default_tax_benefit_system`` class, so the variables are added
    in ``__init__``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for variable in [
            *SOURCES,
            *READERS,
            yearly_tenure,
            pay_and_grant,
            pay_if_granted,
            owner_by_add,
            owner_by_divide,
            owner_twelfth,
            *PROBES,
        ]:
            self.add_variable(variable)


@lru_cache(maxsize=None)
def tax_benefit_system() -> ProbeSystem:
    return ProbeSystem()


class ProbeMicrosimulation(Microsimulation):
    default_tax_benefit_system = ProbeSystem
    default_tax_benefit_system_instance = tax_benefit_system()
    default_calculation_period = DATASET_YEAR


class ProbeSimulation(Simulation):
    default_tax_benefit_system = ProbeSystem
    default_tax_benefit_system_instance = tax_benefit_system()
    default_calculation_period = DATASET_YEAR


class NoChange(Reform):
    """A reform that changes nothing, so the simulation has a baseline."""

    def apply(self):
        pass


def example_dataset(inputs: Inputs = EXAMPLE) -> Dataset:
    return Dataset.from_dataframe(dataframe(inputs), DATASET_YEAR)


def microsimulation(**options) -> ProbeMicrosimulation:
    return ProbeMicrosimulation(dataset=example_dataset(), **options)


def simulation(**options) -> ProbeSimulation:
    return ProbeSimulation(dataset=example_dataset(), **options)
