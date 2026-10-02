"""Variables, simulations and a reference model for the ``set_input`` helper tests.

Shared by ``tests/core/test_set_input_helper_order.py`` (regressions) and
``tests/core/test_set_input_helper_order_property.py`` (properties).

The reference model (``Reference``) holds inputs only: it has no notion of a
value the simulation calculated. For each variable and branch it keeps the
input of each of the variable's own periods (months or years), and applies
the two helpers' rules to those:

- divide: the sub-periods that already have an input keep it, and what is
  left of the value given for the longer period is divided between the
  others (an error if there are none and something is left);
- dispatch: each sub-period with no input takes the value given for the
  longer period, or the input of the latest earlier sub-period that had one.

A simulation agrees with the model only if what it calculated before an
input was set has no effect on the inputs it holds afterwards.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.enums import Enum
from policyengine_core.experimental import MemoryConfig
from policyengine_core.holders import (
    set_input_dispatch_by_period,
    set_input_divide_by_period,
)
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable

COUNT = 2
BRANCH_NAME = "what_if"


class Status(Enum):
    none = "None"
    some = "Some"
    all = "All"


class flow_m(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly flow input with no formula (divided by default)"


class count_m(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly integer input with no formula (dispatched by default)"


class flag_m(Variable):
    value_type = bool
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly boolean input with no formula (dispatched by default)"


class status_m(Variable):
    value_type = Enum
    possible_values = Status
    default_value = Status.none
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly enum input with no formula (dispatched by default)"


class dispatched_flow_m(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    set_input = set_input_dispatch_by_period
    label = "Monthly flow input that repeats a longer period's value in each month"


class formula_flow_m(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly flow with a formula that reads no other variable"

    def formula(person, period, parameters):
        return person.filled_array(10.0 + period.start.month)


class flow_y(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly flow input with no formula (divided by default)"


class count_y(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly integer input with no formula (dispatched by default)"


VARIABLES = [
    flow_m,
    count_m,
    flag_m,
    status_m,
    dispatched_flow_m,
    formula_flow_m,
    flow_y,
    count_y,
]
NAMES = [variable.__name__ for variable in VARIABLES]
MONTHLY = [
    name
    for name in NAMES
    if getattr(globals()[name], "definition_period") == periods.MONTH
]
YEARLY = [name for name in NAMES if name not in MONTHLY]
DIVIDED = ["flow_m", "formula_flow_m", "flow_y"]
#: Variables whose plain read over a longer (or shorter) period is a sum (or
#: a twelfth) that ``calculate`` caches at that period.
FLOWS = ["flow_m", "dispatched_flow_m", "formula_flow_m", "flow_y"]

MONTHS = [f"{year}-{month:02d}" for year in (2013, 2014) for month in range(1, 13)]
YEARS = ["2013", "2014"]

#: The variable's own periods an input can be set for directly.
OWN_PERIODS = {name: (MONTHS if name in MONTHLY else YEARS) for name in NAMES}
#: Longer periods, which go through the variable's ``set_input`` helper.
HELPER_PERIODS = {
    name: (
        ["2013", "2014", "year:2013:2"]
        if name in MONTHLY
        else ["month:2013-01:12", "month:2014-01:12", "month:2013-01:24"]
    )
    for name in NAMES
}
#: Periods read with a plain ``calculate``: the variable's own, with one on
#: each side of the two years inputs are set in.
OWN_READS = {
    name: (
        ["2012-12"] + MONTHS + ["2015-01"]
        if name in MONTHLY
        else ["2012"] + YEARS + ["2015"]
    )
    for name in NAMES
}
#: Periods of another size a plain ``calculate`` accepts: a monthly
#: variable over one or two years (its sum if a flow, else its last month),
#: a yearly one at a month (a twelfth if a flow, else the year's value).
OTHER_READS = {
    name: (
        ["2013", "2014", "year:2013:2"]
        if name in MONTHLY
        else ["2013-05", "2014-01", "2014-12"]
    )
    for name in NAMES
}


def build_system() -> CountryTaxBenefitSystem:
    system = CountryTaxBenefitSystem()
    system.add_variables(*VARIABLES)
    return system


def build_simulation(system=None, on_disk: bool = False):
    """A simulation of ``COUNT`` people with no inputs.

    With ``on_disk``, every value is stored on disk rather than in memory.
    """
    simulation = SimulationBuilder().build_default_simulation(
        system or build_system(), count=COUNT
    )
    if on_disk:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulation.memory_config = MemoryConfig(max_memory_occupation=0)
        # A holder takes its storage from the configuration it is created
        # under, and building the simulation creates them all: drop the
        # (still empty) ones of the fixture's variables.
        for name in NAMES:
            del simulation.get_variable_population(name)._holders[name]
    return simulation


def sub_periods(name: str, period: str) -> List[str]:
    """The variable's own periods within ``period``, as a helper walks them."""
    unit = periods.MONTH if name in MONTHLY else periods.YEAR
    period = periods.period(period)
    after = period.start.offset(period.size, period.unit)
    result = []
    sub_period = period.start.period(unit)
    while sub_period.start < after:
        result.append(str(sub_period))
        sub_period = sub_period.offset(1)
    return result


def _dtype(name: str):
    return {
        "flow_m": np.float32,
        "dispatched_flow_m": np.float32,
        "formula_flow_m": np.float32,
        "flow_y": np.float32,
        "count_m": np.int32,
        "count_y": np.int32,
        "flag_m": np.bool_,
        "status_m": np.int16,
    }[name]


def as_input(name: str, values) -> np.ndarray:
    """``values`` as the array ``set_input`` takes for the variable."""
    if name == "status_m":
        return np.array(values, dtype=str)
    return np.array(values, dtype=_dtype(name))


def _stored(name: str, values) -> np.ndarray:
    """``values`` as the simulation stores them."""
    if name == "status_m":
        return np.array([Status[value].index for value in values], dtype=np.int16)
    return np.array(values, dtype=_dtype(name))


class Reference:
    """Inputs-only model of ``set_input`` (see the module docstring)."""

    def __init__(self):
        # (name, branch, own period) -> array
        self.inputs: Dict[Tuple[str, str, str], np.ndarray] = {}

    def copy(self) -> "Reference":
        other = Reference()
        other.inputs = dict(self.inputs)
        return other

    def set_input(
        self, name: str, period: str, values, branch: str = "default"
    ) -> Optional[str]:
        """Apply an input; return ``"ValueError"`` if the helper refuses it."""
        array = _stored(name, values)
        if period in OWN_PERIODS[name]:
            self.inputs[(name, branch, period)] = array
            return None
        own = sub_periods(name, period)
        if name in DIVIDED:
            remaining = array.copy()
            free = []
            for sub_period in own:
                known = self.inputs.get((name, branch, sub_period))
                if known is not None:
                    remaining -= known
                else:
                    free.append(sub_period)
            if free:
                share = (remaining / len(free)).astype(array.dtype)
                for sub_period in free:
                    self.inputs[(name, branch, sub_period)] = share
            elif not (remaining == 0).all():
                return "ValueError"
            return None
        for sub_period in own:
            known = self.inputs.get((name, branch, sub_period))
            if known is None:
                self.inputs[(name, branch, sub_period)] = array
            else:
                array = known
        return None

    def value(self, name: str, period: str, branch: str = "default") -> np.ndarray:
        """What a plain ``calculate`` of one of the variable's own periods
        returns: the branch's input, else the default branch's, else the
        formula result or the default value."""
        for branch_name in dict.fromkeys([branch, "default"]):
            known = self.inputs.get((name, branch_name, period))
            if known is not None:
                return known
        if name == "formula_flow_m":
            month = periods.period(period).start.month
            return np.full(COUNT, 10.0 + month, dtype=np.float32)
        return np.zeros(COUNT, dtype=_dtype(name))


def apply_input(simulation, name: str, period: str, values) -> Optional[str]:
    """``simulation.set_input``; return the error's type name if it raises."""
    try:
        simulation.set_input(name, period, as_input(name, values))
    except ValueError as error:
        return type(error).__name__
    return None


def read(simulation, name: str, period: str) -> np.ndarray:
    """A plain ``calculate``, as a plain array (enums as their indices)."""
    return np.asarray(simulation.calculate(name, period))


def read_all(simulation, names=NAMES, other_periods_of=NAMES) -> dict:
    """Every read of ``names`` in a fixed order: each variable's own periods,
    then, for those in ``other_periods_of``, the periods of another size."""
    result = {}
    for name in names:
        for period in OWN_READS[name]:
            result[(name, period)] = read(simulation, name, period)
    for name in names:
        if name in other_periods_of:
            for period in OTHER_READS[name]:
                result[(name, period)] = read(simulation, name, period)
    return result


def input_record(simulation) -> set:
    """The simulation's record of inputs for the fixture's variables."""
    return {
        (name, branch, str(period))
        for name, branch, period in simulation._user_input_keys
        if name in NAMES
    }


def same_arrays(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(left, right)
    )
