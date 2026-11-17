from contextvars import ContextVar, Token
from typing import Dict, Optional, Tuple, Type

from microdf import MicroDataFrame, MicroSeries
import numpy as np
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import EnumArray
from policyengine_core.periods import Period
from policyengine_core.periods import period as get_period
from policyengine_core.periods.config import MONTH, YEAR
from policyengine_core.simulations.simulation import Simulation
from policyengine_core.types import ArrayLike


class _RunningCall:
    """The outermost ``Microsimulation`` calculate call running in a context."""

    __slots__ = ("running",)

    def __init__(self):
        self.running = True


# Set while a ``Microsimulation``'s ``calculate``, ``calculate_add`` or
# ``calculate_divide`` call runs, by the outermost such call in the context.
# Core reads variables for its own use through those same methods (the terms
# of ``adds``, a ``defined_for`` mask, each month of a monthly variable read
# over a year, the terms of ADD and DIVIDE), as do formulas. A call that
# arrives while this is set comes from inside a calculation, and its defaults
# give a plain array with enum codes. A thread started meanwhile has its own
# empty context, so its calls count as calls from outside; a context copied
# meanwhile sees the call until it ends, when ``running`` turns false.
_running_call: ContextVar[Optional[_RunningCall]] = ContextVar(
    "_microsimulation_running_call", default=None
)


def _inside_a_calculation() -> bool:
    call = _running_call.get()
    return call is not None and call.running


def _start_call() -> Optional[Tuple[_RunningCall, Token]]:
    """Record an outermost call; return what ``_end_call`` needs, or ``None``."""
    if _inside_a_calculation():
        return None
    call = _RunningCall()
    return call, _running_call.set(call)


def _end_call(started: Optional[Tuple[_RunningCall, Token]]) -> None:
    if started is None:
        return
    call, token = started
    call.running = False
    _running_call.reset(token)


def _decode(values: ArrayLike) -> ArrayLike:
    """Decode enum codes to member names.

    The twelfth of a code that ``calculate_divide`` returns over a month names
    no member, so it is returned as it is, as a ``Simulation`` returns it.
    """
    if isinstance(values, EnumArray) and np.issubdtype(values.dtype, np.integer):
        return values.decode_to_str()
    return values


class Microsimulation(Simulation):
    """A `Simulation` whose entities use weights to represent larger populations.

    ``calculate``, ``calculate_add`` and ``calculate_divide`` return a
    weighted ``MicroSeries`` to a caller outside a calculation, and
    ``calculate`` decodes enums to member names for it. A call made while
    another of these calls runs (by core, or by a formula) returns a plain
    array, with enums left as codes, as a ``Simulation`` does: core and
    formulas use the values, not the weights. Passing ``use_weights`` or
    ``decode_enums`` overrides either default.
    """

    def get_weights(
        self, variable_name: str, period: Period, map_to: str = None
    ) -> ArrayLike:
        """Return weights for a variable's entity from ``household_weight``."""
        time_period = get_period(period)
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )
        entity_key = map_to or variable.entity.key
        weight_variable = self.tax_benefit_system.get_variable(
            "household_weight", check_existence=True
        )
        weight_period = None

        if time_period.unit == weight_variable.definition_period:
            weight_period = time_period
        elif (time_period.unit == MONTH) and (
            weight_variable.definition_period == YEAR
        ):
            # Common use-case. To-do: implement others if needed.
            weight_period = time_period.this_year

        if weight_period is None:
            return None

        household_weights = self.calculate(
            "household_weight",
            weight_period,
            use_weights=False,
            decode_enums=False,
        )
        return self._project_household_weights(household_weights, entity_key)

    def _project_household_weights(
        self,
        household_weights: ArrayLike,
        entity_key: str,
    ) -> ArrayLike:
        household_entity = self.tax_benefit_system.get_variable(
            "household_weight", check_existence=True
        ).entity.key
        if entity_key == household_entity:
            return household_weights

        household_population = self.populations[household_entity]
        person_weights = household_population.project(household_weights)
        if entity_key == self.tax_benefit_system.person_entity.key:
            return person_weights

        return self.populations[entity_key].value_from_first_person(person_weights)

    def calculate(
        self,
        variable_name: str,
        period: Period = None,
        map_to: str = None,
        use_weights: Optional[bool] = None,
        decode_enums: Optional[bool] = None,
    ) -> MicroSeries:
        """Calculate ``variable_name`` for ``period``.

        ``use_weights`` and ``decode_enums`` default to ``True`` for a call
        from outside a calculation and to ``False`` for a call made inside one
        (see ``Microsimulation``).
        """
        if use_weights is None or decode_enums is None:
            outside = not _inside_a_calculation()
            if use_weights is None:
                use_weights = outside
            if decode_enums is None:
                decode_enums = outside
        if period is not None and not isinstance(period, Period):
            period = get_period(period)
        elif period is None and self.default_calculation_period is not None:
            period = get_period(self.default_calculation_period)
        started = _start_call()
        try:
            values = super().calculate(variable_name, period, map_to, decode_enums)
        finally:
            _end_call(started)
        if not use_weights:
            return values
        weights = self.get_weights(variable_name, period, map_to)
        return MicroSeries(np.array(values), weights=weights)

    def calculate_add(
        self,
        variable_name: str,
        period: Period = None,
        map_to: str = None,
        use_weights: Optional[bool] = None,
        decode_enums: Optional[bool] = None,
    ) -> MicroSeries:
        """Sum ``variable_name`` over the subperiods of ``period``.

        ``use_weights`` defaults to ``True`` for a call from outside a
        calculation and to ``False`` for a call made inside one (see
        ``Microsimulation``). ``decode_enums`` defaults to ``False``.
        """
        if use_weights is None:
            use_weights = not _inside_a_calculation()
        if decode_enums is None:
            decode_enums = False
        # ``Population.__call__`` passes ``decode_enums`` with every ADD or
        # DIVIDE read in a formula. The terms are read inside this call, so
        # they are enum codes; the sum is decoded once, here.
        started = _start_call()
        try:
            values = super().calculate_add(
                variable_name, period, decode_enums=decode_enums
            )
        finally:
            _end_call(started)
        if decode_enums:
            values = _decode(values)
        if map_to is not None:
            source_entity = self.tax_benefit_system.get_variable(
                variable_name, check_existence=True
            ).entity.key
            values = self.map_result(np.array(values), source_entity, map_to)
        if not use_weights:
            return values
        weights = self.get_weights(variable_name, period, map_to)
        return MicroSeries(np.array(values), weights=weights)

    def calculate_divide(
        self,
        variable_name: str,
        period: Period = None,
        map_to: str = None,
        use_weights: Optional[bool] = None,
        decode_enums: Optional[bool] = None,
    ) -> MicroSeries:
        """Divide the yearly value of ``variable_name`` to match ``period``.

        ``use_weights`` defaults to ``True`` for a call from outside a
        calculation and to ``False`` for a call made inside one (see
        ``Microsimulation``). ``decode_enums`` defaults to ``False``.
        """
        if use_weights is None:
            use_weights = not _inside_a_calculation()
        if decode_enums is None:
            decode_enums = False
        # Over a year, ``Simulation.calculate_divide`` reads the variable with
        # ``self.calculate``, inside this call, so it gets enum codes; the
        # result is decoded once, here.
        started = _start_call()
        try:
            values = super().calculate_divide(
                variable_name, period, decode_enums=decode_enums
            )
        finally:
            _end_call(started)
        if decode_enums:
            values = _decode(values)
        if map_to is not None:
            source_entity = self.tax_benefit_system.get_variable(
                variable_name, check_existence=True
            ).entity.key
            values = self.map_result(np.array(values), source_entity, map_to)
        if not use_weights:
            return values
        weights = self.get_weights(variable_name, period, map_to)
        return MicroSeries(np.array(values), weights=weights)

    def calculate_dataframe(
        self,
        variable_names: list,
        period: Period = None,
        map_to: str = None,
        use_weights: bool = True,
    ) -> MicroDataFrame:
        if period is not None and not isinstance(period, Period):
            period = get_period(period)
        elif period is None and self.default_calculation_period is not None:
            period = get_period(self.default_calculation_period)
        values = super().calculate_dataframe(variable_names, period, map_to)
        if not use_weights:
            return values
        weights = self.get_weights(variable_names[0], period, map_to)
        return MicroDataFrame(values, weights=weights)
