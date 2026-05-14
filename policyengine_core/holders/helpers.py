import logging

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.holders.holder import Holder
from policyengine_core.periods import Period

log = logging.getLogger(__name__)


def get_input_branch(holder: Holder) -> str:
    simulation = getattr(holder, "simulation", None)
    user_input_contexts = getattr(simulation, "_user_input_contexts", None)
    if user_input_contexts:
        return user_input_contexts[-1]
    return "default"


def get_stored_array(holder: Holder, period: Period, branch_name: str) -> ArrayLike:
    return holder._get_array_from_storage(period, branch_name)


def set_input_dispatch_by_period(holder: Holder, period: Period, array: ArrayLike):
    """
    This function can be declared as a ``set_input`` attribute of a variable.

    In this case, the variable will accept inputs on larger periods that its definition period, and the value for the larger period will be applied to all its subperiods.

    To read more about ``set_input`` attributes, check the `documentation <https://openfisca.org/doc/coding-the-legislation/35_periods.html#set-input-automatically-process-variable-inputs-defined-for-periods-not-matching-the-definition-period>`_.
    """
    array = holder._to_array(array)

    period_size = period.size
    period_unit = period.unit

    if holder.variable.definition_period == periods.MONTH:
        cached_period_unit = periods.MONTH
    elif holder.variable.definition_period == periods.YEAR:
        cached_period_unit = periods.YEAR
    else:
        raise ValueError(
            "set_input_dispatch_by_period can be used only for yearly or monthly variables."
        )

    after_instant = period.start.offset(period_size, period_unit)

    # Cache the input data, skipping the existing cached months
    branch_name = get_input_branch(holder)
    sub_period = period.start.period(cached_period_unit)
    while sub_period.start < after_instant:
        existing_array = get_stored_array(holder, sub_period, branch_name)
        if existing_array is None:
            holder._set(sub_period, array, branch_name)
        else:
            # The array of the current sub-period is reused for the next ones.
            # TODO: refactor or document this behavior
            array = existing_array
        sub_period = sub_period.offset(1)


def set_input_divide_by_period(holder: Holder, period: Period, array: ArrayLike):
    """
    This function can be declared as a ``set_input`` attribute of a variable.

    In this case, the variable will accept inputs on larger periods that its definition period, and the value for the larger period will be divided between its subperiods.

    To read more about ``set_input`` attributes, check the `documentation <https://openfisca.org/doc/coding-the-legislation/35_periods.html#set-input-automatically-process-variable-inputs-defined-for-periods-not-matching-the-definition-period>`_.
    """
    if not isinstance(array, numpy.ndarray):
        array = numpy.array(array)
    period_size = period.size
    period_unit = period.unit

    if holder.variable.definition_period == periods.MONTH:
        cached_period_unit = periods.MONTH
    elif holder.variable.definition_period == periods.YEAR:
        cached_period_unit = periods.YEAR
    else:
        raise ValueError(
            "set_input_divide_by_period can be used only for yearly or monthly variables."
        )

    after_instant = period.start.offset(period_size, period_unit)

    # Count the number of elementary periods to change, and the difference with what is already known.
    branch_name = get_input_branch(holder)
    remaining_array = array.copy()
    sub_period = period.start.period(cached_period_unit)
    sub_periods_count = 0
    while sub_period.start < after_instant:
        existing_array = get_stored_array(holder, sub_period, branch_name)
        if existing_array is not None:
            remaining_array -= existing_array
        else:
            sub_periods_count += 1
        sub_period = sub_period.offset(1)

    # Cache the input data
    if sub_periods_count > 0:
        divided_array = remaining_array / sub_periods_count
        sub_period = period.start.period(cached_period_unit)
        while sub_period.start < after_instant:
            if get_stored_array(holder, sub_period, branch_name) is None:
                holder._set(sub_period, divided_array, branch_name)
            sub_period = sub_period.offset(1)
    elif not (remaining_array == 0).all():
        raise ValueError(
            "Inconsistent input: variable {0} has already been set for all months contained in period {1}, and value {2} provided for {1} doesn't match the total ({3}). This error may also be thrown if you try to call set_input twice for the same variable and period.".format(
                holder.variable.name, period, array, array - remaining_array
            )
        )
