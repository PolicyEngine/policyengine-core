import logging
from typing import List

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


def _is_input(
    holder: Holder, period: Period, branch_name: str, storage_name: str
) -> bool:
    """Whether the value stored for ``period`` under ``branch_name`` was set
    as an input, rather than calculated by the simulation (a formula result,
    or a default, carried-over or uprated value).

    Use the same supplied-input tier record as reform replay. A cache write
    can leave a registry entry behind or hide an input in another tier, and
    its carry-over mark (``derived=False``) does not make it a supplied input.
    A holder with no simulation, or a simulation with no record, keeps the
    legacy behavior: every stored value counts as an input there.
    """
    simulation = getattr(holder, "simulation", None)
    input_keys = getattr(simulation, "_user_input_keys", None)
    if input_keys is None:
        return True
    period = holder._storage_period(period)
    if (holder.variable.name, branch_name, period) not in input_keys:
        return False
    holder._seed_input_storage(period, branch_name)
    return holder._stores_user_input(period, branch_name, storage_name)


def get_stored_input(holder: Holder, period: Period, branch_name: str) -> ArrayLike:
    """The supplied input under this exact branch, preferring memory to disk.

    A surviving disk input is still known when a non-input memory cache
    hides it; overlap cleanup removes that cache after the helper succeeds.
    """
    # Checked before reading, so that a calculated value a branch still
    # shares with its parent is not copied only to be replaced.
    for storage_name, storage in (
        ("memory", holder._memory_storage),
        ("disk", holder._disk_storage),
    ):
        if storage is not None and _is_input(holder, period, branch_name, storage_name):
            value = storage.get(period, branch_name)
            if value is not None:
                return value
    return None


def _store_input(
    holder: Holder, period: Period, array: ArrayLike, branch_name: str
) -> None:
    """Store ``array`` as the input for ``period``, replacing any value the
    simulation calculated for it.

    The value is added to the simulation's record of inputs even when the
    helper is called directly instead of through ``Holder.set_input``, and
    ``calculate``'s fast cache drops the value it held for the period.
    """
    simulation = getattr(holder, "simulation", None)
    input_keys = getattr(simulation, "_user_input_keys", None)
    # Let the holder normalize and record the supplied-input tier, including
    # direct helper calls. Keep a legacy holder without an input record in
    # that mode for the whole call, rather than creating a partial record.
    holder._set(period, array, branch_name, is_input=input_keys is not None)


def _evict_fast_cache(holder: Holder, changed_periods, branch_name: str) -> None:
    """Drop ``calculate``'s fast-cache entries for the variable's
    ``changed_periods``, if the simulation reads ``branch_name``.

    The fast cache holds what this simulation's ``calculate`` returned; a
    value stored under a branch it does not read changes none of that.
    """
    simulation = getattr(holder, "simulation", None)
    fast_cache = getattr(simulation, "_fast_cache", None)
    if not fast_cache:
        return
    get_visible_branch_names = getattr(simulation, "_get_visible_branch_names", None)
    if (
        get_visible_branch_names is not None
        and branch_name not in get_visible_branch_names()
    ):
        return
    for changed_period in changed_periods:
        fast_cache.pop((holder.variable.name, changed_period), None)


def _branches_read_with(holder: Holder, branch_name: str) -> List[str]:
    """``branch_name`` and the branches whose values the holder's simulation
    reads when ``branch_name`` stores none: its ancestors, then ``default``."""
    simulation = getattr(holder, "simulation", None)
    get_visible_branch_names = getattr(simulation, "_get_visible_branch_names", None)
    if get_visible_branch_names is not None:
        visible_branch_names = get_visible_branch_names()
        if branch_name in visible_branch_names:
            return visible_branch_names
    return [branch_name]


def _drop_calculated_overlapping(
    holder: Holder, period: Period, branch_name: str
) -> None:
    """Drop what the simulation calculated for the variable over ``period``,
    once a helper has set inputs for its sub-periods.

    Besides the sub-periods the helper replaced, the simulation may hold
    values it calculated from them: ``calculate_add`` caches the sum of a
    variable over a longer period (a monthly variable's ``2013``), and
    ``calculate_divide`` a twelfth of a yearly variable at a month. Any such
    value for a period that overlaps ``period`` is out of date, and
    ``calculate`` would return it instead of adding up or dividing the
    input. Inputs are kept.

    This holder's storage is searched under ``branch_name`` and under the
    branches read with it (see ``_branches_read_with``): a branch starts with
    its own index of what its parent calculated, and reads those values
    until they are dropped. The simulation the branch was created from keeps
    its values.
    """
    branch_names = set(_branches_read_with(holder, branch_name))
    dropped_periods = set()
    # Memory keys are "{branch}:{period}"; branch names cannot contain ":".
    # Disk keys are "{branch}_{period}"; branch names can contain "_", but
    # period strings cannot, so the period follows the last "_".
    memory = holder._memory_storage
    for key in list(memory._arrays):
        stored_branch_name, period_string = key.split(":", 1)
        if _is_calculated_overlapping(
            holder, stored_branch_name, period_string, branch_names, period, "memory"
        ):
            del memory._arrays[key]
            dropped_periods.add(_period_from_storage_key(period_string))
    memory._stop_sharing_dropped_keys()
    memory._unmark_dropped_keys()
    disk = holder._disk_storage
    if disk is not None:
        for key in list(disk._files):
            stored_branch_name, period_string = key.rsplit("_", 1)
            if _is_calculated_overlapping(
                holder, stored_branch_name, period_string, branch_names, period, "disk"
            ):
                del disk._files[key]
                dropped_periods.add(_period_from_storage_key(period_string))
        disk._derived.intersection_update(disk._files)
    _evict_fast_cache(holder, dropped_periods, branch_name)


def _period_from_storage_key(period_string: str) -> Period:
    """Parse a stored period whose early year may have lost leading zeros."""
    components = period_string.split(":")
    date_index = 1 if len(components) > 1 else 0
    date = components[date_index].split("-")
    date[0] = date[0].zfill(4)
    components[date_index] = "-".join(date)
    return periods.period(":".join(components))


def _is_calculated_overlapping(
    holder: Holder,
    stored_branch_name: str,
    period_string: str,
    branch_names: set,
    period: Period,
    storage_name: str,
) -> bool:
    if stored_branch_name not in branch_names:
        return False
    try:
        stored_period = _period_from_storage_key(period_string)
    except ValueError:
        # Not a key ``put`` wrote (say, a file ``restore`` found).
        return False
    if stored_period.unit == periods.ETERNITY:
        return False
    if stored_period.start > period.stop or stored_period.stop < period.start:
        return False
    return not _is_input(holder, stored_period, stored_branch_name, storage_name)


def set_input_dispatch_by_period(holder: Holder, period: Period, array: ArrayLike):
    """
    This function can be declared as a ``set_input`` attribute of a variable.

    In this case, the variable will accept inputs on larger periods that its definition period, and the value for the larger period will be applied to all its subperiods.

    Only inputs count as already known. A sub-period whose value the
    simulation calculated (a formula result, or a default, carried-over or
    uprated value) takes the new input like a sub-period with no value, so
    the result does not depend on what was calculated before the input was
    set.

    A sub-period that already has an input keeps it, and that input, not the
    value given for the larger period, is applied to the sub-periods after
    it that have none: with ``3`` set for March, setting ``7`` for the year
    gives ``7`` in January and February and ``3`` from March to December.
    The value of a stock known at some month holds for the rest of the
    period. (Setting the two inputs in the other order gives ``7`` in every
    month but March.)

    To read more about ``set_input`` attributes, check the `documentation <https://openfisca.org/doc/coding-the-legislation/35_periods.html#set-input-automatically-process-variable-inputs-defined-for-periods-not-matching-the-definition-period>`_.
    """
    array = holder._to_array(array, validate_nan=True)

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

    # Store the input data, skipping the sub-periods that already have an input
    branch_name = get_input_branch(holder)
    sub_period = period.start.period(cached_period_unit)
    while sub_period.start < after_instant:
        existing_input = get_stored_input(holder, sub_period, branch_name)
        if existing_input is None:
            _store_input(holder, sub_period, array, branch_name)
        else:
            # The input of the current sub-period is applied to the next
            # ones (see the docstring).
            array = existing_input
        sub_period = sub_period.offset(1)
    _drop_calculated_overlapping(holder, period, branch_name)


def set_input_divide_by_period(holder: Holder, period: Period, array: ArrayLike):
    """
    This function can be declared as a ``set_input`` attribute of a variable.

    In this case, the variable will accept inputs on larger periods that its definition period, and the value for the larger period will be divided between its subperiods.

    Sub-periods that already have an input keep it, and what is left of the
    value is divided between the others. Only inputs count as already known.
    A sub-period whose value the simulation calculated (a formula result, or
    a default, carried-over or uprated value) takes its share of the new
    input like a sub-period with no value, so the result does not depend on
    what was calculated before the input was set.

    To read more about ``set_input`` attributes, check the `documentation <https://openfisca.org/doc/coding-the-legislation/35_periods.html#set-input-automatically-process-variable-inputs-defined-for-periods-not-matching-the-definition-period>`_.
    """
    if not isinstance(array, numpy.ndarray):
        array = numpy.array(array)
    array = holder._to_array(array, validate_nan=True)
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

    # Find the elementary periods to change, and the difference with the inputs already known.
    branch_name = get_input_branch(holder)
    remaining_array = array.copy()
    sub_period = period.start.period(cached_period_unit)
    sub_periods_to_set = []
    while sub_period.start < after_instant:
        existing_input = get_stored_input(holder, sub_period, branch_name)
        if existing_input is not None:
            remaining_array -= existing_input
        else:
            sub_periods_to_set.append(sub_period)
        sub_period = sub_period.offset(1)

    # Store the input data
    if sub_periods_to_set:
        divided_array = remaining_array / len(sub_periods_to_set)
        for sub_period in sub_periods_to_set:
            _store_input(holder, sub_period, divided_array, branch_name)
    elif not (remaining_array == 0).all():
        raise ValueError(
            "Inconsistent input: variable {0} has already been set for all months contained in period {1}, and value {2} provided for {1} doesn't match the total ({3}). This error may also be thrown if you try to call set_input twice for the same variable and period.".format(
                holder.variable.name, period, array, array - remaining_array
            )
        )
    _drop_calculated_overlapping(holder, period, branch_name)
