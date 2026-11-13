import datetime as dt
import math
import numbers
import sys
from fractions import Fraction

from numpy import ceil, floor

# rrule is purposely imported this way to allow for programmatic
# calling of rrule.YEARLY, rrule.MONTHLY, and rrule.DAILY
from dateutil import rrule
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from datetime import datetime
from typing import Callable, Optional, Union

from policyengine_core.parameters.operations.get_parameter import get_parameter
from policyengine_core.parameters.parameter import Parameter
from policyengine_core.parameters.parameter_at_instant import (
    ParameterAtInstant,
)
from policyengine_core.parameters.parameter_node import ParameterNode
from policyengine_core.periods import Instant, Period, instant

#: Keys a base-mode ``uprating`` block may hold. Base mode rejects the others
#: (``start_instant``, ``at_defined_interval`` and the ``self`` keys) rather
#: than ignore them, and a misspelt key, such as ``rouding``, would otherwise
#: drop the rounding rule without a word.
BASE_MODE_UPRATING_KEYS = frozenset({"parameter", "base", "rounding"})

#: Keys of an ``uprating.base`` block.
UPRATING_BASE_KEYS = frozenset({"period", "value", "floor"})

#: Keys of a base-mode ``rounding`` mapping.
ROUNDING_KEYS = frozenset({"type", "interval"})

#: Base mode rounds the increase in exact rational arithmetic on the decimals
#: the base amount, the index values and the interval print as, so an index
#: written as 102.6 counts as 102.6, not as the binary float nearest it, and
#: an increase that decimal arithmetic puts on a multiple of the interval, or
#: on a half for ``nearest``, rounds from exactly there.
#:
#: Every recorded decimal is exact, including a computed float's printed
#: value. Its digit count cannot establish whether it contains index error.
#: Index producers must supply the intended precision; guessing here can
#: make a higher index produce a lower amount from a positive base.


def round_half_up(intervals: Fraction) -> int:
    """Round to the nearest whole number, taking an exact half upwards.

    Half up is the usual reading of "rounded to the nearest multiple", and
    the hand-coded projections PolicyEngine-US replaced with base mode used
    it (``math.floor(x / interval + 0.5)``). Chained mode uses Python's
    ``round``, which takes a half to the even number.
    """
    return math.floor(intervals + Fraction(1, 2))


#: The functions ``rounding.type`` names, which round a number of intervals,
#: held as an exact fraction, to a whole number.
INCREASE_ROUNDING_FUNCTIONS: dict[str, Callable[[Fraction], int]] = {
    "nearest": round_half_up,
    "upwards": math.ceil,
    "downwards": math.floor,
}


def uprate_parameters(root: ParameterNode) -> ParameterNode:
    """Uprates parameters according to their metadata.

    Args:
        root (ParameterNode): The root of the parameter tree.

    Returns:
        ParameterNode: The same root, with uprating applied to descendants.
    """

    parameters = [
        parameter
        for parameter in root.get_descendants()
        if isinstance(parameter, Parameter)
    ]
    parameter_paths = get_parameter_paths(root)

    for parameter in sort_parameters_by_uprating_dependencies(
        parameters,
        parameter_paths,
    ):
        uprate_parameter(parameter, root, parameter_paths)
    return root


def normalize_uprating_metadata(meta: Union[dict, str]) -> dict:
    if meta == "self":
        return dict(parameter="self")
    if isinstance(meta, str):
        return dict(parameter=meta)
    return meta


def get_uprating_dependency_name(parameter: Parameter) -> Optional[str]:
    meta = parameter.metadata.get("uprating")
    if meta is None:
        return None
    meta = normalize_uprating_metadata(meta)
    dependency_name = meta.get("parameter") if isinstance(meta, dict) else None
    if not isinstance(dependency_name, str):
        raise ValueError(
            f"Failed to uprate {parameter.name}: its uprating metadata "
            f"{meta!r} has no `parameter` that names an index."
        )
    if dependency_name == "self":
        return None
    return dependency_name


def get_parameter_paths(root: ParameterNode) -> dict[int, str]:
    paths = {}

    def visit_parameter_node(node: ParameterNode, path: str) -> None:
        for child_name, child in node.children.items():
            child_path = f"{path}.{child_name}" if path else child_name
            visit_child(child, child_path)

    def visit_child(child, path: str) -> None:
        if isinstance(child, Parameter):
            paths[id(child)] = path
        elif isinstance(child, ParameterNode):
            visit_parameter_node(child, path)
        else:
            brackets = getattr(child, "__dict__", {}).get("brackets")
            if brackets is not None:
                for index, bracket in enumerate(brackets):
                    visit_parameter_node(bracket, f"{path}[{index}]")

    visit_parameter_node(root, "")
    return paths


def join_parameter_path(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def get_parameter_scope_prefixes(
    parameter: Parameter,
    parameter_paths: dict[int, str],
) -> Optional[tuple[str, str]]:
    parameter_path = parameter_paths.get(id(parameter))
    if parameter_path is None or parameter_path == parameter.name:
        return None
    parameter_name_parts = parameter.name.split(".")
    parameter_path_parts = parameter_path.split(".")
    common_suffix_length = 0
    max_common_suffix_length = min(
        len(parameter_name_parts),
        len(parameter_path_parts),
    )
    while (
        common_suffix_length < max_common_suffix_length
        and parameter_name_parts[-common_suffix_length - 1]
        == parameter_path_parts[-common_suffix_length - 1]
    ):
        common_suffix_length += 1
    if common_suffix_length == 0:
        return None
    original_prefix = ".".join(parameter_name_parts[:-common_suffix_length])
    current_prefix = ".".join(parameter_path_parts[:-common_suffix_length])
    return original_prefix, current_prefix


def map_original_parameter_path(
    parameter_path: str,
    original_prefix: str,
    current_prefix: str,
) -> Optional[str]:
    if not original_prefix:
        return join_parameter_path(current_prefix, parameter_path)
    if parameter_path == original_prefix:
        return current_prefix
    original_prefix_with_separator = f"{original_prefix}."
    if parameter_path.startswith(original_prefix_with_separator):
        return join_parameter_path(
            current_prefix,
            parameter_path[len(original_prefix_with_separator) :],
        )
    return None


def get_scoped_uprating_dependency_names(
    parameter: Parameter,
    dependency_name: str,
    parameter_paths: dict[int, str],
) -> list[str]:
    parameter_path = parameter_paths.get(id(parameter))
    if parameter_path is None or parameter_path == parameter.name:
        return [dependency_name]

    dependency_names = []

    def add_dependency_name(candidate: Optional[str]) -> None:
        if candidate and candidate not in dependency_names:
            dependency_names.append(candidate)

    scope_prefixes = get_parameter_scope_prefixes(parameter, parameter_paths)
    if scope_prefixes is not None:
        original_prefix, current_prefix = scope_prefixes
        add_dependency_name(
            map_original_parameter_path(
                dependency_name,
                original_prefix,
                current_prefix,
            )
        )

    add_dependency_name(dependency_name)
    return dependency_names


def get_parameter_lookup_names(
    parameter: Parameter,
    parameter_paths: dict[int, str],
) -> set[str]:
    parameter_path = parameter_paths.get(id(parameter))
    if parameter_path is None:
        return {parameter.name}
    return {parameter_path}


def get_uprating_parameter(
    root: ParameterNode,
    parameter: Parameter,
    dependency_name: str,
    parameter_paths: dict[int, str],
) -> Parameter:
    for scoped_dependency_name in get_scoped_uprating_dependency_names(
        parameter,
        dependency_name,
        parameter_paths,
    ):
        try:
            return get_parameter(root, scoped_dependency_name)
        except ValueError:
            continue
    return get_parameter(root, dependency_name)


def sort_parameters_by_uprating_dependencies(
    parameters: list[Parameter],
    parameter_paths: Optional[dict[int, str]] = None,
) -> list[Parameter]:
    if parameter_paths is None:
        parameter_paths = {}
    parameters_to_uprate = [
        parameter
        for parameter in parameters
        if parameter.metadata.get("uprating") is not None
    ]
    parameters_by_name = {}
    for parameter in parameters_to_uprate:
        for name in get_parameter_lookup_names(parameter, parameter_paths):
            parameters_by_name.setdefault(name, []).append(parameter)
    ordered_parameters = []
    visited = set()
    visiting = []
    visiting_ids = set()

    def visit(parameter: Parameter):
        parameter_id = id(parameter)
        if parameter_id in visited:
            return
        if parameter_id in visiting_ids:
            cycle_start = next(
                index
                for index, visiting_parameter in enumerate(visiting)
                if id(visiting_parameter) == parameter_id
            )
            cycle = visiting[cycle_start:] + [parameter]
            raise ValueError(
                "Cyclic uprating dependency detected: "
                + " -> ".join(parameter.name for parameter in cycle)
            )
        visiting.append(parameter)
        visiting_ids.add(parameter_id)
        dependency_name = get_uprating_dependency_name(parameter)
        dependency_parameters = []
        if dependency_name is not None:
            for scoped_dependency_name in get_scoped_uprating_dependency_names(
                parameter,
                dependency_name,
                parameter_paths,
            ):
                dependency_parameters = parameters_by_name.get(
                    scoped_dependency_name,
                    [],
                )
                if dependency_parameters:
                    break
        for dependency in dependency_parameters:
            visit(dependency)
        visiting.pop()
        visiting_ids.remove(parameter_id)
        visited.add(parameter_id)
        ordered_parameters.append(parameter)

    for parameter in sorted(parameters_to_uprate, key=lambda p: p.name):
        visit(parameter)

    return ordered_parameters


def uprate_parameter(
    parameter: Parameter,
    root: ParameterNode,
    parameter_paths: Optional[dict[int, str]] = None,
) -> None:
    """Add projected values to a parameter according to its uprating metadata.

    Chained mode, the default, grows the parameter's last explicit value by the
    index's change since that value's instant and rounds the uprated amount.
    Base mode, chosen by an ``uprating.base`` block, computes each projected
    value from a fixed base amount and rounds the increase over it; see
    :func:`uprate_parameter_from_base`.

    Args:
        parameter: The parameter to uprate. Its ``values_list`` gains the
            projected values.
        root: The root of the parameter tree, where the index is looked up.
        parameter_paths: The parameter's path in a cloned subtree, which
            scopes the index lookup; see :func:`get_parameter_paths`.
    """
    if parameter_paths is None:
        parameter_paths = {}
    # Pull the uprating definition dict
    meta = normalize_uprating_metadata(parameter.metadata["uprating"])

    if isinstance(meta, dict) and "base" in meta:
        uprate_parameter_from_base(parameter, root, meta, parameter_paths)
        return

    # If param is "self", construct the uprating table
    if meta["parameter"] == "self":
        uprating_parameter = construct_uprater_self(
            parameter,
            meta,
        )
    # Otherwise, pull uprating table from YAML
    else:
        uprating_parameter = get_uprating_parameter(
            root,
            parameter,
            meta["parameter"],
            parameter_paths,
        )

    # If uprating with a set candence, ensure that all
    # required values are present
    cadence_meta = meta.get("at_defined_interval")
    cadence_options = {}
    if cadence_meta:
        cadence_options_test = [
            cadence_meta.get("start"),
            cadence_meta.get("end"),
            cadence_meta.get("enactment"),
        ]

        # Ensure that all options are properly defined
        if not all(cadence_options_test):
            raise SyntaxError(
                f"Failed to uprate {parameter.name} using cadence; start, end, and enactment must all be provided"
            )

        # Construct cadence options object
        cadence_options = construct_cadence_options(cadence_meta, parameter)

        # Ensure that end comes after start and enactment comes after end
        if cadence_options["end"] <= cadence_options["start"]:
            raise ValueError(
                f"Failed to uprate {parameter.name} using {uprating_parameter.name}: end must come after start"
            )
        if cadence_options["enactment"] <= cadence_options["end"]:
            raise ValueError(
                f"Failed to uprate {parameter.name} using {uprating_parameter.name}: enactment must come after end"
            )

        # Determine the first date from which to start uprating -
        # this should be the first application date (month, day)
        # following the last defined param value (not including the
        # final value)
        uprating_first_date: datetime = find_cadence_first(parameter, cadence_options)
        uprating_last_date: datetime = find_cadence_last(
            uprating_parameter, cadence_options
        )

        # Uprate data
        uprated_data = uprate_by_cadence(
            parameter,
            uprating_parameter,
            cadence_options,
            uprating_first_date,
            uprating_last_date,
            meta,
        )

        # Append uprated data to parameter values list
        parameter.values_list.extend(uprated_data)

    else:
        # Start from the latest value
        if "start_instant" in meta:
            last_instant = instant(meta["start_instant"])
        else:
            last_instant = instant(parameter.values_list[0].instant_str)

        # Pre-compute values that don't change in the loop
        value_at_start = parameter(last_instant)
        uprater_at_start = uprating_parameter(last_instant)

        if uprater_at_start is None:
            raise ValueError(
                f"Failed to uprate using {uprating_parameter.name} at {last_instant} for {parameter.name} because the uprating parameter is not defined at {last_instant}."
            )

        has_rounding = "rounding" in meta

        # For each defined instant in the uprating parameter
        for entry in uprating_parameter.values_list[::-1]:
            entry_instant = instant(entry.instant_str)
            # If the uprater instant is defined after the last parameter instant
            if entry_instant > last_instant:
                # Apply the uprater and add to the parameter
                uprater_at_entry = uprating_parameter(entry_instant)
                uprater_change = uprater_at_entry / uprater_at_start
                uprated_value = value_at_start * uprater_change
                if has_rounding:
                    uprated_value = round_uprated_value(meta, uprated_value)
                parameter.values_list.append(
                    ParameterAtInstant(
                        parameter.name,
                        entry.instant_str,
                        data=uprated_value,
                    )
                )
    # Whether using cadence or not, sort the parameter values_list
    parameter.values_list.sort(key=lambda x: x.instant_str, reverse=True)


def uprate_parameter_from_base(
    parameter: Parameter,
    root: ParameterNode,
    meta: dict,
    parameter_paths: Optional[dict[int, str]] = None,
) -> None:
    """Project a parameter from a fixed base amount (base mode).

    Statutes such as 26 U.S.C. 63(c)(7)(B)(ii) and 213(d)(10)(B) increase a
    stated dollar amount by the change in a price index since a base year and
    round the increase, not the increased amount. Base mode computes, for
    every instant ``t`` at which the uprating index has a value after the
    parameter's last explicit value (and no earlier than the base period)::

        base_value + round(base_value * (index(t) / index(base) - 1))

    ``rounding`` takes the same ``type`` and ``interval`` as in chained mode
    and applies them to the increase, computed exactly on the decimals the
    amounts and index values print as; ``nearest`` takes an exact half up
    (see :func:`project_from_base`). Without ``rounding`` the value is
    ``base_value * index(t) / index(base)``. Explicit values always win: base
    mode adds values only after the last of them. An index below its
    base-period level gives a value below the base unless ``base.floor`` is
    set, and an infinite base amount, such as a scale's ``-inf`` lowest
    threshold, stays infinite.

    The ``base`` block holds:

    * ``period`` (required): a year, such as ``2025``, or a date, such as
      ``1997-01-01`` (from Python, also an ``Instant``, or a ``Period``, read
      at its start). Base mode reads the index there, so it names the base
      in the index's own periods. Where an index's value at
      period ``Y`` stands for calendar year ``Y - 1``, a statute's base year
      ``Y - 1`` is ``period: Y``.
    * ``value`` (optional): the base amount. It defaults to the parameter's
      value at ``period``. When the parameter has a value there, a stated
      ``value`` must match it to one part in a billion, which catches a
      ``period`` one year off.
    * ``floor`` (optional, default ``false``): when ``true``, base mode counts
      the index as no lower than its base-period level, so the index ratio is
      at least 1 and a positive base never projects below itself. Statutes
      that add "the percentage (if any) by which" an index exceeds its
      base-year level, such as 26 U.S.C. 1(f)(3), floor the adjustment this
      way.

    Args:
        parameter: The parameter to uprate. Its ``values_list`` gains the
            projected values.
        root: The root of the parameter tree, where the index is looked up.
        meta: The parameter's uprating metadata, holding ``parameter``,
            ``base`` and optionally ``rounding``. It is read, never changed:
            a scale's brackets can share one dict.
        parameter_paths: The parameter's path in a cloned subtree, which
            scopes the index lookup; see :func:`get_parameter_paths`.

    Raises:
        ValueError: The metadata is malformed or combines ``base`` with
            ``start_instant``, ``at_defined_interval`` or ``self``; the index
            is undefined, not positive or not finite where base mode reads
            it; the base amount is missing, not a number, or differs from the
            parameter's value at the base period; the parameter has no values
            or its last value is null; or a projected value overflows.
    """
    if parameter_paths is None:
        parameter_paths = {}
    base_instant, stated_value, floor_at_base = parse_uprating_base(parameter, meta)
    rounding = (
        parse_increase_rounding(parameter, meta["rounding"])
        if "rounding" in meta
        else None
    )
    try:
        uprating_parameter = get_uprating_parameter(
            root,
            parameter,
            meta["parameter"],
            parameter_paths,
        )
    except ValueError as error:
        raise base_mode_error(parameter, f"its index: {error}") from error
    index_at_base = get_base_mode_index_value(
        parameter, uprating_parameter, base_instant
    )
    base_value = get_base_value(parameter, base_instant, stated_value)

    if not parameter.values_list:
        raise base_mode_error(parameter, "it has no values to project after.")
    latest = max(parameter.values_list, key=lambda value: value.instant_str)
    if latest.value is None:
        raise base_mode_error(
            parameter,
            f"its last value, at {latest.instant_str}, is null, which ends the "
            "parameter; base mode does not project past an explicit end.",
        )
    last_instant = instant(latest.instant_str)

    for entry in uprating_parameter.values_list[::-1]:
        entry_instant = instant(entry.instant_str)
        if entry_instant <= last_instant or entry_instant < base_instant:
            continue
        index_at_entry = get_base_mode_index_value(
            parameter, uprating_parameter, entry_instant
        )
        try:
            value = project_from_base(
                base_value,
                index_at_entry,
                index_at_base,
                rounding,
                floor=floor_at_base,
            )
        except OverflowError:
            value = math.copysign(math.inf, base_value)
        if not math.isfinite(value) and math.isfinite(base_value):
            raise base_mode_error(
                parameter,
                f"its projected value at {entry.instant_str} is {value!r}.",
            )
        parameter.values_list.append(
            ParameterAtInstant(parameter.name, entry.instant_str, data=value)
        )
    parameter.values_list.sort(key=lambda x: x.instant_str, reverse=True)


def base_mode_error(parameter: Parameter, reason: str) -> ValueError:
    return ValueError(f"Failed to uprate {parameter.name} from its base: {reason}")


def is_number(value) -> bool:
    """Whether ``value`` is a float other than NaN, or an int other than a
    bool that a float can hold."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= sys.float_info.max
    return isinstance(value, float) and not math.isnan(value)


def is_real_number(value) -> bool:
    """Whether ``value`` is a finite int or float other than a bool."""
    return is_number(value) and math.isfinite(value)


def parse_uprating_base(
    parameter: Parameter, meta: dict
) -> tuple[Instant, Optional[float], bool]:
    """Validate a base-mode ``uprating`` block and read its ``base``.

    Returns:
        The base instant; the stated base ``value`` (``None`` when the block
        leaves it to the parameter's value at the base period); and whether
        ``floor`` keeps the index ratio at or above 1.
    """
    unknown_keys = sorted(set(meta) - BASE_MODE_UPRATING_KEYS, key=str)
    if unknown_keys:
        raise base_mode_error(
            parameter,
            "base mode takes only `parameter`, `base` and `rounding`, not "
            + ", ".join(f"`{key}`" for key in unknown_keys)
            + ".",
        )
    if not isinstance(meta.get("parameter"), str):
        raise base_mode_error(parameter, "`parameter` names no index.")
    if meta["parameter"] == "self":
        raise base_mode_error(
            parameter,
            "`parameter: self` names no index for base mode to read.",
        )
    base = meta["base"]
    if not isinstance(base, dict):
        raise base_mode_error(
            parameter,
            "`base` must be a mapping with a `period` and an optional `value` "
            f"and `floor`, not {base!r}.",
        )
    unknown_base_keys = sorted(set(base) - UPRATING_BASE_KEYS, key=str)
    if unknown_base_keys:
        raise base_mode_error(
            parameter,
            "`base` takes only `period`, `value` and `floor`, not "
            + ", ".join(f"`{key}`" for key in unknown_base_keys)
            + ".",
        )
    if base.get("period") is None:
        raise base_mode_error(parameter, "`base` has no `period`.")
    period = base["period"]
    base_instant = None
    # YAML gives a year as an int and an unquoted date as a ``datetime.date``.
    if isinstance(period, (int, str, dt.date, Instant, Period)) and not isinstance(
        period, bool
    ):
        try:
            base_instant = instant(period)
            # ``instant`` accepts impossible dates, such as 2020-02-30 or year
            # 0, that fail only when something reads them as a date. A year
            # past the C int range overflows instead.
            base_instant.date
        except (AssertionError, OverflowError, TypeError, ValueError):
            base_instant = None
    if not isinstance(base_instant, Instant):
        raise base_mode_error(
            parameter,
            f"`base.period` {period!r} is not a period or instant, such as "
            "2025 or 2025-01-01.",
        )
    stated_value = base.get("value")
    if stated_value is not None and not is_real_number(stated_value):
        raise base_mode_error(
            parameter,
            f"`base.value` {stated_value!r} is not a finite number.",
        )
    floor_at_base = base.get("floor", False)
    if not isinstance(floor_at_base, bool):
        raise base_mode_error(
            parameter,
            f"`base.floor` {floor_at_base!r} is not `true` or `false`.",
        )
    return base_instant, stated_value, floor_at_base


def parse_increase_rounding(parameter: Parameter, rounding_config) -> tuple[float, str]:
    """Read a base-mode ``rounding`` block.

    It takes chained mode's forms: a mapping with a ``type`` (``nearest``,
    ``upwards`` or ``downwards``) and a positive ``interval``, or a bare
    interval, which rounds to the nearest multiple.

    Returns:
        The interval and the rounding type.
    """
    if isinstance(rounding_config, dict):
        unknown_keys = sorted(set(rounding_config) - ROUNDING_KEYS, key=str)
        if unknown_keys:
            raise base_mode_error(
                parameter,
                "`rounding` takes only `type` and `interval`, not "
                + ", ".join(f"`{key}`" for key in unknown_keys)
                + ".",
            )
        rounding_type = rounding_config.get("type")
        if not (
            isinstance(rounding_type, str)
            and rounding_type in INCREASE_ROUNDING_FUNCTIONS
        ):
            raise base_mode_error(
                parameter,
                f"`rounding.type` {rounding_type!r} is not one of "
                + ", ".join(f"`{name}`" for name in INCREASE_ROUNDING_FUNCTIONS)
                + ".",
            )
        interval = rounding_config.get("interval")
    else:
        rounding_type = "nearest"
        interval = rounding_config
    if not is_real_number(interval) or interval <= 0:
        raise base_mode_error(
            parameter,
            f"the rounding interval {interval!r} is not a positive number.",
        )
    return interval, rounding_type


def get_base_mode_index_value(
    parameter: Parameter, uprating_parameter: Parameter, at: Instant
) -> float:
    """Read the uprating index at ``at``, which must be finite and positive."""
    value = uprating_parameter(at)
    if not is_real_number(value) or value <= 0:
        raise base_mode_error(
            parameter,
            f"the uprating index {uprating_parameter.name} is {value!r} at {at}; "
            "base mode needs a finite, positive value there.",
        )
    return value


def get_base_value(
    parameter: Parameter, base_instant: Instant, stated_value: Optional[float]
) -> float:
    """The base amount: the stated ``value``, else the parameter's at the base.

    The parameter's own value may be infinite, as a scale's lowest threshold
    often is; :func:`project_from_base` keeps an infinite base as it is.
    """
    value_at_base = parameter(base_instant)
    if stated_value is None:
        if not is_number(value_at_base):
            raise base_mode_error(
                parameter,
                f"it has no numeric value at the base period {base_instant} "
                f"(it has {value_at_base!r}); state the base amount as "
                "`base.value`.",
            )
        return value_at_base
    if value_at_base is not None and not (
        is_real_number(value_at_base)
        and math.isclose(stated_value, value_at_base, rel_tol=1e-9)
    ):
        raise base_mode_error(
            parameter,
            f"`base.value` is {stated_value!r}, but the parameter's value at the "
            f"base period {base_instant} is {value_at_base!r}. `base.period` "
            "names the period whose value is the base amount, in the uprating "
            "index's periods; check it is not a year off. If a reform applied "
            "before uprating changed the base-period value, omit `base.value` "
            "so the base follows the reform.",
        )
    return stated_value


def exact_decimal(value: float) -> Fraction:
    """The exact rational value of the decimal ``value`` prints as.

    An int is exact already. A float becomes the shortest decimal that reads
    back as it, which for a value written in YAML is the value as written:
    102.6 becomes 1026/10, not the binary float a hair below it.
    """
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return Fraction(int(value))
    return Fraction(repr(float(value)))


def project_from_base(
    base_value: float,
    index_at_t: float,
    index_at_base: float,
    rounding: Optional[tuple[float, str]],
    floor: bool = False,
) -> float:
    """The base amount plus the (rounded) increase the index implies.

    With ``rounding``, the increase ``base_value * (index_at_t /
    index_at_base - 1)`` is computed exactly on the decimals the inputs print
    as (:func:`exact_decimal`), rounded to a whole number of intervals, and
    added to the base. Every recorded input is exact: a computed float is
    read at its printed decimal value, without guessing its intended index
    precision from its digit count. Without ``rounding``, the value is ``base_value *
    (index_at_t / index_at_base)`` in floating point, the order chained mode
    uses.

    An infinite base, such as a scale's ``-inf`` lowest threshold, stays
    infinite: the index ratio is positive, even where it underflows a float.

    Args:
        base_value: The base amount.
        index_at_t: The index at the projected period, finite and positive.
        index_at_base: The index at the base period, finite and positive.
        rounding: The interval and rounding type for the increase, or
            ``None`` for no rounding.
        floor: Whether to count the index as no lower than its base-period
            level, so the ratio is at least 1.

    Raises:
        OverflowError: The rounded amount is too large for a float.
    """
    if math.isinf(base_value):
        return float(base_value)
    if rounding is None:
        ratio = index_at_t / index_at_base
        if floor:
            ratio = max(ratio, 1.0)
        return float(base_value * ratio)
    interval, rounding_type = rounding
    ratio = exact_decimal(index_at_t) / exact_decimal(index_at_base)
    if floor:
        ratio = max(ratio, Fraction(1))
    base = exact_decimal(base_value)
    step = exact_decimal(interval)
    intervals = base * (ratio - 1) / step
    whole_intervals = INCREASE_ROUNDING_FUNCTIONS[rounding_type](intervals)
    return float(base + whole_intervals * step)


def round_uprated_value(meta: dict, uprated_value: float) -> float:
    rounding_config = meta["rounding"]
    if isinstance(rounding_config, float):
        interval = rounding_config
        rounding_fn = round
    elif isinstance(rounding_config, dict):
        interval = rounding_config["interval"]
        rounding_fn = dict(
            nearest=round,
            upwards=ceil,
            downwards=floor,
        )[rounding_config["type"]]
    uprated_value = rounding_fn(uprated_value / interval) * interval
    return uprated_value


def find_cadence_first(parameter: Parameter, cadence_options: dict) -> datetime:
    """
    Find first value to uprate. This should be the same (month, day) as
    the uprating enactment date, but occurring after the last value within
    the default parameter. This is overriden by the "effective" date option

    >>> if enactment: "0002-04-01", last value: "2022-10-01"
    "2023-04-01"
    >>> if enactment: "0002-04-01", last value: "2022-02-01"
    "2022-04-01"
    >>> if enactment: "0002-04-01", last value: "2022-04-01"
    "2023-04-01"

    """

    # Clone parameter's value list
    # param_values: ParameterNode = parameter.values_list.clone()
    param_values = []
    for i in range(len(parameter.values_list)):
        param_values.append(parameter.values_list[i].clone())

    # There's no guarantee of a particular order for the value list's items;
    # sort the list to ensure newest item is first, akin to defined order elsewhere in repo
    param_values.sort(key=lambda x: x.instant_str, reverse=True)

    # If an "effective" date is provided, return that;
    # note that cadence_options["effective"] is already of type
    # Instant within the options object
    if cadence_options.get("effective") is not None:
        return cadence_options["effective"]

    # Save the "interval", if provided, else set to "year"
    interval = None
    if cadence_options.get("interval") is not None:
        interval = cadence_options["interval"]
    else:
        interval = "year"

    # Pull of the first (newest) value and parse into datetime object
    newest_param: datetime = parse(param_values[0].instant_str)

    # Create an offset of one day; if the newest param date is the same as our
    # enactment date, (e.g., newest param is 2022-04-01 and enactment is 04-01),
    # we don't want to include 2022-04-01, but rrule inclusively applies the
    # dtstart arg, so we need to offset by one day when evaluating
    offset: relativedelta = relativedelta(days=1)
    offset_param = newest_param + offset

    # Conditionally determine settings to utilize
    # Set the month only if the interval is "year"
    month_option = cadence_options["enactment"].month if interval == "year" else None
    # Set the day if interval is "year" or "month", just not "day"
    day_option = cadence_options["enactment"].day if interval != "day" else None

    rrule_obj = rrule.rrule(
        freq=rrule.YEARLY,
        dtstart=offset_param,
        bymonth=month_option,
        bymonthday=day_option,
        count=1,
    )

    return rrule_obj[0]


def find_cadence_last(uprater: Parameter, cadence_options: dict) -> datetime:
    """
    Determine latest date to uprate until. This should be first (month, day) to
    occur after the last defined uprating end value

    >>> if enactment: "0002-04-01", end: "0001-10-01", last uprating value: "2022-10-01"
    "2023-04-01"
    >>> if enactment: "0003-04-01", end: "0001-10-01", last uprating value: "2022-10-01"
    "2024-04-01"
    """

    # Clone parameter's value list
    # param_values: ParameterNode = parameter.values_list.clone()
    uprater_values = []
    for i in range(len(uprater.values_list)):
        uprater_values.append(uprater.values_list[i].clone())

    # There's no guarantee of a particular order for the value list's items;
    # sort the list to ensure newest item is first, akin to defined order elsewhere in repo
    uprater_values.sort(key=lambda x: x.instant_str, reverse=True)

    # Save the "interval", if provided, else set to "year"
    interval = None
    if cadence_options.get("interval") is not None:
        interval = cadence_options["interval"]
    else:
        interval = "year"

    # Pull of the first (newest) value and parse into datetime object
    last_param: datetime = parse(uprater_values[0].instant_str)

    # Create an offset to allow us to search for one year, from one year
    # minus 1 day before the last uprater param, to the last uprater param;
    # this must be 1 year, minus 1 day, because rrule is inclusive of its
    # start date. This must also be done as 1 year, then 1 day, because
    # leap years are not handled differently when specifying a day-based offset,
    # resulting in 25% of offsets being one day short if calculated
    # using days only; further, this must be done with relativedelta and not
    # datetime's timedelta, which doesn't handle year-based offsets
    start_date: datetime = last_param - relativedelta(years=1) + relativedelta(days=1)

    # Conditionally determine settings to utilize
    # Set the month only if the interval is "year"
    month_option = cadence_options["end"].month if interval == "year" else None
    # Set the day if interval is "year" or "month", just not "day"
    day_option = cadence_options["end"].day if interval != "day" else None

    rrule_obj = rrule.rrule(
        freq=rrule.YEARLY,
        dtstart=start_date,
        until=last_param,
        bymonth=month_option,
        bymonthday=day_option,
    )

    # rrule can't look backward, so we generated an entire array of options
    # for a year; for monthly and daily intervals, we'll need the last item
    # in this array
    last_end: datetime = rrule_obj[-1]

    # Ensure that data exists for the "start" parameter for this "end,"
    # otherwise raise error
    end_start_diff: relativedelta = relativedelta(
        cadence_options["end"], cadence_options["start"]
    )
    last_start = last_end - end_start_diff
    if (uprater(instant(last_start.date()))) is None:
        raise ValueError(
            f"Error while uprating using {uprater.name}: no valid start, end pair found; did "
            + "you forget to add earlier values, or is there a long gap between start and end year?"
        )

    # We're seeking the last uprating enactment date, not end date; find gap between
    # end and enactment, find the date corresponding to that gap, then return that
    enactment_end_diff: relativedelta = relativedelta(
        cadence_options["enactment"], cadence_options["end"]
    )
    last_enactment: datetime = last_end + enactment_end_diff
    return last_enactment


def uprate_by_cadence(
    parameter: Parameter,
    uprating_parameter: Parameter,
    cadence_options: dict,
    first_date: datetime,
    last_date: datetime,
    meta: dict,
) -> list[ParameterAtInstant]:
    # Determine the frequency module to utilize within rrule
    interval = ""
    rrule_interval = ""
    if cadence_options.get("interval"):
        interval = cadence_options["interval"]
        rrule_interval = getattr(rrule, ((interval + "LY").upper()))
    else:
        interval = "year"
        rrule_interval = rrule.YEARLY

    # Generate a list of iterations
    iterations = rrule.rrule(freq=rrule_interval, dtstart=first_date, until=last_date)

    # Determine the offset between the first enactment
    # date and the first start and end date
    uprated_data: list[ParameterAtInstant] = []
    enactment_start_offset: relativedelta = relativedelta(
        cadence_options["enactment"], cadence_options["start"]
    )
    enactment_end_offset: relativedelta = relativedelta(
        cadence_options["enactment"], cadence_options["end"]
    )

    # Set a starting reference value to calculate against
    reference_value = parameter.get_at_instant(instant(first_date.date()))

    # For each entry (corresponding to an enactment date) in the iteration list...
    for enactment_date in iterations:
        # Calculate the start and end calculation dates
        start_calc_date: datetime = enactment_date - enactment_start_offset
        end_calc_date: datetime = enactment_date - enactment_end_offset

        # Find uprater value at cadence start
        start_val = uprating_parameter.get_at_instant(instant(start_calc_date.date()))

        # Find uprater value at cadence end
        end_val = uprating_parameter.get_at_instant(instant(end_calc_date.date()))

        # Ensure that earliest date exists within uprater
        if not start_val:
            raise ValueError(
                f"Failed to uprate {parameter.name} using {uprating_parameter.name}: uprater missing values at date {start_calc_date.date()}"
            )

        # Find difference of these values
        difference = end_val / start_val

        # Uprate value
        uprated_value = difference * reference_value
        if "rounding" in meta:
            uprated_value = round_uprated_value(meta, uprated_value)

        # Add uprated value to data list
        uprated_data.append(
            ParameterAtInstant(
                parameter.name, str(enactment_date.date()), data=uprated_value
            )
        )

        # Swap reference_value with new value
        reference_value = uprated_value

    return uprated_data


def construct_cadence_options(cadence_settings: dict, parameter: Parameter) -> dict:
    # Define all settings with fixed input options
    # so as to test that input is valid
    FIXED_OPTIONS = {"interval": ["year", "month", "day"]}

    # All of these will be converted to datetime objects
    DATE_KEYS = ["start", "end", "enactment", "effective"]

    # All of these will remain strings
    STRING_KEYS = ["interval"]

    cadence_options = {}
    for key in DATE_KEYS:
        if cadence_settings.get(key) is None:
            continue

        # Ensure that date is YYYY-MM-DD
        date_test = cadence_settings[key].split("-")
        if (
            len(date_test) != 3
            or len(date_test[0]) != 4
            or len(date_test[1]) != 2
            or len(date_test[2]) != 2
            or int(date_test[1]) > 12
            or int(date_test[2]) > 31
        ):
            raise SyntaxError(
                f"Unable to uprate {parameter.name} using setting '{key}': '{key}' must be in format 'YYYY-MM-DD'"
            )

        cadence_options[key] = parse(cadence_settings[key])

    for key in STRING_KEYS:
        if cadence_settings.get(key) is None:
            continue

        if FIXED_OPTIONS.get(key):
            value = cadence_settings.get(key)
            if str(value).lower() not in FIXED_OPTIONS[key]:
                valid_options = ", ".join(str(x) for x in FIXED_OPTIONS[key])
                raise SyntaxError(
                    f"Unable to uprate {parameter.name}: {key} in cadence settings contains invalid value; valid options are ({valid_options})"
                )

        cadence_options[key] = str(cadence_settings[key])

    return cadence_options


def construct_uprater_self(parameter: Parameter, meta: dict) -> Parameter:
    last_instant = instant(parameter.values_list[0].instant_str)
    start_instant = instant(
        meta.get(
            "from",
            last_instant.offset(-1, meta.get("interval", "year")),
        )
    )
    start = parameter(start_instant)
    end_instant = instant(meta.get("to", last_instant))
    end = parameter(end_instant)
    increase = end / start
    if "from" in meta:
        # This won't work for non-year periods, which are more complicated
        increase / (end_instant.year - start_instant.year)
    data = {}
    value = parameter.values_list[0].value
    for i in range(meta.get("number", 10)):
        data[str(last_instant.offset(i, meta.get("interval", "year")))] = (
            value * increase
        )
        value *= increase
    return Parameter("self", data=data)
