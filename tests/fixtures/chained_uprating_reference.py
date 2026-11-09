"""Chained-mode uprating as it was before base mode, for differential tests.

``reference_uprate_parameter`` is ``uprate_parameter`` from
``policyengine_core/parameters/operations/uprate_parameters.py`` at
PolicyEngine/policyengine-core@7c84d941b, the last commit before base mode,
copied verbatim except for its name. Base mode added an early branch to that
function and left the helpers it calls unchanged, so the copy imports them.

``test_base_uprating_property.py`` checks that ``uprate_parameters`` gives
every chained-mode parameter exactly what this copy gives it. A change that
means to alter chained mode should update this copy with the reason.
"""

from datetime import datetime
from typing import Optional

from policyengine_core.parameters.parameter import Parameter
from policyengine_core.parameters.parameter_at_instant import ParameterAtInstant
from policyengine_core.parameters.parameter_node import ParameterNode
from policyengine_core.parameters.operations.uprate_parameters import (
    construct_cadence_options,
    construct_uprater_self,
    find_cadence_first,
    find_cadence_last,
    get_parameter_paths,
    get_uprating_parameter,
    normalize_uprating_metadata,
    round_uprated_value,
    sort_parameters_by_uprating_dependencies,
    uprate_by_cadence,
)
from policyengine_core.periods import instant


def reference_uprate_parameters(root: ParameterNode) -> ParameterNode:
    """``uprate_parameters`` at 7c84d941b, calling the copy below."""
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
        reference_uprate_parameter(parameter, root, parameter_paths)
    return root


def reference_uprate_parameter(
    parameter: Parameter,
    root: ParameterNode,
    parameter_paths: Optional[dict[int, str]] = None,
) -> None:
    if parameter_paths is None:
        parameter_paths = {}
    # Pull the uprating definition dict
    meta = normalize_uprating_metadata(parameter.metadata["uprating"])

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
