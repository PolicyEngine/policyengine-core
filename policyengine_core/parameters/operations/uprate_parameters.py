from numpy import ceil, floor

from policyengine_core.parameters.operations.get_parameter import get_parameter
from policyengine_core.parameters.parameter import Parameter
from policyengine_core.parameters.parameter_at_instant import (
    ParameterAtInstant,
)
from policyengine_core.parameters.parameter_node import ParameterNode
from policyengine_core.parameters.parameter_scale import ParameterScale
from policyengine_core.periods import instant, period


def uprate_parameters(root: ParameterNode) -> ParameterNode:
    """Uprates parameters according to their metadata.

    Args:
        root (ParameterNode): The root of the parameter tree.

    Returns:
        ParameterNode: The same root, with uprating applied to descendants.
    """

    descendants = list(root.get_descendants())

    scales = list(filter(lambda p: isinstance(p, ParameterScale), descendants))
    for scale in scales:
        for bracket in scale.brackets:
            for allowed_key in bracket._allowed_keys:
                if hasattr(bracket, allowed_key):
                    descendants.append(getattr(bracket, allowed_key))

    for parameter in descendants:
        if isinstance(parameter, Parameter):
            if parameter.metadata.get("uprating") is not None:
                
                # Pull the uprating definition dict
                meta = parameter.metadata["uprating"]

                # If defined in short method (i.e. "uprating: PARAM"),
                # redefine this as dict with param key
                if meta == "self":
                    meta = dict(parameter="self")
                elif isinstance(meta, str):
                    meta = dict(parameter=meta)

                # If uprating with a set candence, ensure that all 
                # required values are present, otherwise ensure they're all None
                cadence_options_test = [
                    meta.get("application_date"),
                    meta.get("interval_start"),
                    meta.get("interval_measurement")
                ]

                if (any(cadence_options_test) != all(cadence_options_test)):
                    raise SyntaxError(
                        f"Failed to uprate {parameter.name}; application_date, interval_start, and interval_measurement must all be provided"
                    )
                
                # If param is "self", construct the uprating table
                if meta["parameter"] == "self":

                    uprating_parameter = construct_uprater_self(
                        parameter,
                        meta,
                    )
                # Otherwise, pull uprating table from YAML
                else:
                    uprating_parameter = get_parameter(root, meta["parameter"])

                # Start from the latest value
                if "start_instant" in meta:
                    last_instant = instant(meta["start_instant"])
                else:
                    last_instant = instant(
                        parameter.values_list[0].instant_str
                    )

                # For each defined instant in the uprating parameter
                for entry in uprating_parameter.values_list[::-1]:
                    entry_instant = instant(entry.instant_str)
                    # If the uprater instant is defined after the last parameter instant
                    if entry_instant > last_instant:
                        # Apply the uprater and add to the parameter
                        value_at_start = parameter(last_instant)
                        uprater_at_start = uprating_parameter(last_instant)
                        if uprater_at_start is None:
                            raise ValueError(
                                f"Failed to uprate using {uprating_parameter.name} at {last_instant} for {parameter.name} at {entry_instant} because the uprating parameter is not defined at {last_instant}."
                            )
                        uprater_at_entry = uprating_parameter(entry_instant)
                        uprater_change = uprater_at_entry / uprater_at_start
                        uprated_value = value_at_start * uprater_change
                        if "rounding" in meta:
                            uprated_value = round_uprated_value(meta, uprated_value)
                        parameter.values_list.append(
                            ParameterAtInstant(
                                parameter.name,
                                entry.instant_str,
                                data=uprated_value,
                            )
                        )
                parameter.values_list.sort(
                    key=lambda x: x.instant_str, reverse=True
                )
    return root

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
    uprated_value = (
        rounding_fn(uprated_value / interval)
        * interval
    )
    return uprated_value

def construct_uprater_self(parameter: Parameter, meta: dict) -> Parameter:
    last_instant = instant(
        parameter.values_list[0].instant_str
    )
    start_instant = instant(
        meta.get(
            "from",
            last_instant.offset(
                -1, meta.get("interval", "year")
            ),
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
        data[
            str(
                last_instant.offset(
                    i, meta.get("interval", "year")
                )
            )
        ] = (value * increase)
        value *= increase
    return Parameter("self", data=data)