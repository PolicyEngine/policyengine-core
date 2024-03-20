from numpy import ceil, floor
from dateutil import parser, rrule
from datetime import datetime, timedelta

from policyengine_core.parameters.operations.get_parameter import get_parameter
from policyengine_core.parameters.parameter import Parameter
from policyengine_core.parameters.parameter_at_instant import (
    ParameterAtInstant,
)
from policyengine_core.parameters.parameter_node import ParameterNode
from policyengine_core.parameters.parameter_scale import ParameterScale
from policyengine_core.periods import instant, Instant


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

                # If param is "self", construct the uprating table
                if meta["parameter"] == "self":
                    uprating_parameter = construct_uprater_self(
                        parameter,
                        meta,
                    )
                # Otherwise, pull uprating table from YAML
                else:
                    uprating_parameter = get_parameter(root, meta["parameter"])

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
                    cadence_options = construct_cadence_options(
                        cadence_meta, parameter
                    )

                    # Determine the first date from which to start uprating -
                    # this should be the first application date (month, day)
                    # following the last defined param value (not including the
                    # final value)
                    uprating_first_date = find_cadence_first(
                        parameter, cadence_options
                    )
                    uprating_last_date = find_cadence_last(
                        uprating_parameter, cadence_options
                    )

                    print("\nNew entry")
                    print(uprating_first_date)
                    print(uprating_last_date)

                    # Use existing uprating parameter to construct cadence-based uprater, where
                    # each resulting uprating option is a multiplication factor to be applied to
                    # the preceding value
                    cadence_uprater = construct_cadence_uprater(
                        parameter,
                        uprating_parameter,
                        cadence_options,
                        uprating_first_date,
                        uprating_last_date,
                    )

                    uprated_data = uprate_by_cadence(
                        parameter, cadence_uprater, uprating_first_date, meta
                    )
                    parameter.values_list.extend(uprated_data)

                else:
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
                            uprater_at_entry = uprating_parameter(
                                entry_instant
                            )
                            uprater_change = (
                                uprater_at_entry / uprater_at_start
                            )
                            uprated_value = value_at_start * uprater_change
                            if "rounding" in meta:
                                uprated_value = round_uprated_value(
                                    meta, uprated_value
                                )
                            parameter.values_list.append(
                                ParameterAtInstant(
                                    parameter.name,
                                    entry.instant_str,
                                    data=uprated_value,
                                )
                            )
                # Whether using cadence or not, sort the parameter values_list
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
    uprated_value = rounding_fn(uprated_value / interval) * interval
    return uprated_value


def uprate_by_cadence(
    parameter: Parameter,
    cadence_uprater: Parameter,
    uprating_first_date: Instant,
    meta: dict,
) -> list[ParameterAtInstant]:

    # Pull out the value that occurred most recently in the parameter
    # at the enactment month, date; this will be changed in future to support
    # different periods than yearly
    reference_value = parameter.get_at_instant(
        uprating_first_date
    )

    uprated_data = []
    for uprater_entry in reversed(cadence_uprater.values_list):

        # Calculate uprated value
        uprated_value = uprater_entry.value * reference_value
        if "rounding" in meta:
            uprated_value = round_uprated_value(meta, uprated_value)

        # Add uprated value to data list
        uprated_data.append(
            ParameterAtInstant(
                parameter.name, uprater_entry.instant_str, data=uprated_value
            )
        )

        # Swap reference_value with new value
        reference_value = uprated_value
    return uprated_data


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
        return parser.parse(cadence_options["effective"])
    
    # Save the "interval", if provided, else set to "year"
    interval = None
    if cadence_options.get("interval") is not None:
        interval = cadence_options["interval"]
    else:
        interval = "year"

    # Pull of the first (newest) value and parse into datetime object
    newest_param: datetime = parser.parse(param_values[0].instant_str)
    
    # Create an offset of one day; if the newest param date is the same as our
    # enactment date, (e.g., newest param is 2022-04-01 and enactment is 04-01),
    # we don't want to include 2022-04-01, but rrule inclusively applies the
    # dtstart arg, so we need to offset by one day when evaluating
    offset: timedelta = timedelta(days=1)
    offset_param = newest_param + offset

    print(offset_param)

    # Conditionally determine settings to utilize
    # Set the month only if the interval is "year"
    month_option = cadence_options["enactment"]["month"] if interval == "year" else None
    # Set the day if interval is "year" or "month", just not "day"
    day_option = cadence_options["enactment"]["day"] if interval != "day" else None

    rrule_array = rrule.rrule(
        freq=rrule.YEARLY, dtstart=offset_param, bymonth=month_option, bymonthday=day_option, count=1
    )

    return(rrule_array[0])

#     # Parse first_date and last_date into datetime objects
#     start_date = parser.parse(str(first_date))
#     end_date = parser.parse(str(last_date))
#     # Determine the frequency module to utilize within rrule
#     interval = ""
#     rrule_interval = ""
#     if cadence_options.get("interval"):
#         interval = cadence_options["interval"]
#         rrule_interval = getattr(rrule, ((interval + "LY").upper()))
#     else:
#         interval = "year"
#         rrule_interval = rrule.YEARLY
# 
#     # Generate a list of iterations
#     iterations = rrule.rrule(
#         freq=rrule_interval, dtstart=start_date, until=end_date
#     )
# 
# 
    # # Pull off the first (newest) value and turn into Instant
    # newest_param: Instant = instant(param_values[0].instant_str)

    # # If month, day of most recent param occurs before enactment date, then
    # # first uprated date exists in same year, otherwise it must be in next year;
    # # in this case, add 1
    # cadence_start_year = None
    # if (
    #     # e.g., If newest_param is 2-1 and enactment is 4-1
    #     newest_param.month < cadence_options["enactment"]["month"]
    #     or (
    #         # e.g., If newest_param is 4-1 and enactment is 4-15
    #         newest_param.month == cadence_options["enactment"]["month"]
    #         and newest_param.day < cadence_options["enactment"]["day"]
    #     )
    # ):
    #     cadence_start_year = newest_param.year
    # else:
    #     cadence_start_year = newest_param.year + 1

    # # Must pass date as tuple if not a string
    # return instant(
    #     (
    #         cadence_start_year,
    #         cadence_options["enactment"]["month"],
    #         cadence_options["enactment"]["day"],
    #     )
    # )


def find_cadence_last(uprater: Parameter, cadence_options: dict) -> Instant:
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

    # Pull off the first (newest) value and turn into Instant
    last_param: Instant = instant(uprater_values[0].instant_str)

    # If month, day of most recent uprater value occurs after or on uprater end
    # date, then last uprater year is last year, otherwise it's current year; in
    # former case, subtract 1
    cadence_end_year = None
    if (
        # e.g., If last_param is 10-1 and uprater end date is 4-1
        last_param.month > cadence_options["end"]["month"]
        or (
            # e.g., If last_param is 4-15 and uprater end date is 4-1
            last_param.month == cadence_options["end"]["month"]
            and last_param.day > cadence_options["end"]["day"]
        )
    ):
        cadence_end_year = last_param.year - 1
    else:
        cadence_end_year = last_param.year

    # We're seeking the last uprating enactment date, not measurement date;
    # thus, we now increment the year by the difference between the uprater measurement
    # end year and the enactment year
    cadence_end_year += (
        cadence_options["enactment"]["year"] - cadence_options["end"]["year"]
    )

    # Must pass date as tuple if not a string
    return instant(
        (
            cadence_end_year,
            cadence_options["enactment"]["month"],
            cadence_options["enactment"]["day"],
        )
    )


def construct_cadence_uprater(
    parameter: Parameter,
    uprating_parameter: Parameter,
    cadence_options: dict,
    first_date: Instant,
    last_date: Instant,
) -> Parameter:

    # Parse first_date and last_date into datetime objects
    start_date = parser.parse(str(first_date))
    end_date = parser.parse(str(last_date))

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
    iterations = rrule.rrule(
        freq=rrule_interval, dtstart=start_date, until=end_date
    )
    # and count entries in list
    iteration_count = rrule.rrule.count(iterations)

    # Determine the offset between the first enactment
    # date and the first start and end date
    edited_uprater_data = {}
    start_year_offset = (
        cadence_options["enactment"]["year"] - cadence_options["start"]["year"]
    )
    end_year_offset = (
        cadence_options["enactment"]["year"] - cadence_options["end"]["year"]
    )

    # Convert these to instants to be offset in the loop below
    period_start = instant(
        (
            first_date.year - start_year_offset,
            cadence_options["start"]["month"],
            cadence_options["start"]["day"],
        )
    )

    period_end = instant(
        (
            first_date.year - end_year_offset,
            cadence_options["end"]["month"],
            cadence_options["end"]["day"],
        )
    )

    # Within the iteration list...
    for i in range(iteration_count):

        start_calc_date = period_start.offset(i, interval)
        end_calc_date = period_end.offset(i, interval)

        # Find uprater value at cadence start
        start_value = uprating_parameter(start_calc_date)

        # Find uprater value at cadence end
        end_value = uprating_parameter(end_calc_date)

        # Ensure that earliest date exists within uprater
        if not start_value:
            raise ValueError(
                f"Failed to uprate {parameter.name} using {uprating_parameter.name}: uprater missing values at date {str(start_calc_date)}"
            )

        # Find difference of these values
        difference = end_value / start_value

        # Apply that at enactment date for this iteration
        edited_uprater_data[str(first_date.offset(i, interval))] = difference

    return Parameter(uprating_parameter.name, edited_uprater_data)


def construct_cadence_options(
    cadence_settings: dict, parameter: Parameter
) -> dict:

    # Define all settings with fixed input options
    # so as to test that input is valid
    FIXED_OPTIONS = {"interval": ["year", "month", "day"]}

    # All of these will be split into a dict of (year: int, month: int, day: int)
    MATHEMATICAL_KEYS = ["start", "end", "enactment"]

    # All of these will be converted to Instant
    INSTANT_KEYS = ["effective"]

    # All of these will remain strings
    STRING_KEYS = ["interval"]

    cadence_options = {}
    for key in MATHEMATICAL_KEYS:

        if cadence_settings.get(key) is None:
            continue

        date_split = cadence_settings[key].split("-")
        if len(date_split) != 3:
            raise SyntaxError(
                f"Error while uprating {parameter.name}: cadence date values must be YYYY-MM-DD"
            )
        cadence_options[key] = {
            "year": int(date_split[0]),
            "month": int(date_split[1]),
            "day": int(date_split[2]),
        }

    for key in INSTANT_KEYS:

        if cadence_settings.get(key) is None:
            continue

        cadence_options[key] = instant(cadence_settings[key])

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
