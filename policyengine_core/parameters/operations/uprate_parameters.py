from numpy import ceil, floor

# rrule is purposely imported this way to allow for programmatic
# calling of rrule.YEARLY, rrule.MONTHLY, and rrule.DAILY
from dateutil import rrule
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from datetime import datetime

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
                    uprating_first_date: datetime = find_cadence_first(
                        parameter, cadence_options
                    )
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


def find_cadence_first(
    parameter: Parameter, cadence_options: dict
) -> datetime:
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
    month_option = (
        cadence_options["enactment"].month if interval == "year" else None
    )
    # Set the day if interval is "year" or "month", just not "day"
    day_option = (
        cadence_options["enactment"].day if interval != "day" else None
    )

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
    start_date: datetime = (
        last_param - relativedelta(years=1) + relativedelta(days=1)
    )

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
    iterations = rrule.rrule(
        freq=rrule_interval, dtstart=first_date, until=last_date
    )

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
        start_val = uprating_parameter.get_at_instant(
            instant(start_calc_date.date())
        )

        # Find uprater value at cadence end
        end_val = uprating_parameter.get_at_instant(
            instant(end_calc_date.date())
        )

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


def construct_cadence_options(
    cadence_settings: dict, parameter: Parameter
) -> dict:
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
