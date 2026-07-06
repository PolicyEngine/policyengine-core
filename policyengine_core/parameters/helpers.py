import os
import traceback
import warnings

import numpy

from policyengine_core import parameters, periods
from policyengine_core.errors import ParameterParsingError
from policyengine_core.parameters import config
from policyengine_core.warnings import ParameterKeyWarning


def contains_nan(vector):
    if numpy.issubdtype(vector.dtype, numpy.record) or numpy.issubdtype(
        vector.dtype, numpy.void
    ):
        return any([contains_nan(vector[name]) for name in vector.dtype.names])
    else:
        return numpy.isnan(vector).any()


def load_parameter_file(file_path, name=""):
    """
    Load parameters from a YAML file (or a directory containing YAML files).

    :returns: An instance of :class:`.ParameterNode` or :class:`.ParameterScale` or :class:`.Parameter`.
    """
    if not os.path.exists(file_path):
        raise ValueError("{} does not exist".format(file_path))
    if os.path.isdir(file_path):
        return parameters.ParameterNode(name, directory_path=file_path)
    data = _load_yaml_file(file_path)
    return _parse_child(name, data, file_path)


def _compose_name(path, child_name=None, item_name=None):
    if not path:
        return child_name
    if child_name is not None:
        return "{}.{}".format(path, child_name)
    if item_name is not None:
        return "{}[{}]".format(path, item_name)


def _load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return config.yaml.load(f, Loader=config.Loader)
        except (
            config.yaml.scanner.ScannerError,
            config.yaml.parser.ParserError,
        ):
            stack_trace = traceback.format_exc()
            raise ParameterParsingError(
                "Invalid YAML. Check the traceback above for more details.",
                file_path,
                stack_trace,
            )
        except Exception:
            stack_trace = traceback.format_exc()
            raise ParameterParsingError(
                "Invalid parameter file content. Check the traceback above for more details.",
                file_path,
                stack_trace,
            )


def _parse_child(child_name, child, child_path):
    if "values" in child:
        return parameters.Parameter(child_name, child, child_path)
    elif "brackets" in child:
        return parameters.ParameterScale(child_name, child, child_path)
    elif isinstance(child, dict) and all(
        [periods.INSTANT_PATTERN.match(str(key)) for key in child.keys()]
    ):
        return parameters.Parameter(child_name, child, child_path)
    else:
        return parameters.ParameterNode(child_name, data=child, file_path=child_path)


def _validate_parameter(parameter, data, data_type=None, allowed_keys=None):
    type_map = {
        dict: "object",
        list: "array",
    }

    if data_type is not None and not isinstance(data, data_type):
        raise ParameterParsingError(
            "'{}' must be of type {}.".format(parameter.name, type_map[data_type]),
            parameter.file_path,
        )

    if allowed_keys is not None and isinstance(data, dict):
        _warn_on_unknown_keys(parameter, data, allowed_keys)


def _warn_on_unknown_keys(parameter, data, allowed_keys):
    """Warn about keys placed alongside the recognized parameter keys.

    Unknown siblings are silently discarded by the loader. The most damaging
    case is an ``uprating:`` block written next to ``values:`` instead of under
    ``metadata:``, which freezes an indexed parameter at its last explicit value
    (see issue #505). Emit a ``ParameterKeyWarning`` naming the file, the
    offending key, and the fix, rather than dropping it silently.

    This is a warning (not an error) so country packages can sweep their trees
    before it becomes a hard error in a future major release.
    """
    unknown_keys = [
        key
        for key in data.keys()
        if key not in allowed_keys and key not in config.LEGACY_COMPAT_KEYS
    ]
    for key in unknown_keys:
        location = (
            "'{}'".format(parameter.file_path)
            if parameter.file_path is not None
            else "parameter '{}'".format(parameter.name)
        )
        hint = ""
        if key == "uprating":
            hint = " Move `uprating` under `metadata:`."
        warnings.warn(
            "Unknown key '{key}' in {location} (parameter '{name}'). "
            "It sits outside the recognized keys and will be ignored.{hint} "
            "Allowed keys are: {allowed}.".format(
                key=key,
                location=location,
                name=parameter.name,
                hint=hint,
                allowed=", ".join(sorted(str(k) for k in allowed_keys)),
            ),
            ParameterKeyWarning,
            stacklevel=3,
        )
