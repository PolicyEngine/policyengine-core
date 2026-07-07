from __future__ import annotations

import copy
from typing import Callable, Union, TYPE_CHECKING

from policyengine_core.parameters import ParameterNode, Parameter
from policyengine_core.taxbenefitsystems import TaxBenefitSystem

if TYPE_CHECKING:
    from policyengine_core.simulations import Simulation
from policyengine_core.periods import (
    period as period_,
    instant as instant_,
    Period,
)

import requests


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class Reform(TaxBenefitSystem):
    """A modified TaxBenefitSystem

    All reforms must subclass `Reform` and implement a method `apply()`.

    In this method, the reform can add or replace variables and call `modify_parameters` to modify the parameters of the legislation.

        Example:

        >>> from policyengine_core import reforms
        >>> from policyengine_core.parameters import load_parameter_file
        >>>
        >>> def modify_my_parameters(parameters):
        >>>     # Add new parameters
        >>>     new_parameters = load_parameter_file(name='reform_name', file_path='path_to_yaml_file.yaml')
        >>>     parameters.add_child('reform_name', new_parameters)
        >>>
        >>>     # Update a value
        >>>     parameters.taxes.some_tax.some_param.update(period=some_period, value=1000.0)
        >>>
        >>>    return parameters
        >>>
        >>> class MyReform(reforms.Reform):
        >>>    def apply(self):
        >>>        self.add_variable(some_variable)
        >>>        self.update_variable(some_other_variable)
        >>>        self.modify_parameters(modifier_function = modify_my_parameters)
    """

    name: str = None
    """The name of the reform. This is used to identify the reform in the UI."""

    country_id: str = None
    """The country id of the reform. This is used to inform any calls to the PolicyEngine API."""

    parameter_values: dict = None
    """The parameter values of the reform. This is used to inform any calls to the PolicyEngine API."""

    simulation: "Simulation" = None

    def __init__(self, baseline: TaxBenefitSystem):
        """
        :param baseline: Baseline TaxBenefitSystem.
        """
        super().__init__(baseline.entities)
        self.baseline = baseline
        # Clone the parameter tree so in-place mutations inside ``apply()``
        # (``set_parameter(...)`` or ``self.parameters.path.update(...)``)
        # don't leak into the baseline. ``modify_parameters()`` was already
        # deep-copying, but plenty of reforms mutate ``self.parameters``
        # directly (bug C4).
        if baseline.parameters is not None:
            self.parameters = baseline.parameters.clone()
        else:
            self.parameters = None
        # Use a fresh at-instant cache so reform mutations aren't shadowed
        # by previously-computed baseline ``ParameterNodeAtInstant`` objects.
        self._parameters_at_instant_cache = {}
        self.variables = baseline.variables.copy()
        self.decomposition_file_path = baseline.decomposition_file_path
        self.key = self.__class__.__name__
        if not hasattr(self, "apply"):
            raise Exception(
                "Reform {} must define an `apply` function".format(self.key)
            )
        self.apply()

    def __getattr__(self, attribute):
        return getattr(self.baseline, attribute)

    @property
    def full_key(self) -> str:
        key = self.key
        assert key is not None, "key was not set for reform {} (name: {!r})".format(
            self, self.name
        )
        if self.baseline is not None and hasattr(self.baseline, "key"):
            baseline_full_key = self.baseline.full_key
            key = ".".join([baseline_full_key, key])
        return key

    def modify_parameters(
        self, modifier_function: Callable[[ParameterNode], ParameterNode]
    ) -> None:
        """Make modifications on the parameters of the legislation.

        Call this function in `apply()` if the reform asks for legislation parameter modifications.

        Args:
            modifier_function: A function that takes a :obj:`.ParameterNode` and should return an object of the same type.
        """
        baseline_parameters = self.baseline.parameters
        baseline_parameters_copy = copy.deepcopy(baseline_parameters)
        reform_parameters = modifier_function(baseline_parameters_copy)
        if not isinstance(reform_parameters, ParameterNode):
            return ValueError(
                "modifier_function {} in module {} must return a ParameterNode".format(
                    modifier_function.__name__,
                    modifier_function.__module__,
                )
            )
        self.parameters = reform_parameters
        self._parameters_at_instant_cache = {}

    @staticmethod
    def from_dict(
        parameter_values: dict,
        country_id: str = None,
        name: str = None,
    ) -> Reform:
        """Create a reform from a dictionary of parameters.

        Args:
            parameter_values: A mapping of ``path -> {period_key: value}``
                (or the ``path -> scalar`` shorthand).

        Period-key formats, interpreted per parameter:
            * Bare ISO instant (``"2026"`` / ``"2026-01"`` / ``"2026-01-01"``):
              the value applies **from that instant onward**, flattening any
              later scheduled breakpoints (e.g. uprating) until the next
              reform key for the same parameter. This is the natural reading
              of a dated policy change and mirrors how a parameter's own
              value history behaves. To bound a change, use a range instead.
            * Range ``"start.stop"`` (e.g. ``"2026-01-01.2027-12-31"``): the
              value applies over the bounded interval ``[start, stop]``; the
              prior value is restored afterwards.
            * Compound bounded period (``"year:2026:5"`` / ``"month:2026-01:3"``):
              the value applies over that bounded period only.
            * ``"ETERNITY"``: the value applies for all time.

        Errors raised by ``Parameter.update`` (bad value, unknown parameter,
        malformed key) propagate instead of being silently swallowed.

        Returns:
            A reform.
        """

        def _start_instant(period_key: str):
            """Start instant of a period key, used only to order application.

            ``update(start=..., stop=None)`` removes every breakpoint at or
            after ``start``, so a later-starting key must be applied after an
            earlier-starting one or it would clobber it. Sorting entries by
            this key makes multi-entry dicts order-independent.
            """
            if period_key == "ETERNITY":
                return instant_("0001-01-01")
            if "." in period_key:
                return instant_(period_key.split(".")[0])
            if ":" in period_key:
                return period_(period_key).start
            return instant_(period_key)

        class reform(Reform):
            def apply(self):
                for path, period_values in parameter_values.items():
                    parameter = self.parameters.get_child(path)
                    if not isinstance(period_values, dict):
                        # Scalar shorthand: apply across the default window.
                        parameter.update(
                            period="year:2000:100", value=period_values
                        )
                        continue
                    for period_key, value in sorted(
                        period_values.items(),
                        key=lambda item: _start_instant(item[0]),
                    ):
                        if period_key == "ETERNITY":
                            parameter.update(value=value)
                        elif "." in period_key:
                            start, stop = period_key.split(".")
                            parameter.update(
                                start=instant_(start),
                                stop=instant_(stop),
                                value=value,
                            )
                        elif ":" in period_key:
                            parameter.update(
                                period=period_(period_key), value=value
                            )
                        else:
                            # Bare ISO instant: apply from this instant onward.
                            parameter.update(
                                start=instant_(period_key), value=value
                            )

        reform.country_id = country_id
        reform.parameter_values = parameter_values
        reform.name = name

        return reform

    @staticmethod
    def from_api(
        api_id: str,
        country_id: str = None,
    ) -> Reform:
        """Create a reform from a dictionary of parameters.

        Args:
            parameters: A dictionary of parameter -> { period -> value } pairs.

        Returns:
            A reform.
        """

        data = requests.get(
            f"https://api.policyengine.org/{country_id}/policy/{api_id}",
            # Timeout so a stalled API can't hang the caller forever
            # (bug M11).
            timeout=30,
        ).json()

        parameter_values = data.get("result", {}).get("policy_json", {})

        for path in parameter_values:
            keys_to_remove = []
            for start_stop_str in list(parameter_values[path].keys()):
                start, stop = start_stop_str.split(".")
                # Clamp to the supported window, then emit an explicit
                # ``start.stop`` range so from_dict always treats API policies
                # as bounded. A single-civil-year intersection would otherwise
                # stringify to a bare "YYYY" which, under the from-onward rule,
                # would extend the change past the policy's end.
                clamped = period_("year:2000:100").intersection(
                    instant_(start), instant_(stop)
                )
                time_period = f"{clamped.start}.{clamped.stop}"
                parameter_values[path][time_period] = parameter_values[path][
                    start_stop_str
                ]
                keys_to_remove.append(start_stop_str)
            for key in keys_to_remove:
                del parameter_values[path][key]

        return Reform.from_dict(
            parameter_values,
            country_id,
            data.get("result", {}).get("label", None),
        )

    @classproperty
    def api_id(self):
        if self.country_id is None:
            raise ValueError(
                "`country_id` is not set. This is required to use the API."
            )
        if self.parameter_values is None:
            raise ValueError(
                "`parameter_values` is not set. This is required to use the API."
            )

        sanitised_parameter_values = {}

        for path, period_values in self.parameter_values.items():
            sanitised_period_values = {}
            for period, value in period_values.items():
                period = period_(period)
                sanitised_period_values[f"{period.start}.{period.stop}"] = value
            sanitised_parameter_values[path] = sanitised_period_values

        response = requests.post(
            f"https://api.policyengine.org/{self.country_id}/policy",
            json={
                "data": sanitised_parameter_values,
                "name": self.name,
            },
            # Timeout so a stalled API can't hang the caller forever
            # (bug M11).
            timeout=30,
        )

        return response.json().get("result", {}).get("policy_id")


def set_parameter(
    path: Union[Parameter, str],
    value: float,
    period: str = None,
    start: str = None,
    stop: str = None,
    return_modifier=False,
) -> Reform:
    if stop is not None:
        stop = instant_(stop)

    if start is not None:
        start = instant_(start)

    if isinstance(path, Parameter):
        path = path.name

    def modifier(parameters: ParameterNode):
        node = parameters
        for name in path.split("."):
            try:
                if "[" not in name:
                    node = node.children[name]
                else:
                    try:
                        name, index = name.split("[")
                        index = int(index[:-1])
                        node = node.children[name].brackets[index]
                    except:
                        raise ValueError(
                            "Invalid bracket syntax (should be e.g. tax.brackets[3].rate"
                        )
            except:
                raise ValueError(
                    f"Could not find the parameter {path} (failed at {name})."
                )
        node.update(period=period, value=value, start=start, stop=stop)
        return parameters

    if return_modifier:
        return modifier

    class reform(Reform):
        def apply(self):
            self.modify_parameters(modifier)

    return reform
