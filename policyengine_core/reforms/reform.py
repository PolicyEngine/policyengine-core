from __future__ import annotations

import copy
from typing import Callable, Union

from policyengine_core.parameters import ParameterNode, Parameter
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.periods import period as period_

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

    def __init__(self, baseline: TaxBenefitSystem):
        """
        :param baseline: Baseline TaxBenefitSystem.
        """
        super().__init__(baseline.entities)
        self.baseline = baseline
        self.parameters = baseline.parameters
        self._parameters_at_instant_cache = (
            baseline._parameters_at_instant_cache
        )
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
        assert (
            key is not None
        ), "key was not set for reform {} (name: {!r})".format(self, self.name)
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
            parameters: A dictionary of parameter -> { period -> value } pairs.

        Returns:
            A reform.
        """

        class reform(Reform):
            def apply(self):
                for path, period_values in parameter_values.items():
                    for period, value in period_values.items():
                        self.modify_parameters(
                            set_parameter(
                                path, value, period, return_modifier=True
                            )
                        )

        reform.country_id = country_id
        reform.parameter_values = parameter_values
        reform.name = name

        return reform

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
                sanitised_period_values[
                    f"{period.start}.{period.stop}"
                ] = value
            sanitised_parameter_values[path] = sanitised_period_values

        response = requests.post(
            f"https://api.policyengine.org/{self.country_id}/policy",
            json={
                "data": sanitised_parameter_values,
                "name": self.name,
            },
        )

        return response.json().get("result", {}).get("policy_id")


def set_parameter(
    path: Union[Parameter, str],
    value: float,
    period: str = "year:2015:10",
    return_modifier=False,
) -> Reform:
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
                    f"Could not find the parameter (failed at {name})."
                )
        node.update(period=period, value=value)
        return parameters

    if return_modifier:
        return modifier

    class reform(Reform):
        def apply(self):
            self.modify_parameters(modifier)

    return reform
