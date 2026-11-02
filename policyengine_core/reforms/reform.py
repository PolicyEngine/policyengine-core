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

# ---------------------------------------------------------------------------
# Reform period-key handling.
#
# A reform dict maps a parameter path to ``{period_key: value}``. Each
# ``period_key`` is classified by format and translated into keyword arguments
# for ``Parameter.update``. These are small pure functions so each format is
# independently testable.
# ---------------------------------------------------------------------------

_SCALAR_REFORM_PERIOD = "year:2000:100"
"""Window applied for the ``{path: scalar}`` shorthand (a value with no period)."""


def _eternity_update_kwargs(value) -> dict:
    """``"ETERNITY"`` -> the value applies for all time."""
    return {"value": value}


def _range_update_kwargs(period_key: str, value) -> dict:
    """``"start.stop"`` -> the bounded interval ``[start, stop]``."""
    start, stop = period_key.split(".")
    return {"start": instant_(start), "stop": instant_(stop), "value": value}


def _bounded_period_update_kwargs(period_key: str, value) -> dict:
    """Compound period (``"year:2026:5"``, ``"month:2026-01:3"``) -> that
    bounded period only."""
    return {"period": period_(period_key), "value": value}


def _from_instant_update_kwargs(period_key: str, value) -> dict:
    """Bare ISO instant (``"2026"`` / ``"2026-01"`` / ``"2026-01-01"``) -> the
    value applies from that instant onward (``stop`` omitted)."""
    return {"start": instant_(period_key), "value": value}


def _period_key_update_kwargs(period_key, value) -> dict:
    """Translate one ``period_key`` into ``Parameter.update`` kwargs.

    Non-string keys (e.g. an int year) are coerced to ``str`` so they behave
    like their string form rather than raising.
    """
    period_key = str(period_key)
    if period_key == "ETERNITY":
        return _eternity_update_kwargs(value)
    if "." in period_key:
        return _range_update_kwargs(period_key, value)
    if ":" in period_key:
        return _bounded_period_update_kwargs(period_key, value)
    return _from_instant_update_kwargs(period_key, value)


def _update_start_instant(update_kwargs: dict):
    """Instant an update begins at, used to order application so a
    later-starting key can't clobber an earlier one via ``stop=None``.
    Value-only (``ETERNITY``) updates sort first, as the all-time base."""
    if "start" in update_kwargs:
        return update_kwargs["start"]
    if "period" in update_kwargs:
        return update_kwargs["period"].start
    return instant_("0001-01-01")


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
    """The country id of the reform, if one was given."""

    parameter_values: dict = None
    """The parameter values a reform created by :meth:`Reform.from_dict` applies."""

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
                (or the ``path -> scalar`` shorthand, applied across
                ``year:2000:100``).

        Period-key formats, interpreted per parameter (see the module-level
        ``_*_update_kwargs`` helpers):
            * Bare ISO instant (``"2026"`` / ``"2026-01"`` / ``"2026-01-01"``):
              the value applies **from that instant onward**, flattening any
              later scheduled breakpoints (e.g. uprating). The instant is
              literal: ``{"2026-06-01": v}`` applies from 1 June 2026, so an
              annual value read at 1 January 2026 is still the baseline — use
              ``"2026-01-01"`` for a full-year-onward change. To keep later
              breakpoints, bound the change with a range instead.
            * Range ``"start.stop"`` (e.g. ``"2026-01-01.2027-12-31"``): the
              value applies over the bounded interval ``[start, stop]``; the
              prior value is restored afterwards.
            * Compound bounded period (``"year:2026:5"`` / ``"month:2026-01:3"``):
              the value applies over that bounded period only.
            * ``"ETERNITY"``: the value applies for all time.

        Entries are applied in ascending start-instant order, so a
        later-starting key never clobbers an earlier one. Entries sharing a
        start instant are applied in their given order (the last wins for any
        overlap). Non-string keys are coerced to ``str``.

        Invalid input raises during construction **before any parameter is
        modified**: a malformed period key or bad value yields no partial
        reform, and errors from ``Parameter.update`` propagate rather than
        being silently swallowed.

        Returns:
            A reform.
        """

        class reform(Reform):
            def apply(self):
                for path, period_values in parameter_values.items():
                    parameter = self.parameters.get_child(path)
                    if not isinstance(period_values, dict):
                        # Scalar shorthand: apply across the default window.
                        parameter.update(
                            period=_SCALAR_REFORM_PERIOD, value=period_values
                        )
                        continue
                    # Translate every entry to update kwargs FIRST, so a
                    # malformed key raises before any parameter is mutated (no
                    # partial reform). Then apply in ascending start-instant
                    # order; the stable sort keeps same-start entries in their
                    # given order (last wins on overlap).
                    updates = [
                        _period_key_update_kwargs(period_key, value)
                        for period_key, value in period_values.items()
                    ]
                    updates.sort(key=_update_start_instant)
                    for update_kwargs in updates:
                        parameter.update(**update_kwargs)

        reform.country_id = country_id
        reform.parameter_values = parameter_values
        reform.name = name

        return reform


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
