"""The record of ``set_input`` values matches a reference model.

Random sequences of ``set_input``, ``calculate``, ``delete_arrays`` (through
the simulation and through a holder), ``clone``, ``get_branch`` and
``_invalidate_all_caches`` run on a family of simulations. A reference model
tracks, for each simulation, which (variable, branch, period) values were
stored through ``set_input`` and not deleted since. After every step, for
every simulation:

- ``_user_input_keys`` is exactly the model's set, so a step on one
  simulation never changes another's record;
- each entry names a value the simulation stores, equal to the input;
- ``to_input_dataframe`` exports the periods the model says are inputs.

One input variable is set through a custom ``set_input`` handler that
calculates another variable first and stores months under string periods.

After ``_invalidate_all_caches`` the simulation and its branches store their
inputs and nothing else. ``test_user_input_keys.py`` pins the same behaviour
with examples.
"""

from __future__ import annotations

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Household
from policyengine_core.model_api import MONTH, Variable
from policyengine_core.simulations import SimulationBuilder

SITUATION = {
    "persons": {
        "a": {"birth": {"ETERNITY": "1960-05-01"}, "salary": {"2025-01": 3_000}},
        "b": {"birth": {"ETERNITY": "1990-01-01"}},
    },
    "households": {"h": {"parents": ["a", "b"], "rent": {"2025-01": 800}}},
}
MONTHS = ["2025-01", "2025-02", "2025-03"]


def _calculate_then_set_months(holder, period, array):
    holder.simulation.calculate("income_tax", MONTHS[0])
    for month in MONTHS:
        holder._set(month, array)


class quarterly_rent(Variable):
    """Set for a year through a custom handler that calculates income tax
    first, then stores the first quarter's months under string periods."""

    value_type = float
    entity = Household
    definition_period = MONTH
    label = "Rent for the first quarter"
    set_input = _calculate_then_set_months


TAX_BENEFIT_SYSTEM = CountryTaxBenefitSystem()
TAX_BENEFIT_SYSTEM.add_variable(quarterly_rent)
INPUTS = ["salary", "income_tax", "rent", "quarterly_rent", "birth"]
EXPORTABLE = ["salary", "rent", "quarterly_rent", "birth"]
CALCULATED = [
    "salary",
    "income_tax",
    "social_security_contribution",
    "disposable_income",
    "rent",
    "housing_allowance",
    "birth",
]
MAX_SIMULATIONS = 6


def _canonical(variable, period):
    if variable == "birth":
        return periods.period(periods.ETERNITY)
    return periods.period(period)


def _slot(key):
    variable, branch, period = key
    return variable, f"{branch}:{_canonical(variable, period)}"


def _input_value(variable, number):
    if variable == "birth":
        return [f"19{50 + number % 50}-01-01"] * 2
    if variable in ("rent", "quarterly_rent"):
        return [float(number)]
    return [float(number), float(number) / 2]


class _Model:
    """What one simulation should record, and the inputs it should store."""

    def __init__(self, branch_name, parent, inputs):
        self.branch_name = branch_name
        self.parent = parent
        self.inputs = dict(inputs)
        self.branches = {}

    def copy(self, branch_name, parent):
        return _Model(branch_name, parent, self.inputs)

    def visible_branch_names(self):
        names = [self.branch_name]
        parent = self.parent
        while parent is not None:
            names.append(parent.branch_name)
            parent = parent.parent
        names.append("default")
        return list(dict.fromkeys(names))

    def delete(self, variable, period, branch_name):
        if period is not None:
            period = periods.period(period)
        for key in list(self.inputs):
            if (
                key[0] == variable
                and key[1] == branch_name
                and (
                    period is None
                    or variable == "birth"
                    or period.contains(periods.period(key[2]))
                )
            ):
                del self.inputs[key]


class _Family:
    def __init__(self, tax_benefit_system):
        root = SimulationBuilder().build_from_entities(tax_benefit_system, SITUATION)
        inputs = {}
        for key in root._user_input_keys:
            variable, branch, period = key
            stored = root.get_holder(variable)._memory_storage._arrays
            inputs[key] = stored[_slot(key)[1]].copy()
        self.simulations = [root]
        self.models = [_Model("default", None, inputs)]

    def pick(self, index):
        index %= len(self.simulations)
        return self.simulations[index], self.models[index]

    def add(self, simulation, model):
        if simulation not in self.simulations:
            self.simulations.append(simulation)
            self.models.append(model)

    def apply(self, operation):
        kind, index, *arguments = operation
        simulation, model = self.pick(index)
        if kind == "set_input":
            variable, period, number = arguments
            value = _input_value(variable, number)
            simulation.set_input(variable, period, value)
            array = simulation.get_holder(variable)._to_array(np.asarray(value))
            # The handler stores the quarter's months. An eternal variable
            # stores one value whatever the period it is set for, and its
            # entry names that value.
            stored = MONTHS if variable == "quarterly_rent" else [period]
            for stored_period in stored:
                key = (variable, model.branch_name, _canonical(variable, stored_period))
                model.inputs[key] = array
        elif kind == "calculate":
            variable, period = arguments
            simulation.calculate(variable, period)
        elif kind == "delete":
            variable, period = arguments
            simulation.delete_arrays(variable, period)
            for branch_name in model.visible_branch_names():
                model.delete(variable, period, branch_name)
        elif kind == "holder_delete":
            variable, period, branch_choice = arguments
            names = model.visible_branch_names()
            branch_name = names[branch_choice % len(names)]
            simulation.get_holder(variable).delete_arrays(period, branch_name)
            model.delete(variable, period, branch_name)
        elif kind == "clone":
            if len(self.simulations) < MAX_SIMULATIONS:
                self.add(
                    simulation.clone(), model.copy(model.branch_name, model.parent)
                )
        elif kind == "branch":
            (name,) = arguments
            if name == model.branch_name or name in model.branches:
                return
            if len(self.simulations) < MAX_SIMULATIONS:
                branch_model = model.copy(name, model)
                model.branches[name] = branch_model
                self.add(simulation.get_branch(name), branch_model)
        elif kind == "invalidate":
            simulation._invalidate_all_caches()
            self._assert_only_inputs_stored(simulation, model)

    def _assert_only_inputs_stored(self, simulation, model):
        stored = {
            (variable, storage_key)
            for population in simulation.populations.values()
            for variable, holder in population._holders.items()
            for storage_key in holder._memory_storage._arrays
        }
        assert stored == {_slot(key) for key in model.inputs}
        for name, branch in simulation.branches.items():
            self._assert_only_inputs_stored(branch, model.branches[name])

    def assert_records_match(self):
        for simulation, model in zip(self.simulations, self.models):
            assert set(simulation._user_input_keys) == set(model.inputs)
            for key, expected in model.inputs.items():
                variable, storage_key = _slot(key)
                holder = simulation.get_holder(variable)
                stored = holder._memory_storage._arrays.get(storage_key)
                assert stored is not None, key
                assert np.array_equal(stored, expected), key
            visible = model.visible_branch_names()
            for variable in EXPORTABLE:
                exported = simulation._get_exportable_input_periods(variable, False)
                expected_periods = {
                    periods.period(period)
                    for name, branch, period in model.inputs
                    if name == variable and branch in visible
                }
                if variable == "birth":
                    assert bool(exported) == bool(expected_periods)
                else:
                    assert set(exported) == expected_periods


_months_or_all = st.sampled_from(MONTHS + ["2025", None])
_index = st.integers(min_value=0, max_value=MAX_SIMULATIONS - 1)
_operation = st.one_of(
    st.tuples(
        st.just("set_input"),
        _index,
        st.sampled_from(["salary", "income_tax", "rent"]),
        st.sampled_from(MONTHS),
        st.integers(min_value=0, max_value=5_000),
    ),
    st.tuples(
        st.just("set_input"),
        _index,
        st.just("birth"),
        st.sampled_from(["ETERNITY", "2025", "2025-02"]),
        st.integers(min_value=0, max_value=49),
    ),
    st.tuples(
        st.just("set_input"),
        _index,
        st.just("quarterly_rent"),
        st.just("2025"),
        st.integers(min_value=0, max_value=5_000),
    ),
    st.tuples(
        st.just("calculate"),
        _index,
        st.sampled_from(CALCULATED),
        st.sampled_from(MONTHS),
    ),
    st.tuples(
        st.just("delete"),
        _index,
        st.sampled_from(INPUTS),
        _months_or_all,
    ),
    st.tuples(
        st.just("holder_delete"),
        _index,
        st.sampled_from(INPUTS),
        _months_or_all,
        st.integers(min_value=0, max_value=3),
    ),
    st.tuples(st.just("clone"), _index),
    st.tuples(st.just("branch"), _index, st.sampled_from(["b1", "b2"])),
    st.tuples(st.just("invalidate"), _index),
)


@hypothesis.settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(operations=st.lists(_operation, max_size=30))
def test_user_input_keys_match_reference_model(operations):
    family = _Family(TAX_BENEFIT_SYSTEM)
    family.assert_records_match()
    for step, operation in enumerate(operations):
        family.apply(operation)
        try:
            family.assert_records_match()
        except AssertionError as error:
            raise AssertionError((step, operation)) from error
