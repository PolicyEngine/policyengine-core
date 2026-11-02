"""The record of ``set_input`` values matches a reference model.

Random sequences of ``set_input``, ``calculate``, ``delete_arrays`` (through
the simulation and through a holder), ``clone``, ``get_branch`` and
``_invalidate_all_caches`` run on a family of simulations. A reference model,
seeded from the situation the simulations are built from, tracks for each
simulation which (variable, branch, period) values were stored through
``set_input`` and not deleted since. After every step, for every simulation:

- ``_user_input_keys`` is exactly the model's set, so a step on one
  simulation never changes another's record;
- each entry names a value the simulation stores, equal to the input;
- ``to_input_dataframe`` exports exactly the model's inputs, with their
  values, and stores nothing.

One input variable is set through a custom ``set_input`` handler that
calculates another variable first and stores months under string periods.
Another is set for twelve months, which storage keys as a year.

After ``_invalidate_all_caches`` the simulation and its branches store their
inputs and nothing else.

A second property runs ``set_input``, ``calculate``, deletes and
``_invalidate_all_caches`` on one simulation whose holders store values in
memory or on disk, switching between the two. Its model tracks input values
separately in each tier and checks the record, reads and exported values;
a calculation surviving in one tier cannot stand in for a deleted input in
the other. ``test_user_input_keys.py`` pins storage behavior with examples.
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
from policyengine_core.experimental import MemoryConfig
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
# The inputs SITUATION sets, as the simulation should store them.
SITUATION_INPUTS = {
    ("salary", "default", "2025-01"): np.array([3_000.0, 0.0]),
    ("birth", "default", "ETERNITY"): np.array(
        ["1960-05-01", "1990-01-01"], dtype="datetime64[D]"
    ),
    ("rent", "default", "2025-01"): np.array([800.0]),
}
# Storage keys a value by its period's string form: twelve months starting on
# the first of a month are the year starting then.
TWELVE_MONTHS = {"month:2025-01:12": "2025", "month:2025-03:12": "year:2025-03"}


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
    return periods.period(TWELVE_MONTHS.get(period, period))


def _slot(key):
    variable, branch, period = key
    return variable, f"{branch}:{_canonical(variable, period)}"


def _input_value(variable, number):
    if variable == "birth":
        return [f"19{50 + number % 50}-01-01"] * 2
    if variable in ("rent", "quarterly_rent"):
        return [float(number)]
    return [float(number), float(number) / 2]


def _expected_array(variable, value):
    if variable == "birth":
        return np.array(value, dtype="datetime64[D]")
    return np.array(value, dtype=float)


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
        inputs = {
            (variable, branch, _canonical(variable, period)): value
            for (variable, branch, period), value in SITUATION_INPUTS.items()
        }
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
            array = _expected_array(variable, value)
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
            self._assert_export_matches(simulation, model)

    def _assert_export_matches(self, simulation, model):
        """``to_input_dataframe`` exports the inputs of the branches the
        simulation reads (for each period, the first such branch's), for
        periods of the variable's own unit, and stores nothing."""
        visible = model.visible_branch_names()
        expected = {}
        for branch in reversed(visible):
            for (variable, input_branch, period), value in model.inputs.items():
                definition_period = TAX_BENEFIT_SYSTEM.get_variable(
                    variable
                ).definition_period
                if (
                    input_branch == branch
                    and variable in EXPORTABLE
                    and period.unit == definition_period
                ):
                    if variable in ("rent", "quarterly_rent"):
                        # One household of both people.
                        value = np.repeat(value, 2)
                    expected[f"{variable}__{period}"] = value
        stored_before = _stored_keys(simulation)
        fast_cache = dict(simulation._fast_cache)

        exported = simulation.to_input_dataframe()

        # Exporting reads stored inputs only. Restore the reads it cached,
        # so checking does not change what later steps calculate.
        simulation._fast_cache = fast_cache
        assert _stored_keys(simulation) == stored_before
        assert set(exported.columns) == set(expected)
        for column, value in expected.items():
            assert np.array_equal(
                exported[column].to_numpy().astype(value.dtype), value
            ), column


def _stored_keys(simulation):
    return {
        (variable, key)
        for population in simulation.populations.values()
        for variable, holder in population._holders.items()
        for key in holder._memory_storage._arrays
    }


_months_or_all = st.sampled_from(MONTHS + ["2025", "year:2025-03", None])
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
        st.just("salary"),
        st.sampled_from(sorted(TWELVE_MONTHS)),
        st.integers(min_value=0, max_value=5_000),
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
    st.tuples(st.just("branch"), _index, st.sampled_from(["b1", "no_salt"])),
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


@hypothesis.settings(max_examples=300, deadline=None)
@hypothesis.example(
    unit=periods.MONTH, year=2025, month=1, day=1, size=12, eternal=False
)
@hypothesis.example(
    unit=periods.MONTH, year=2025, month=3, day=1, size=12, eternal=False
)
@hypothesis.given(
    unit=st.sampled_from([periods.DAY, periods.MONTH, periods.YEAR]),
    year=st.integers(min_value=1, max_value=20_000),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28),
    size=st.integers(min_value=1, max_value=36),
    eternal=st.booleans(),
)
def test_period_normalization_matches_storage_strings(
    unit, year, month, day, size, eternal
):
    """Normalization preserves storage keys and agrees with the old parser
    where it accepts the year, while accepting early and late years too."""
    from policyengine_core.data_storage import InMemoryStorage
    from policyengine_core.holders import Holder

    holder = object.__new__(Holder)
    holder._memory_storage = InMemoryStorage(is_eternal=eternal)
    period = periods.Period((unit, periods.instant((year, month, day)), size))

    canonical = holder._storage_period(period)

    assert holder._storage_period(canonical) == canonical
    if eternal:
        assert canonical == periods.period(periods.ETERNITY)
    else:
        assert str(canonical) == str(period)
        if 1_000 <= year <= 9_999:
            assert canonical == periods.period(str(period))


DISK_INPUTS = ["salary", "income_tax", "rent", "birth"]


class _Storages:
    """One simulation whose holders store in memory or on disk, and the
    input values independently tracked in each storage tier.

    A surviving calculation for the same key cannot replace a deleted
    tier's input in the model. The model retains both copies when both
    were supplied through ``set_input``, with memory first for reads.
    """

    def __init__(self, tax_benefit_system):
        self.simulation = SimulationBuilder().build_from_entities(
            tax_benefit_system, SITUATION
        )
        self.simulation.memory_config = MemoryConfig(max_memory_occupation=0)
        for variable in tax_benefit_system.variables:
            holder = self.simulation.get_holder(variable)
            holder._disk_storage = holder.create_disk_storage()
            holder._on_disk_storable = True
        self.memory_inputs = {
            (variable, branch, _canonical(variable, period)): value.copy()
            for (variable, branch, period), value in SITUATION_INPUTS.items()
        }
        self.disk_inputs = {}
        self.on_disk = True

    def holds(self, key, storage):
        variable, branch, period = key
        holder = self.simulation.get_holder(variable)
        tier = holder._memory_storage if storage == "memory" else holder._disk_storage
        return tier.has(period, branch)

    def _forget_deleted_inputs(self):
        self.memory_inputs = {
            key: value
            for key, value in self.memory_inputs.items()
            if self.holds(key, "memory")
        }
        self.disk_inputs = {
            key: value
            for key, value in self.disk_inputs.items()
            if self.holds(key, "disk")
        }

    def readable_inputs(self):
        return {**self.disk_inputs, **self.memory_inputs}

    def stored(self):
        return {
            (variable, branch, periods.period(period))
            for population in self.simulation.populations.values()
            for variable, holder in population._holders.items()
            for branch, period in [
                key.split(":", 1) for key in holder._memory_storage._arrays
            ]
            + [key.rsplit("_", 1) for key in holder._disk_storage._files]
        }

    def apply(self, operation):
        kind, *arguments = operation
        simulation = self.simulation
        if kind == "set_input":
            variable, period, number = arguments
            key = (variable, "default", _canonical(variable, period))
            holder = simulation.get_holder(variable)
            # Existing memory values are replaced in memory, regardless of
            # the threshold controlling subsequent writes to empty slots.
            writes_to_disk = self.on_disk and not holder._memory_storage.has(
                key[2], key[1]
            )
            value = _input_value(variable, number)
            simulation.set_input(variable, period, value)
            inputs = self.disk_inputs if writes_to_disk else self.memory_inputs
            inputs[key] = _expected_array(variable, value)
        elif kind == "calculate":
            simulation.calculate(*arguments)
        elif kind == "delete":
            simulation.delete_arrays(*arguments)
            self._forget_deleted_inputs()
        elif kind == "holder_delete":
            variable, period = arguments
            simulation.get_holder(variable).delete_arrays(period)
            self._forget_deleted_inputs()
        elif kind == "on_disk":
            (on_disk,) = arguments
            self.on_disk = on_disk
            simulation.memory_config.max_memory_occupation_pc = 0 if on_disk else 101
        elif kind == "invalidate":
            expected = self.readable_inputs()
            simulation._invalidate_all_caches()
            # Replay preserves memory before disk and keeps one copy. It
            # must discard every stored value that the model did not input.
            assert self.stored() == set(expected)
            self.disk_inputs = {
                key: value
                for key, value in self.disk_inputs.items()
                if key not in self.memory_inputs
            }

    def assert_record_matches(self):
        simulation = self.simulation
        inputs = self.readable_inputs()
        assert set(simulation._user_input_keys) == set(inputs)
        for tier, expected_inputs in (
            ("memory", self.memory_inputs),
            ("disk", self.disk_inputs),
        ):
            for (variable, branch, period), expected in expected_inputs.items():
                holder = simulation.get_holder(variable)
                storage = (
                    holder._memory_storage if tier == "memory" else holder._disk_storage
                )
                actual = storage.get(period, branch)
                assert actual is not None, (tier, variable, branch, period)
                assert np.array_equal(actual, expected), (
                    tier,
                    variable,
                    branch,
                    period,
                )
        for (variable, branch, period), expected in inputs.items():
            actual = simulation.get_holder(variable).get_array(period, branch)
            assert actual is not None, (variable, branch, period)
            assert np.array_equal(actual, expected), (variable, branch, period)
        self._assert_export_matches(inputs)

    def _assert_export_matches(self, inputs):
        simulation = self.simulation
        expected = {}
        for (variable, branch, period), value in inputs.items():
            definition_period = simulation.tax_benefit_system.get_variable(
                variable
            ).definition_period
            if (
                branch == "default"
                and variable in EXPORTABLE
                and period.unit == definition_period
            ):
                if variable in ("rent", "quarterly_rent"):
                    value = np.repeat(value, 2)
                expected[f"{variable}__{period}"] = value
        stored_before = self.stored()
        fast_cache = dict(simulation._fast_cache)

        exported = simulation.to_input_dataframe()

        # Preserve the generated sequence's cache state while checking
        # export's actual columns and values against the independent model.
        simulation._fast_cache = fast_cache
        assert self.stored() == stored_before
        assert set(exported.columns) == set(expected)
        for column, value in expected.items():
            assert np.array_equal(
                exported[column].to_numpy().astype(value.dtype), value
            ), column


_disk_operation = st.one_of(
    st.tuples(
        st.just("set_input"),
        st.sampled_from(["salary", "income_tax", "rent"]),
        st.sampled_from(MONTHS + sorted(TWELVE_MONTHS)),
        st.integers(min_value=0, max_value=5_000),
    ),
    st.tuples(
        st.just("set_input"),
        st.just("birth"),
        st.sampled_from(["ETERNITY", "2025", "2025-02"]),
        st.integers(min_value=0, max_value=49),
    ),
    st.tuples(
        st.just("calculate"), st.sampled_from(CALCULATED), st.sampled_from(MONTHS)
    ),
    st.tuples(st.just("delete"), st.sampled_from(DISK_INPUTS), _months_or_all),
    st.tuples(st.just("holder_delete"), st.sampled_from(DISK_INPUTS), _months_or_all),
    st.tuples(st.just("on_disk"), st.booleans()),
    st.tuples(st.just("invalidate")),
)


@hypothesis.settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.example(
    operations=[
        ("calculate", "rent", "2025-02"),
        ("on_disk", False),
        ("set_input", "rent", "2025-02", 500),
        ("delete", "rent", "2025"),
        ("invalidate",),
    ]
)
@hypothesis.given(operations=st.lists(_disk_operation, max_size=25))
def test_user_input_keys_follow_memory_and_disk_storage(operations):
    storages = _Storages(TAX_BENEFIT_SYSTEM)
    storages.assert_record_matches()
    for step, operation in enumerate(operations):
        storages.apply(operation)
        try:
            storages.assert_record_matches()
        except AssertionError as error:
            raise AssertionError((step, operation)) from error
