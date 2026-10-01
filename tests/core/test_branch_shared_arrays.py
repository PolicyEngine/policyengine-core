"""Branches share their parent's cached arrays instead of copying them.

``Simulation.get_branch`` used to copy every cached array of every holder
for every period (``InMemoryStorage.clone``), so each branch cost a full copy
of the simulation's cache. A branch now gets its own index of read-only views
of those arrays. These tests pin what must not change:

- isolation: nothing done through a branch changes a value that its parent,
  or any ancestor, reads;
- snapshot: a branch reads its parent's values as they were when the branch
  was created, whatever the parent stores, recalculates or deletes after;
- nested branches and ``set_input`` on branches behave as before;
- ``Simulation.clone`` still copies every array.

The differential test at the end runs random sequences of operations on a
simulation whose branches share arrays and on one whose branches are copies
(the previous ``get_branch``), and requires every read to agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from policyengine_core import periods
from policyengine_core.country_template import Microsimulation
from policyengine_core.enums import EnumArray
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.simulations.simulation import _cloning_branch
from policyengine_core.variables import Variable

SITUATION = {
    "persons": {
        "a": {"birth": {"ETERNITY": "1950-03-01"}, "salary": {"2017-01": 4000}},
        "b": {"birth": {"ETERNITY": "1984-01-01"}, "salary": {"2017-01": 2500}},
        "c": {"birth": {"ETERNITY": "2010-06-01"}},
        "d": {"birth": {"ETERNITY": "1940-01-01"}, "salary": {"2017-01": 100}},
    },
    "households": {
        "h1": {
            "parents": ["a", "b"],
            "children": ["c"],
            "rent": {"2017-01": 1200},
            "accommodation_size": {"2017-01": 80},
            "housing_occupancy_status": {"2017-01": "tenant"},
        },
        "h2": {
            "parents": ["d"],
            "rent": {"2017-01": 300},
            "accommodation_size": {"2017-01": 40},
            "housing_occupancy_status": {"2017-01": "owner"},
        },
    },
}

JANUARY = periods.period("2017-01")


def _build(tax_benefit_system):
    return SimulationBuilder().build_from_entities(tax_benefit_system, SITUATION)


def _stored_arrays(simulation):
    """Every (variable, storage key) -> array in a simulation's memory storage."""
    return {
        (name, key): array
        for population in simulation.populations.values()
        for name, holder in population._holders.items()
        for key, array in holder._memory_storage._arrays.items()
    }


def _snapshot(simulation):
    return {
        key: np.array(array, copy=True)
        for key, array in _stored_arrays(simulation).items()
    }


def _assert_unchanged(simulation, snapshot):
    current = _stored_arrays(simulation)
    assert current.keys() == snapshot.keys()
    for key, array in current.items():
        assert np.array_equal(np.asarray(array), np.asarray(snapshot[key])), key


def _deep_copy_branch(simulation, name, clone_system=False):
    """``get_branch`` as it was before branches shared arrays."""
    if name == simulation.branch_name:
        return simulation
    if name in simulation.branches:
        return simulation.branches[name]
    branch = simulation.clone(clone_tax_benefit_system=clone_system)
    simulation.branches[name] = branch
    branch.branch_name = name
    branch.parent_branch = simulation
    if simulation.trace:
        branch.trace = True
        branch.tracer = simulation.tracer
    return branch


# ----- Sharing ----- #


def test_branch_shares_parent_arrays_read_only(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    simulation.calculate("housing_tax", "2017")
    parent_arrays = _stored_arrays(simulation)

    branch = simulation.get_branch("branch")
    branch_arrays = _stored_arrays(branch)

    assert branch_arrays.keys() == parent_arrays.keys()
    for key, array in branch_arrays.items():
        assert np.shares_memory(array, parent_arrays[key]), key
        assert not array.flags.writeable, key
        # The parent keeps writing to its own arrays as before.
        assert parent_arrays[key].flags.writeable, key


def test_branch_storage_index_is_its_own(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    branch = simulation.get_branch("branch")
    for population in simulation.populations.values():
        branch_population = branch.populations[population.entity.key]
        for name, holder in population._holders.items():
            branch_holder = branch_population._holders[name]
            assert branch_holder is not holder
            assert branch_holder._memory_storage is not holder._memory_storage
            assert (
                branch_holder._memory_storage._arrays
                is not holder._memory_storage._arrays
            )


def test_writing_in_place_through_a_branch_raises_and_leaves_parent(
    tax_benefit_system,
):
    simulation = _build(tax_benefit_system)
    salary = simulation.calculate("salary", JANUARY)
    expected = salary.copy()
    branch = simulation.get_branch("branch")
    shared = branch.calculate("salary", JANUARY)

    with pytest.raises(ValueError, match="read-only"):
        shared[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        shared += 1.0
    with pytest.raises(ValueError, match="read-only"):
        np.add(shared, 1.0, out=shared)

    assert np.array_equal(simulation.calculate("salary", JANUARY), expected)
    assert np.array_equal(branch.calculate("salary", JANUARY), expected)


def test_formula_writing_into_a_shared_array_says_which_formula(
    isolated_tax_benefit_system,
):
    class salary_plus_one(Variable):
        value_type = float
        entity = isolated_tax_benefit_system.person_entity
        definition_period = periods.MONTH
        label = "Salary plus one, written in place"

        def formula(person, period, parameters):
            salary = person("salary", period)
            salary += 1
            return salary

    class salary_floor(Variable):
        value_type = float
        entity = isolated_tax_benefit_system.person_entity
        definition_period = periods.MONTH
        label = "Salary with a floor, written in place through pandas"

        def formula(person, period, parameters):
            salary = pd.Series(person("salary", period), copy=False)
            salary[salary < 1_000] = 1_000
            return salary.values

    isolated_tax_benefit_system.add_variables(salary_plus_one, salary_floor)
    simulation = _build(isolated_tax_benefit_system)
    salary = simulation.calculate("salary", JANUARY).copy()
    branch = simulation.get_branch("branch")

    for variable in ("salary_plus_one", "salary_floor"):
        with pytest.raises(ValueError) as raised:
            branch.calculate(variable, JANUARY)
        notes = "\n".join(getattr(raised.value, "__notes__", []))
        assert f"The formula of '{variable}' wrote in place" in notes
        assert "`x = x + y` rather than `x += y`" in notes

    assert np.array_equal(simulation.calculate("salary", JANUARY), salary)
    assert np.array_equal(branch.calculate("salary", JANUARY), salary)


def test_other_formula_value_errors_get_no_note(isolated_tax_benefit_system):
    class raises_value_error(Variable):
        value_type = float
        entity = isolated_tax_benefit_system.person_entity
        definition_period = periods.MONTH
        label = "Raises an unrelated ValueError"

        def formula(person, period, parameters):
            raise ValueError("not about arrays")

    isolated_tax_benefit_system.add_variables(raises_value_error)
    branch = _build(isolated_tax_benefit_system).get_branch("branch")

    with pytest.raises(ValueError, match="not about arrays") as raised:
        branch.calculate("raises_value_error", JANUARY)
    assert not getattr(raised.value, "__notes__", [])


def test_branch_own_results_are_writeable(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    branch = simulation.get_branch("branch")
    # Calculated in the branch, so stored by the branch, not shared.
    assert branch.calculate("income_tax", JANUARY).flags.writeable


def test_shared_enum_arrays_keep_their_enum(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    status = simulation.calculate("housing_occupancy_status", JANUARY)
    branch = simulation.get_branch("branch")
    shared = branch.calculate("housing_occupancy_status", JANUARY)

    assert isinstance(shared, EnumArray)
    assert shared.possible_values is status.possible_values
    assert list(shared.decode_to_str()) == list(status.decode_to_str())
    assert np.shares_memory(shared, status)
    assert not shared.flags.writeable


def test_shared_string_arrays(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.set_input("postal_code", JANUARY, np.array(["75001", "69002"]))
    postal_code = simulation.calculate("postal_code", JANUARY)
    branch = simulation.get_branch("branch")
    shared = branch.calculate("postal_code", JANUARY)
    assert shared.dtype == postal_code.dtype
    assert np.array_equal(shared, postal_code)
    assert np.shares_memory(shared, postal_code)
    assert not shared.flags.writeable


def test_branch_of_a_traced_simulation_shares_arrays(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.trace = True
    salary = simulation.calculate("salary", JANUARY)

    branch = simulation.get_branch("branch")

    assert branch.trace
    assert branch.tracer is simulation.tracer
    assert np.shares_memory(branch.calculate("salary", JANUARY), salary)
    branch.set_input("salary", JANUARY, np.zeros(4))
    assert branch.calculate("income_tax", JANUARY).sum() == 0
    assert simulation.calculate("income_tax", JANUARY).sum() > 0


def test_baseline_branch_of_a_reform_simulation_shares_its_inputs():
    reform_rate, baseline_rate, instant = 0.42, 0.15, "2022-01-01"
    simulation = Microsimulation(
        reform={"taxes.income_tax_rate": {instant: reform_rate}}
    )
    baseline = simulation.baseline
    assert baseline is simulation.branches["baseline"]

    shared = _stored_arrays(baseline)
    parent_arrays = _stored_arrays(simulation)
    assert shared
    for key, array in shared.items():
        assert np.shares_memory(array, parent_arrays[key]), key
        assert not array.flags.writeable, key

    # The reform simulation calculates after the baseline branch exists; the
    # branch must not pick up the reform's results.
    reform_tax = simulation.calculate("income_tax", "2022-01").sum()
    baseline_tax = baseline.calculate("income_tax", "2022-01").sum()
    assert reform_tax > 0
    assert baseline_tax == pytest.approx(
        reform_tax * baseline_rate / reform_rate, rel=1e-6
    )


def test_branch_reads_values_its_parent_stored_on_disk(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = simulation.get_holder("rent")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    february = periods.period("2017-02")
    simulation.set_input("rent", february, np.array([11.0, 22.0]))
    assert holder._memory_storage.get(february, "default") is None

    branch = simulation.get_branch("branch")
    assert np.array_equal(branch.calculate("rent", february), [11.0, 22.0])
    # In-memory values of the same holder are still shared.
    assert np.shares_memory(
        branch.calculate("rent", JANUARY), simulation.calculate("rent", JANUARY)
    )

    branch.delete_arrays("rent", february)
    assert branch.get_array("rent", february) is None
    assert np.array_equal(simulation.calculate("rent", february), [11.0, 22.0])


# ----- Isolation: a branch never changes its parent ----- #


def test_set_input_on_branch_leaves_parent(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    parent_tax = simulation.calculate("income_tax", JANUARY).copy()
    snapshot = _snapshot(simulation)

    branch = simulation.get_branch("branch")
    branch.set_input("salary", JANUARY, np.array([10_000.0, 0.0, 0.0, 0.0]))
    branch_tax = branch.calculate("income_tax", JANUARY)

    # ``income_tax`` was cached before branching, so the branch keeps the
    # parent's value, exactly as when the branch held a copy of it.
    assert np.array_equal(branch_tax, parent_tax)
    branch.delete_arrays("income_tax", JANUARY)
    assert branch.calculate("income_tax", JANUARY)[0] > parent_tax[0]

    _assert_unchanged(simulation, snapshot)
    assert np.array_equal(simulation.calculate("income_tax", JANUARY), parent_tax)


def test_branch_calculations_stay_in_branch(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.calculate("salary", JANUARY)
    snapshot = _snapshot(simulation)

    branch = simulation.get_branch("branch")
    branch.calculate("disposable_income", JANUARY)
    branch.calculate("housing_tax", "2017")
    branch.calculate("salary", "2017")  # Sums months into a new yearly entry.

    _assert_unchanged(simulation, snapshot)
    assert simulation.get_array("disposable_income", JANUARY) is None


def test_delete_arrays_on_branch_leaves_parent(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    snapshot = _snapshot(simulation)

    branch = simulation.get_branch("branch")
    branch.delete_arrays("salary")
    branch.delete_arrays("disposable_income", JANUARY)

    assert branch.get_array("salary", JANUARY) is None
    _assert_unchanged(simulation, snapshot)


def test_dropping_a_branch_and_clearing_its_storage_leaves_parent(
    tax_benefit_system,
):
    # The pattern policyengine-uk's Marriage Allowance election uses to free
    # a short-lived branch.
    simulation = _build(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    snapshot = _snapshot(simulation)

    branch = simulation.get_branch("short_lived")
    branch.set_input("salary", JANUARY, np.zeros(4))
    branch.calculate("disposable_income", JANUARY)
    del simulation.branches["short_lived"]
    for population in branch.populations.values():
        for holder in population._holders.values():
            holder._memory_storage._arrays.clear()

    _assert_unchanged(simulation, snapshot)
    again = simulation.get_branch("short_lived")
    assert np.array_equal(
        again.calculate("salary", JANUARY), simulation.calculate("salary", JANUARY)
    )


def test_apply_reform_on_branch_leaves_parent(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    snapshot = _snapshot(simulation)

    branch = simulation.get_branch("branch", clone_system=True)
    branch._invalidate_all_caches()
    branch.calculate("disposable_income", JANUARY)

    _assert_unchanged(simulation, snapshot)


# ----- Snapshot: the parent's later changes stay out of the branch ----- #


def test_parent_set_input_after_branching_is_not_seen(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    original = simulation.calculate("salary", JANUARY).copy()
    branch = simulation.get_branch("branch")

    simulation.set_input("salary", JANUARY, np.array([1.0, 2.0, 3.0, 4.0]))

    assert np.array_equal(branch.calculate("salary", JANUARY), original)


def test_parent_calculation_after_branching_is_not_seen(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    branch = simulation.get_branch("branch")
    branch.set_input("salary", JANUARY, np.array([9_000.0, 0.0, 0.0, 0.0]))

    parent_tax = simulation.calculate("income_tax", JANUARY)

    assert branch.get_array("income_tax", JANUARY) is None
    branch_tax = branch.calculate("income_tax", JANUARY)
    assert not np.array_equal(branch_tax, parent_tax)


def test_parent_delete_after_branching_is_not_seen(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    tax = simulation.calculate("income_tax", JANUARY).copy()
    branch = simulation.get_branch("branch")

    simulation.delete_arrays("income_tax")
    simulation.set_input("salary", JANUARY, np.zeros(4))
    simulation.calculate("income_tax", JANUARY)

    assert np.array_equal(branch.get_array("income_tax", JANUARY), tax)


def test_parent_writing_in_place_after_branching_reaches_the_branch(
    tax_benefit_system,
):
    """The one way a parent's later change reaches a branch.

    The branch reads the parent's array through a view, so a write INTO that
    array, rather than a new array stored with ``set_input``, shows in the
    branch too. Core never writes into a stored array (see the test that
    freezes the parent's arrays below); this pins what happens when calling
    code does.
    """
    simulation = _build(tax_benefit_system)
    salary = simulation.calculate("salary", JANUARY)
    branch = simulation.get_branch("branch")

    salary[0] = 123.0

    assert simulation.calculate("salary", JANUARY)[0] == 123.0
    assert branch.calculate("salary", JANUARY)[0] == 123.0

    # A deep copy made before the write is unaffected.
    clone = simulation.clone()
    salary[0] = 456.0
    assert clone.calculate("salary", JANUARY)[0] == 123.0


def test_parent_apply_reform_after_branching_keeps_branch_inputs(
    tax_benefit_system,
):
    simulation = _build(tax_benefit_system)
    salary = simulation.calculate("salary", JANUARY).copy()
    simulation.calculate("disposable_income", JANUARY)
    branch = simulation.get_branch("branch")

    # ``apply_reform`` wipes formula caches in the simulation and its
    # branches, and keeps user inputs.
    simulation._invalidate_all_caches()

    assert branch.get_array("disposable_income", JANUARY) is None
    assert np.array_equal(branch.calculate("salary", JANUARY), salary)
    assert np.array_equal(
        branch.calculate("disposable_income", JANUARY),
        simulation.calculate("disposable_income", JANUARY),
    )


# ----- Nested branches ----- #


def test_nested_branch_reads_its_ancestors(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.calculate("income_tax", JANUARY)
    root_snapshot = _snapshot(simulation)

    child = simulation.get_branch("itemizing")
    child.set_input("rent", JANUARY, np.array([2_000.0, 50.0]))
    child.calculate("housing_allowance", JANUARY)
    child_snapshot = _snapshot(child)

    grandchild = child.get_branch("no_salt")
    assert np.array_equal(grandchild.calculate("rent", JANUARY), [2_000.0, 50.0])
    assert np.array_equal(
        grandchild.calculate("income_tax", JANUARY),
        simulation.calculate("income_tax", JANUARY),
    )
    child_arrays = _stored_arrays(child)
    grandchild_arrays = _stored_arrays(grandchild)
    assert grandchild_arrays.keys() == child_arrays.keys()
    for key, array in grandchild_arrays.items():
        assert not array.flags.writeable, key
        assert np.shares_memory(array, child_arrays[key]), key
        assert np.array_equal(np.asarray(array), child_snapshot[key]), key

    grandchild.set_input("rent", JANUARY, np.array([1.0, 1.0]))
    grandchild.set_input("salary", JANUARY, np.zeros(4))
    grandchild.delete_arrays("income_tax")
    grandchild.calculate("disposable_income", JANUARY)

    _assert_unchanged(simulation, root_snapshot)
    _assert_unchanged(child, child_snapshot)


def test_nested_branch_snapshot_of_its_parent(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    child = simulation.get_branch("child")
    child.set_input("rent", JANUARY, np.array([700.0, 70.0]))
    grandchild = child.get_branch("grandchild")

    child.set_input("rent", JANUARY, np.array([1.0, 1.0]))
    simulation.set_input("rent", JANUARY, np.array([2.0, 2.0]))

    assert np.array_equal(grandchild.calculate("rent", JANUARY), [700.0, 70.0])


def test_nested_branch_shares_with_every_ancestor(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    salary = simulation.calculate("salary", JANUARY)
    child = simulation.get_branch("child")
    rent = child.calculate("rent", JANUARY)  # Shared from the root.
    child.set_input("accommodation_size", JANUARY, np.array([10.0, 20.0]))
    size = child.get_array("accommodation_size", JANUARY)
    grandchild = child.get_branch("grandchild")

    assert np.shares_memory(grandchild.calculate("salary", JANUARY), salary)
    assert np.shares_memory(grandchild.calculate("rent", JANUARY), rent)
    assert np.shares_memory(grandchild.calculate("accommodation_size", JANUARY), size)


# ----- set_input on branches ----- #


def test_yearly_set_input_on_branch_divides_into_branch_months(
    tax_benefit_system,
):
    simulation = _build(tax_benefit_system)
    snapshot = _snapshot(simulation)
    branch = simulation.get_branch("branch")

    # ``salary`` divides a yearly input across the months of the year,
    # skipping months this branch already stores.
    branch.set_input("salary", "2018", np.full(4, 12_000.0))

    for month in periods.period("2018").get_subperiods(periods.MONTH):
        assert np.array_equal(branch.calculate("salary", month), np.full(4, 1_000.0))
        assert simulation.get_array("salary", month) is None
    _assert_unchanged(simulation, snapshot)


def test_set_input_on_nested_branch(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    child = simulation.get_branch("child")
    grandchild = child.get_branch("grandchild")
    root_snapshot = _snapshot(simulation)
    child_snapshot = _snapshot(child)

    grandchild.set_input("salary", "2017-02", np.array([5.0, 6.0, 7.0, 8.0]))
    grandchild.set_input("rent", "2017-02", np.array([3.0, 4.0]))

    assert np.array_equal(
        grandchild.calculate("salary", "2017-02"), [5.0, 6.0, 7.0, 8.0]
    )
    assert child.get_array("salary", "2017-02") is None
    _assert_unchanged(simulation, root_snapshot)
    _assert_unchanged(child, child_snapshot)


def test_set_input_records_the_branch_input(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    branch = simulation.get_branch("branch")
    branch.set_input("rent", "2017-03", np.array([1.0, 2.0]))
    assert ("rent", "branch", periods.period("2017-03")) in branch._user_input_keys


# ----- Core never writes into a shared array ----- #


def test_branch_calculations_never_write_into_parent_arrays(tax_benefit_system):
    """Freeze every parent array, then exercise a branch end to end."""
    simulation = _build(tax_benefit_system)
    variables = [
        "salary",
        "age",
        "basic_income",
        "income_tax",
        "social_security_contribution",
        "pension",
        "disposable_income",
        "housing_allowance",
        "parenting_allowance",
        "household_income",
        "total_benefits",
        "total_taxes",
    ]
    for variable in variables:
        simulation.calculate(variable, JANUARY)
    simulation.calculate("housing_tax", "2017")
    for array in _stored_arrays(simulation).values():
        array.flags.writeable = False

    branch = simulation.get_branch("branch")
    child = branch.get_branch("child")
    for sim in (branch, child):
        sim.set_input("salary", "2018", np.full(4, 24_000.0))
        sim.set_input("rent", "2017-02", np.array([100.0, 200.0]))
        for variable in variables:
            for month in ("2017-01", "2017-02", "2018-06"):
                sim.calculate(variable, month)
            sim.calculate_add(variable, "2017")
        sim.calculate("housing_tax", "2018")
        sim.delete_arrays("income_tax", JANUARY)
        sim.calculate("income_tax", JANUARY)


# ----- clone() still copies ----- #


def test_clone_copies_arrays(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    parent_arrays = _stored_arrays(simulation)

    clone = simulation.clone()

    for key, array in _stored_arrays(clone).items():
        assert not np.shares_memory(array, parent_arrays[key]), key
        assert array.flags.writeable, key


def test_clone_of_a_branch_copies_arrays(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    branch = simulation.get_branch("branch")
    branch_arrays = _stored_arrays(branch)

    clone = branch.clone()

    for key, array in _stored_arrays(clone).items():
        assert not np.shares_memory(array, branch_arrays[key]), key
        assert array.flags.writeable, key


def test_get_branch_shares_through_a_country_clone_override(tax_benefit_system):
    """A ``clone`` override with the existing signature still shares arrays.

    policyengine-us's SPM ``Simulation`` overrides ``clone(self, debug,
    trace, clone_tax_benefit_system)`` and calls ``super().clone``.
    """

    class CountrySimulation(Simulation):
        def clone(self, debug=False, trace=False, clone_tax_benefit_system=True):
            cloned = super().clone(
                debug=debug,
                trace=trace,
                clone_tax_benefit_system=clone_tax_benefit_system,
            )
            cloned.cloned_by_country = True
            return cloned

    simulation = _build(tax_benefit_system)
    simulation.__class__ = CountrySimulation
    salary = simulation.calculate("salary", JANUARY)

    branch = simulation.get_branch("branch")
    assert branch.cloned_by_country
    assert np.shares_memory(branch.calculate("salary", JANUARY), salary)

    clone = simulation.clone()
    assert clone.cloned_by_country
    assert not np.shares_memory(clone.calculate("salary", JANUARY), salary)


def test_sharing_ends_when_get_branch_returns(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    salary = simulation.calculate("salary", JANUARY)
    simulation.get_branch("branch")
    assert _cloning_branch.get() is False
    clone = simulation.clone()
    assert not np.shares_memory(clone.calculate("salary", JANUARY), salary)


def test_sharing_ends_when_clone_raises(tax_benefit_system):
    class FailingSimulation(Simulation):
        def clone(self, debug=False, trace=False, clone_tax_benefit_system=True):
            raise RuntimeError("clone failed")

    simulation = _build(tax_benefit_system)
    simulation.__class__ = FailingSimulation
    with pytest.raises(RuntimeError, match="clone failed"):
        simulation.get_branch("branch")
    assert _cloning_branch.get() is False
    assert "branch" not in simulation.branches


# ----- Differential: shared-array branches agree with copied branches ----- #

PERSON_INPUTS = ["salary"]
HOUSEHOLD_INPUTS = ["rent", "accommodation_size"]
READ_VARIABLES = [
    ("salary", "month"),
    ("salary", "year"),
    ("age", "month"),
    ("basic_income", "month"),
    ("income_tax", "month"),
    ("social_security_contribution", "month"),
    ("pension", "month"),
    ("disposable_income", "month"),
    ("rent", "month"),
    ("accommodation_size", "month"),
    ("housing_occupancy_status", "month"),
    ("housing_allowance", "month"),
    ("parenting_allowance", "month"),
    ("household_income", "month"),
    ("housing_tax", "year"),
]
MONTHS = ["2017-01", "2017-02", "2018-01"]
YEARS = ["2017", "2018"]
BRANCH_NAMES = ["a", "b", "c"]

_operation = st.one_of(
    st.tuples(
        st.just("set_input"),
        st.integers(0, 7),
        st.sampled_from(PERSON_INPUTS + HOUSEHOLD_INPUTS),
        st.sampled_from(MONTHS + YEARS),
        st.integers(0, 5_000),
    ),
    st.tuples(
        st.just("read"),
        st.integers(0, 7),
        st.sampled_from(READ_VARIABLES),
        st.integers(0, 1),
        st.booleans(),
    ),
    st.tuples(
        st.just("delete"),
        st.integers(0, 7),
        st.sampled_from([name for name, _ in READ_VARIABLES]),
        st.sampled_from(MONTHS + YEARS + [None]),
    ),
    st.tuples(
        st.just("branch"),
        st.integers(0, 7),
        st.sampled_from(BRANCH_NAMES),
        st.booleans(),
    ),
    st.tuples(st.just("drop"), st.integers(0, 7)),
    st.tuples(st.just("invalidate"), st.integers(0, 7)),
)


class _Tree:
    """A root simulation and the branches made from it, in creation order."""

    def __init__(self, tax_benefit_system, make_branch):
        self.nodes = [_build(tax_benefit_system)]
        self.make_branch = make_branch

    def node(self, index):
        return self.nodes[index % len(self.nodes)]


def _result(function):
    try:
        value = function()
    except Exception as error:  # Compare failures as well as values.
        return ("error", type(error).__name__)
    if value is None:
        return ("none",)
    if isinstance(value, EnumArray):
        return ("enum", value.possible_values, np.asarray(value.view(np.ndarray)))
    return ("array", np.asarray(value))


def _same(left, right):
    if left[0] != right[0]:
        return False
    if left[0] == "array":
        return left[1].dtype == right[1].dtype and np.array_equal(left[1], right[1])
    if left[0] == "enum":
        return left[1] is right[1] and np.array_equal(left[2], right[2])
    return left == right


def _apply(tree, operation):
    kind, index = operation[0], operation[1]
    simulation = tree.node(index)
    if kind == "set_input":
        _, _, variable, period, seed = operation
        count = (
            simulation.persons.count
            if variable in PERSON_INPUTS
            else simulation.household.count
        )
        values = (np.arange(count, dtype=float) + 1) * seed
        return _result(lambda: simulation.set_input(variable, period, values))
    if kind == "read":
        _, _, (variable, unit), period_index, get_only = operation
        period = (MONTHS if unit == "month" else YEARS)[period_index]
        if get_only:
            return _result(lambda: simulation.get_array(variable, period))
        return _result(lambda: simulation.calculate(variable, period))
    if kind == "delete":
        _, _, variable, period = operation
        return _result(lambda: simulation.delete_arrays(variable, period))
    if kind == "branch":
        _, _, name, clone_system = operation
        branch = tree.make_branch(simulation, name, clone_system)
        if branch is not simulation and all(branch is not n for n in tree.nodes):
            tree.nodes.append(branch)
        return ("branch", tree.nodes.index(branch))
    if kind == "drop":
        parent = simulation.parent_branch
        if parent is None:
            return ("root",)
        del parent.branches[simulation.branch_name]
        for population in simulation.populations.values():
            for holder in population._holders.values():
                holder._memory_storage._arrays.clear()
        tree.nodes.remove(simulation)
        return ("dropped",)
    if kind == "invalidate":
        simulation._invalidate_all_caches()
        return ("invalidated",)
    raise AssertionError(kind)


def _assert_shared_arrays_read_only(tree):
    """Every array a branch shares with its parent is read-only."""
    for simulation in tree.nodes[1:]:
        parent_arrays = list(_stored_arrays(simulation.parent_branch).values())
        for key, array in _stored_arrays(simulation).items():
            if any(np.shares_memory(array, other) for other in parent_arrays):
                assert not array.flags.writeable, key


@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(operations=st.lists(_operation, max_size=40))
def test_shared_array_branches_match_copied_branches(tax_benefit_system, operations):
    shared = _Tree(
        tax_benefit_system,
        lambda simulation, name, clone_system: simulation.get_branch(
            name, clone_system
        ),
    )
    copied = _Tree(tax_benefit_system, _deep_copy_branch)

    for step, operation in enumerate(operations):
        assert _same(_apply(shared, operation), _apply(copied, operation)), (
            step,
            operation,
        )
        assert len(shared.nodes) == len(copied.nodes)
        _assert_shared_arrays_read_only(shared)

    # Every simulation in both trees reads the same value for everything.
    for shared_simulation, copied_simulation in zip(shared.nodes, copied.nodes):
        for variable, unit in READ_VARIABLES:
            for period in MONTHS if unit == "month" else YEARS:
                assert _same(
                    _result(lambda: shared_simulation.calculate(variable, period)),
                    _result(lambda: copied_simulation.calculate(variable, period)),
                ), (shared_simulation.branch_name, variable, period)
