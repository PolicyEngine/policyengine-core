"""``set_input`` on a branch drops what may depend on the value it replaces.

A branch starts with every value its parent holds. ``set_input`` on a branch
used to store the new value and keep everything calculated from the old one,
so a value the parent calculated before branching answered for the branch,
whatever the branch's inputs said, and the branch's results depended on what
had been calculated before it was created.

Every stored value now carries a sequence number from one process-wide
counter, and a value is stored after everything it was calculated from. Each
simulation keeps a history of the first stores its values may have been
calculated from: a branch copies its parent's, and a formula that calculates
in another simulation takes in that simulation's history. An input set on a
branch drops the branch's non-input values stored at or after the earliest
recorded store, or derivation, of the overridden variable for an overlapping
period (``StoreHistory.earliest_dependency``), and the records the drop made
obsolete.

The invariant tested throughout (and, over random sequences of calculations,
branches and inputs, by ``test_branch_input_invalidation_property.py``):
whatever the branch's family calculated before, and in whatever order, a
calculation in a branch equals the same calculation in a new simulation given
the branch's inputs, its own and those it inherited when it was created,
before calculating anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.data_storage.store_history import (
    StoreHistory,
    next_sequence_number,
)
from policyengine_core.experimental import MemoryConfig
from policyengine_core.model_api import Reform
from policyengine_core.simulations import SimulationBuilder
import policyengine_core.simulations.simulation as simulation_module
from tests.fixtures.branch_input_invalidation import (
    CARRY_OVER_SYSTEM,
    FORMULA_RUNS,
    ROOT_INPUTS,
    branching_simulation,
    synthetic_simulation,
)

JANUARY = periods.period("2017-01")
JANUARY_2022 = periods.period("2022-01")

SITUATION = {
    "persons": {
        "a": {"birth": {"ETERNITY": "1980-01-01"}, "salary": {"2017-01": 4000}},
        "b": {"birth": {"ETERNITY": "1985-01-01"}, "salary": {"2017-01": 1000}},
    },
    "households": {
        "h": {
            "parents": ["a", "b"],
            "accommodation_size": {"2017-01": 80},
            "housing_occupancy_status": {"2017-01": "tenant"},
            "rent": {"2017-01": 900},
        }
    },
}


def _build(tax_benefit_system, situation=SITUATION):
    return SimulationBuilder().build_from_entities(tax_benefit_system, situation)


def _with_salary(tax_benefit_system, salary, period=JANUARY):
    """A new simulation given ``salary`` before calculating anything."""
    simulation = _build(tax_benefit_system)
    simulation.set_input("salary", period, np.asarray(salary, dtype=float))
    return simulation


def _stored_keys(simulation, variable):
    return set(simulation.get_holder(variable)._memory_storage._arrays)


# ----- Examples on the country template ----- #


def test_branch_input_reaches_a_value_the_parent_calculated_first(
    tax_benefit_system,
):
    simulation = _build(tax_benefit_system)
    simulation.calculate("income_tax", JANUARY)
    simulation.calculate("disposable_income", JANUARY)

    branch = simulation.get_branch("raise")
    branch.set_input("salary", JANUARY, np.array([5000.0, 1000.0]))

    fresh = _with_salary(tax_benefit_system, [5000.0, 1000.0])
    for variable in ("income_tax", "disposable_income", "household_income"):
        assert np.array_equal(
            branch.calculate(variable, JANUARY), fresh.calculate(variable, JANUARY)
        ), variable
    # The parent keeps its own values.
    assert np.array_equal(
        simulation.calculate("income_tax", JANUARY),
        _build(tax_benefit_system).calculate("income_tax", JANUARY),
    )


def test_branch_keeps_values_stored_before_the_input_first_existed(
    tax_benefit_system,
):
    """Values stored before the overridden variable was first stored stay."""
    situation = {
        "persons": {"a": {"birth": {"ETERNITY": "1980-01-01"}}},
        "households": {
            "h": {
                "parents": ["a"],
                "accommodation_size": {"2017-01": 80},
                "housing_occupancy_status": {"2017-01": "tenant"},
            }
        },
    }
    simulation = _build(tax_benefit_system, situation)
    # housing_tax is stored before salary has any value.
    simulation.calculate("housing_tax", "2017")
    simulation.set_input("salary", JANUARY, np.array([3000.0]))
    simulation.calculate("income_tax", JANUARY)

    branch = simulation.get_branch("raise")
    branch.set_input("salary", JANUARY, np.array([4000.0]))

    assert branch.calculate("income_tax", JANUARY) == pytest.approx(
        4000 * tax_benefit_system.parameters(JANUARY).taxes.income_tax_rate
    )
    assert _stored_keys(branch, "housing_tax") == {"default:2017"}


def test_branch_input_with_no_history_drops_nothing(tax_benefit_system):
    """The usual case costs nothing: no value can depend on a value never stored."""
    simulation = _build(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    branch = simulation.get_branch("later")
    before = {
        variable: _stored_keys(branch, variable)
        for variable in ("disposable_income", "income_tax", "salary")
    }

    # salary has never been stored, read or derived for 2018-03.
    dropped = branch._drop_values_that_may_depend_on(
        "salary", periods.period("2018-03")
    )
    branch.set_input("salary", "2018-03", np.array([1.0, 2.0]))

    assert dropped == 0
    for variable, keys in before.items():
        assert _stored_keys(branch, variable) >= keys, variable


def test_branch_keeps_inputs_of_the_dataset_and_of_ancestor_branches(
    tax_benefit_system,
):
    simulation = _build(tax_benefit_system)
    simulation.calculate("disposable_income", JANUARY)
    parent_branch = simulation.get_branch("parent")
    parent_branch.set_input("rent", JANUARY, np.array([500.0]))
    parent_branch.calculate("housing_allowance", JANUARY)

    child = parent_branch.get_branch("child")
    child.set_input("salary", JANUARY, np.array([0.0, 0.0]))

    assert "default:2017-01" in _stored_keys(child, "accommodation_size")
    assert "parent:2017-01" in _stored_keys(child, "rent")
    assert child.calculate("rent", JANUARY) == 500.0
    fresh = _with_salary(tax_benefit_system, [0.0, 0.0])
    fresh.set_input("rent", JANUARY, np.array([500.0]))
    for variable in ("housing_allowance", "household_income", "income_tax"):
        assert np.array_equal(
            child.calculate(variable, JANUARY), fresh.calculate(variable, JANUARY)
        ), variable


def test_input_for_a_year_drops_values_calculated_from_a_month(
    tax_benefit_system,
):
    simulation = _build(tax_benefit_system)
    simulation.calculate("income_tax", JANUARY)

    branch = simulation.get_branch("annual")
    # salary divides a yearly input over the months the branch has not set.
    branch.set_input("salary", "2017", np.array([12000.0, 24000.0]))

    rate = tax_benefit_system.parameters(JANUARY).taxes.income_tax_rate
    assert np.allclose(
        branch.calculate("income_tax", JANUARY), [1000 * rate, 2000 * rate]
    )


def test_input_set_again_on_a_branch_drops_what_the_first_input_fed(
    tax_benefit_system,
):
    simulation = _build(tax_benefit_system)
    branch = simulation.get_branch("raise")
    branch.set_input("salary", JANUARY, np.array([5000.0, 1000.0]))
    branch.calculate("disposable_income", JANUARY)
    branch.set_input("salary", JANUARY, np.array([7000.0, 1000.0]))

    fresh = _with_salary(tax_benefit_system, [7000.0, 1000.0])
    assert np.array_equal(
        branch.calculate("disposable_income", JANUARY),
        fresh.calculate("disposable_income", JANUARY),
    )


def test_set_input_on_a_simulation_that_is_not_a_branch_drops_nothing(
    tax_benefit_system,
):
    """Unchanged: a root simulation's later inputs do not reach its cache."""
    simulation = _build(tax_benefit_system)
    cached = simulation.calculate("income_tax", JANUARY).copy()
    simulation.set_input("salary", JANUARY, np.array([9000.0, 9000.0]))

    assert np.array_equal(simulation.calculate("income_tax", JANUARY), cached)


def test_existing_child_branches_keep_their_values(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    branch = simulation.get_branch("branch")
    branch.calculate("income_tax", JANUARY)
    child = branch.get_branch("child")
    child_value = child.calculate("income_tax", JANUARY).copy()

    branch.set_input("salary", JANUARY, np.array([0.0, 0.0]))

    assert np.array_equal(child.calculate("income_tax", JANUARY), child_value)
    assert np.array_equal(branch.calculate("income_tax", JANUARY), [0.0, 0.0])


def test_disk_storage_drops_calculated_values_and_keeps_inputs(tax_benefit_system):
    situation = {
        "persons": {"a": {"birth": {"ETERNITY": "1980-01-01"}}},
        "households": {"h": {"parents": ["a"]}},
    }
    simulation = _build(tax_benefit_system, situation)
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    for variable in ("rent", "housing_allowance"):
        holder = simulation.get_holder(variable)
        holder._disk_storage = holder.create_disk_storage()
        holder._on_disk_storable = True
    simulation.set_input("rent", JANUARY, np.array([700.0]))
    simulation.calculate("housing_allowance", JANUARY)
    assert simulation.get_holder("housing_allowance")._disk_storage._files

    branch = simulation.get_branch("cheaper")
    branch.set_input("rent", JANUARY, np.array([300.0]))

    dropped = branch.get_holder("housing_allowance")._disk_storage._files == {}
    fresh = _build(tax_benefit_system, situation)
    fresh.set_input("rent", JANUARY, np.array([300.0]))
    assert np.array_equal(
        branch.calculate("housing_allowance", JANUARY),
        fresh.calculate("housing_allowance", JANUARY),
    )
    assert dropped
    rent_disk = branch.get_holder("rent")._disk_storage
    assert set(rent_disk._files) == {"default_2017-01", "cheaper_2017-01"}
    assert rent_disk._input_keys == {"default_2017-01", "cheaper_2017-01"}


def test_value_served_from_the_macro_cache_is_tracked(tax_benefit_system, monkeypatch):
    """A macro-cache read is not stored, but what reads it is."""
    import policyengine_core.simulations.simulation as simulation_module

    class CachedIncomeTax:
        def __init__(self, tax_benefit_system):
            pass

        def set_cache_path(self, *args):
            pass

        def get_cache_path(self):
            return type("Path", (), {"exists": lambda self: True})()

        def get_cache_value(self, path):
            return np.array([100.0, 100.0], dtype=np.float32)

    monkeypatch.setattr(simulation_module, "SimulationMacroCache", CachedIncomeTax)
    simulation = _build(tax_benefit_system)
    simulation.macro_cache_read = True
    simulation.dataset = type(
        "Dataset", (), {"file_path": simulation_module.Path("cache"), "name": "d"}
    )()
    monkeypatch.setattr(
        type(simulation),
        "check_macro_cache",
        lambda self, variable, period: variable == "income_tax",
    )
    simulation.calculate("disposable_income", JANUARY)
    assert not _stored_keys(simulation, "income_tax")

    branch = simulation.get_branch("branch")
    branch.set_input("income_tax", JANUARY, np.array([0.0, 0.0]))

    assert not _stored_keys(branch, "disposable_income")
    fresh = _build(tax_benefit_system)
    fresh.set_input("income_tax", JANUARY, np.array([0.0, 0.0]))
    assert np.array_equal(
        branch.calculate("disposable_income", JANUARY),
        fresh.calculate("disposable_income", JANUARY),
    )


# ----- drop_computed_arrays ----- #


def test_drop_computed_arrays_keeps_only_inputs(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    simulation.calculate("total_taxes", JANUARY)
    branch = simulation.get_branch("branch")
    branch.set_input("rent", JANUARY, np.array([100.0]))
    branch.calculate("total_benefits", JANUARY)

    dropped = branch.drop_computed_arrays()

    assert dropped > 0
    remaining = {
        (name, key)
        for population in branch.populations.values()
        for name, holder in population._holders.items()
        for key in holder._memory_storage._arrays
    }
    inputs = {
        (name, key)
        for population in branch.populations.values()
        for name, holder in population._holders.items()
        for key in holder._memory_storage._input_keys
    }
    assert remaining == inputs
    assert {("rent", "branch:2017-01"), ("salary", "default:2017-01")} <= remaining
    assert not _stored_keys(branch, "total_taxes")
    assert not _stored_keys(branch, "total_benefits")
    assert np.array_equal(branch.calculate("rent", JANUARY), [100.0])
    # The parent is untouched.
    assert _stored_keys(simulation, "total_taxes") == {"default:2017-01"}


def test_drop_computed_arrays_after_changing_a_branch_parameter(tax_benefit_system):
    """Parameters are not inputs: a branch with other parameters drops its values."""
    simulation = _build(tax_benefit_system)
    simulation.calculate("income_tax", JANUARY)

    branch = simulation.get_branch("higher_rate", clone_system=True)
    branch.tax_benefit_system.parameters.taxes.income_tax_rate.update(
        period="2017", value=0.5
    )
    stale = branch.calculate("income_tax", JANUARY).copy()
    branch.drop_computed_arrays()

    assert np.array_equal(branch.calculate("income_tax", JANUARY), [2000.0, 500.0])
    assert not np.array_equal(stale, [2000.0, 500.0])


def test_apply_reform_keeps_branch_inputs(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    branch = simulation.get_branch("branch")
    branch.set_input("salary", JANUARY, np.array([10.0, 20.0]))
    branch.calculate("income_tax", JANUARY)

    class NoOp(Reform):
        def apply(self):
            pass

    simulation.apply_reform(NoOp)

    assert _stored_keys(branch, "income_tax") == set()
    assert np.array_equal(branch.calculate("salary", JANUARY), [10.0, 20.0])


def test_subsample_starts_a_new_store_history():
    """``subsample`` replaces every value, so earlier stores no longer count."""
    from policyengine_core.country_template import Microsimulation

    simulation = Microsimulation()
    simulation.calculate("income_tax", JANUARY_2022)
    last_before = next_sequence_number()
    simulation.subsample(n=3, seed="store-history", time_period="2022")

    recorded = [
        number
        for stored in simulation._store_history._first_stored.values()
        for number in stored.values()
    ]
    assert recorded and min(recorded) > last_before

    # A branch's input still reaches what the parent calculated first.
    salary = np.asarray(simulation.calculate("salary", JANUARY_2022)) + 1000
    before = simulation.get_branch("before")
    before.set_input("salary", JANUARY_2022, salary)
    expected = np.asarray(
        before.calculate("social_security_contribution", JANUARY_2022)
    )
    simulation.calculate("social_security_contribution", JANUARY_2022)
    after = simulation.get_branch("after")
    after.set_input("salary", JANUARY_2022, salary)
    assert np.array_equal(
        np.asarray(after.calculate("social_security_contribution", JANUARY_2022)),
        expected,
    )


# ----- Storage and history ----- #


def test_storage_records_sequence_numbers_and_inputs_through_clone():
    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), periods.period("2020"), is_input=True)
    first = storage._sequence_numbers["default:2020"]
    storage.put(np.array([2.0]), periods.period("2021"))
    clone = storage.clone()

    assert clone._sequence_numbers == storage._sequence_numbers
    assert clone._sequence_numbers["default:2021"] > first
    assert clone._input_keys == {"default:2020"}

    assert clone.drop_computed(since=first) == 1
    assert set(clone._arrays) == {"default:2020"}
    assert set(storage._arrays) == {"default:2020", "default:2021"}


def test_storage_drops_only_values_at_or_after_since():
    storage = InMemoryStorage(is_eternal=False)
    numbers = []
    for year in (2020, 2021, 2022):
        number = next_sequence_number()
        storage.put(np.array([0.0]), periods.period(year), sequence_number=number)
        numbers.append(number)

    assert storage.drop_computed(since=numbers[1]) == 2
    assert set(storage._arrays) == {"default:2020"}
    storage.delete(periods.period(2020))
    assert storage._sequence_numbers == {}


def test_history_matches_overlapping_periods_and_derived_values():
    history = StoreHistory()
    history.record_store("v", periods.period("2020-03"), 10)
    history.record_store("v", periods.period("2020-03"), 99)
    history.record_store("v", periods.period("2021"), 20)

    assert history.earliest_dependency("v", periods.period("2020")) == 10
    assert history.earliest_dependency("v", periods.period("2020-04")) is None
    assert history.earliest_dependency("v", periods.period("2021-06")) == 20
    assert history.earliest_dependency("w", periods.period("2020")) is None

    history.record_derived("v", 15)
    assert history.earliest_dependency("v", periods.period("2020-04")) == 15
    history.record_store("e", periods.period(periods.ETERNITY), 5)
    assert history.earliest_dependency("e", periods.period("1999-01")) == 5


# ----- Derived values and branches inside formulas, on a synthetic system ----- #


def test_value_carried_over_from_an_earlier_input_is_dropped():
    simulation = synthetic_simulation(ROOT_INPUTS, system=CARRY_OVER_SYSTEM)
    simulation.calculate("p_cond", "2015")  # p_c 2015 carried over from 2012

    branch = simulation.get_branch("branch")
    branch.set_input("p_c", "2014", np.array([70.0, 80.0, 90.0]))

    fresh = synthetic_simulation(
        {**ROOT_INPUTS, ("p_c", "2014"): (70.0, 80.0, 90.0)},
        system=CARRY_OVER_SYSTEM,
    )
    assert np.array_equal(fresh.calculate("p_c", "2015"), [70.0, 80.0, 90.0])
    for variable in ("p_c", "p_cond"):
        assert np.array_equal(
            branch.calculate(variable, "2015"), fresh.calculate(variable, "2015")
        ), variable


def test_uprated_value_is_dropped():
    simulation = synthetic_simulation(ROOT_INPUTS)
    simulation.calculate("p_prod", "2015")  # p_up 2015 uprated from 2012

    branch = simulation.get_branch("branch")
    branch.set_input("p_up", "2014", np.array([1.0, 2.0, 3.0]))

    fresh = synthetic_simulation({**ROOT_INPUTS, ("p_up", "2014"): (1.0, 2.0, 3.0)})
    assert np.allclose(
        branch.calculate("p_prod", "2015"), fresh.calculate("p_prod", "2015")
    )


def test_value_calculated_only_inside_another_branch_is_tracked():
    """A formula's own branch stores the overridden variable; its result is dropped.

    ``p_inner`` calculates ``p_inner_only`` in a branch it deletes when done,
    so the parent never stores ``p_inner_only`` itself. It takes in that
    branch's history when its calculation returns, and the parent's later
    branches copy it.
    """
    simulation = synthetic_simulation(ROOT_INPUTS)
    simulation.calculate("p_inner", "2013")
    assert not simulation.get_holder("p_inner_only")._memory_storage._arrays

    branch = simulation.get_branch("branch")
    branch.set_input("p_inner_only", "2013", np.array([1.0, 1.0, 1.0]))

    fresh = synthetic_simulation(
        {**ROOT_INPUTS, ("p_inner_only", "2013"): (1.0, 1.0, 1.0)}
    )
    assert np.array_equal(
        branch.calculate("p_inner", "2013"), fresh.calculate("p_inner", "2013")
    )


def test_value_a_holder_does_not_keep_is_tracked():
    """``variables_to_drop`` values are not stored, but what reads them is."""
    config = MemoryConfig(max_memory_occupation=1, variables_to_drop=["p_sum"])
    simulation = synthetic_simulation(ROOT_INPUTS, memory_config=config)
    simulation.calculate("p_prod", "2013")
    assert not simulation.get_holder("p_sum")._memory_storage._arrays

    branch = simulation.get_branch("branch")
    branch.set_input("p_sum", "2013", np.array([0.0, 0.0, 0.0]))

    assert np.array_equal(branch.calculate("p_prod", "2013"), [0.0, 0.0, 0.0])


def test_value_a_spiral_defaults_is_tracked():
    """The default a spiral returns is not stored, but what reads it is."""
    simulation = synthetic_simulation(ROOT_INPUTS)
    branch = simulation.get_branch("branch")
    branch.calculate("p_spiral", "2015")
    deepest = min(
        periods.period(key.split(":", 1)[1])
        for key in branch.get_holder("p_spiral")._memory_storage._arrays
    ).last_year

    branch.set_input("p_spiral", deepest, np.array([100.0, 100.0, 100.0]))

    fresh = synthetic_simulation(
        {**ROOT_INPUTS, ("p_spiral", str(deepest)): (100.0,) * 3}
    )
    assert np.array_equal(
        branch.calculate("p_spiral", "2015"), fresh.calculate("p_spiral", "2015")
    )


# ----- Values that reach a simulation without being stored there ----- #


def test_macro_cache_files_a_branch_wrote_are_not_read_after_its_input_changes(
    tmp_path,
):
    """Macro-cache files are keyed by branch and period, not by inputs."""
    from policyengine_core.data.dataset import Dataset
    from policyengine_core.entities import build_entity
    from policyengine_core.simulations import Simulation
    from policyengine_core.taxbenefitsystems import TaxBenefitSystem
    from policyengine_core.variables import Variable

    person = build_entity("person", "persons", "Person", is_person=True)

    class source(Variable):
        label = "Source"
        value_type = float
        entity = person
        definition_period = periods.YEAR

    class result(Variable):
        label = "Result"
        value_type = float
        entity = person
        definition_period = periods.YEAR
        exhaustive_parameter_dependencies = []

        def formula(person, period):
            return 2 * person("source", period)

    class OnePerson(Dataset):
        name = "one_person"
        label = "One person"
        file_path = tmp_path / "one_person.h5"
        data_format = Dataset.ARRAYS
        time_period = "2022"

        def generate(self):
            self.save_dataset({"person_id": np.array([0]), "source": np.array([1.0])})

    system = TaxBenefitSystem([person])
    system.add_variables(source, result)
    parent = Simulation(tax_benefit_system=system, dataset=OnePerson)
    parent.macro_cache_read = True
    branch = parent.get_branch("test")
    assert branch.calculate("result", "2022").tolist() == [2.0]  # writes the file

    branch.set_input("source", "2022", np.array([5.0]))

    assert branch.calculate("result", "2022").tolist() == [10.0]


def test_blacklisted_value_is_tracked():
    """With ``opt_out_cache``, blacklisted values are not stored; what reads them is."""
    simulation = synthetic_simulation(ROOT_INPUTS, opt_out_cache=True)
    simulation.calculate("p_prod", "2013")
    assert not simulation.get_holder("p_sum")._memory_storage._arrays

    branch = simulation.get_branch("branch")
    branch.set_input("p_sum", "2013", np.array([0.0, 0.0, 0.0]))

    assert np.array_equal(branch.calculate("p_prod", "2013"), [0.0, 0.0, 0.0])


def test_value_written_straight_into_storage_counts_as_a_dependency(
    tax_benefit_system,
):
    """A value stored without ``put`` has no number, so anything may depend on it."""
    simulation = _build(tax_benefit_system)
    holder = simulation.get_holder("salary")
    holder._memory_storage._arrays["default:2017-02"] = np.array([4000.0, 0.0])
    simulation.calculate("income_tax", "2017-02")

    branch = simulation.get_branch("branch")
    branch.set_input("salary", "2017-02", np.array([0.0, 0.0]))

    assert np.array_equal(branch.calculate("income_tax", "2017-02"), [0.0, 0.0])


def test_cached_value_another_simulation_returns_is_tracked():
    """A formula that reads another simulation's cached value takes in its history."""
    simulation = branching_simulation()
    # Cached in the persistent branch before any formula of the parent reads it.
    simulation.get_branch("persistent").calculate("tax", "2020")
    simulation.calculate("from_persistent_branch", "2020")  # a cache hit there

    branch = simulation.get_branch("branch")
    branch.set_input("tax", "2020", np.zeros(3))

    # A new simulation given the input first creates its persistent branch
    # from itself, so the branch's input reaches it.
    assert np.array_equal(
        branch.calculate("from_persistent_branch", "2020"), np.zeros(3)
    )


def test_history_is_taken_in_again_after_a_drop_forgets_it():
    """A drop forgets records; reading the other simulation again restores them."""
    simulation = branching_simulation()
    branch = simulation.get_branch("branch")
    branch.calculate("from_persistent_branch", "2020")
    branch.drop_computed_arrays()  # forgets what the persistent branch stored
    branch.calculate("from_persistent_branch", "2020")  # a cache hit there

    child = branch.get_branch("child")
    child.set_input("tax", "2020", np.zeros(3))

    assert np.array_equal(
        child.calculate("from_persistent_branch", "2020"), np.zeros(3)
    )


def _one_person_system(*variables):
    from policyengine_core.country_template import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    system.add_variables(*variables)
    return system


def _yearly_variable(name, formula=None):
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    namespace = dict(
        value_type=float,
        entity=entities.Person,
        definition_period=periods.YEAR,
        label=name,
    )
    if formula is not None:
        namespace["formula"] = formula
    return type(name, (Variable,), namespace)


@pytest.mark.parametrize("drop", ["drop_computed_arrays", "set_input"])
def test_drop_while_a_formula_runs_keeps_what_it_read_recorded(drop):
    """A formula still holding a value it read keeps that value's record."""

    def result(person, period):
        source = person("source", period)
        if drop == "drop_computed_arrays":
            person.simulation.drop_computed_arrays()
        else:
            person.simulation.set_input("trigger", period, np.ones(person.count))
        return source * 2

    system = _one_person_system(
        _yearly_variable("source", lambda person, period: np.ones(person.count)),
        _yearly_variable("trigger", lambda person, period: np.zeros(person.count)),
        _yearly_variable("result", result),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.calculate("trigger", "2020")  # stored before ``source``
    branch = simulation.get_branch("branch")
    assert branch.calculate("result", "2020").tolist() == [2.0]

    branch.set_input("source", "2020", np.array([3.0]))

    assert branch.calculate("result", "2020").tolist() == [6.0]


def test_disk_restore_reads_the_latest_file_of_each_key(tmp_path, monkeypatch):
    """Sequence numbers restart in each process; restore goes by write time."""
    import itertools
    import os

    import policyengine_core.data_storage.on_disk_storage as on_disk_storage
    import policyengine_core.data_storage.store_history as store_history
    from policyengine_core.data_storage import OnDiskStorage

    monkeypatch.setattr(on_disk_storage, "_PROCESS_TOKEN", "aaaaaaaaaaaa")
    writer = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    for _ in range(5):
        writer.put(np.array([1.0]), periods.period("2020"))
    # A later process: its counter restarts, and it has its own token.
    monkeypatch.setattr(store_history, "_sequence", itertools.count(1))
    monkeypatch.setattr(on_disk_storage, "_PROCESS_TOKEN", "bbbbbbbbbbbb")
    later = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    later.put(np.array([2.0]), periods.period("2020"))
    # The writer's files clearly earlier, even on a coarse file clock.
    for path in tmp_path.glob("default_2020.aaaaaaaaaaaa.*.npy"):
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns - 10**9))

    reader = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    reader.restore()

    assert reader.get(periods.period("2020")).tolist() == [2.0]


def test_uprated_value_another_simulation_returns_is_tracked():
    """Taking in another simulation's history includes its uprated values."""
    inputs = {("p_up", "2012"): (100.0, 100.0, 100.0)}
    simulation = synthetic_simulation(inputs)
    simulation.calculate("p_imported_up", "2015")  # uprated in a temporary branch

    branch = simulation.get_branch("branch")
    branch.set_input("p_up", "2014", np.full(3, 300.0))

    fresh = synthetic_simulation({**inputs, ("p_up", "2014"): (300.0,) * 3})
    assert np.allclose(
        branch.calculate("p_imported_up", "2015"),
        fresh.calculate("p_imported_up", "2015"),
    )


def test_value_a_formula_calculates_in_a_worker_thread_is_tracked():
    """A thread started without the formula's context still hands its history up."""
    from concurrent.futures import ThreadPoolExecutor

    def result(person, period):
        simulation = person.simulation
        child = simulation.get_branch("worker")
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(child.calculate, "value", period).result() * 2
        finally:
            del simulation.branches["worker"]

    system = _one_person_system(
        _yearly_variable("value", lambda person, period: np.zeros(person.count)),
        _yearly_variable("result", result),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.calculate("result", "2020")

    branch = simulation.get_branch("branch")
    branch.set_input("value", "2020", np.array([10.0]))

    assert branch.calculate("result", "2020").tolist() == [20.0]


def test_failed_prerequisite_request_does_not_satisfy_the_gate():
    def prerequisite(person, period):
        raise RuntimeError("the prerequisite failed")

    dependent = _yearly_variable(
        "dependent", lambda person, period: np.full(person.count, 42.0)
    )
    dependent.requires_computation_after = "prerequisite"
    system = _one_person_system(
        _yearly_variable("prerequisite", prerequisite), dependent
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    with pytest.raises(RuntimeError):
        simulation.calculate("prerequisite", "2020")

    with pytest.raises(ValueError, match="requires prerequisite"):
        simulation.calculate("dependent", "2020")


def test_disk_files_from_another_process_are_not_overwritten(tmp_path, monkeypatch):
    """Processes write their own files even when their sequence numbers repeat."""
    import itertools

    import policyengine_core.data_storage.on_disk_storage as on_disk_storage
    import policyengine_core.data_storage.store_history as store_history
    from policyengine_core.data_storage import OnDiskStorage

    monkeypatch.setattr(store_history, "_sequence", itertools.count(1))
    monkeypatch.setattr(on_disk_storage, "_PROCESS_TOKEN", "aaaaaaaaaaaa")
    writer = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    writer.put(np.array([1.0]), periods.period("2020"))
    snapshot = writer.clone()

    # A later process: its counter restarts.
    monkeypatch.setattr(store_history, "_sequence", itertools.count(1))
    monkeypatch.setattr(on_disk_storage, "_PROCESS_TOKEN", "bbbbbbbbbbbb")
    reader = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    reader.restore()
    reader.put(np.array([99.0]), periods.period("2020"))

    assert snapshot.get(periods.period("2020")).tolist() == [1.0]
    reader.restore()
    assert reader.get(periods.period("2020")).tolist() == [99.0]


def test_result_calculated_across_its_own_input_change_is_calculated_again():
    """A formula that read an input and then replaced it runs again from the new one."""

    def result(person, period):
        source = person("source", period)
        person.simulation.set_input("source", period, np.full(person.count, 3.0))
        return source * 2

    system = _one_person_system(
        _yearly_variable("source", lambda person, period: np.ones(person.count)),
        _yearly_variable("result", result),
    )
    branch = SimulationBuilder().build_default_simulation(system).get_branch("b")
    assert branch.calculate("result", "2020").tolist() == [6.0]

    assert branch.calculate("result", "2020").tolist() == [6.0]


def test_result_calculated_again_is_kept_for_carry_over():
    """A result that was not kept would leave its period unknown to carry-over."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    class flag(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "flag"

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "source"

        def formula_2021(person, period):
            if np.any(person("flag", "2021") == 0):
                person.simulation.set_input("flag", "2021", np.ones(person.count))
            return np.full(person.count, 3.0)

        def formula_2022(person, period):
            return None

    system = _one_person_system(flag, source)
    system.auto_carry_over_input_variables = True
    root = SimulationBuilder().build_default_simulation(system)
    root.set_input("flag", "2021", np.array([0.0]))
    branch = root.get_branch("branch")

    assert branch.calculate("source", "2021").tolist() == [3.0]
    assert branch.calculate("source", "2022").tolist() == [3.0]  # carried over


def test_result_is_calculated_again_until_no_input_changes():
    """Each run may change another input it read; the kept result follows the last."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def flag(name):
        return type(
            name,
            (Variable,),
            dict(
                value_type=float,
                entity=entities.Person,
                definition_period=periods.YEAR,
                label=name,
            ),
        )

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "source"

        def formula_2021(person, period):
            for name in ("first_flag", "second_flag"):
                if np.any(person(name, "2021") == 0):
                    person.simulation.set_input(name, "2021", np.ones(person.count))
                    break
            return np.full(person.count, 3.0)

        def formula_2022(person, period):
            return None

    system = _one_person_system(flag("first_flag"), flag("second_flag"), source)
    system.auto_carry_over_input_variables = True
    root = SimulationBuilder().build_default_simulation(system)
    root.set_input("first_flag", "2021", np.array([0.0]))
    root.set_input("second_flag", "2021", np.array([0.0]))
    branch = root.get_branch("branch")

    assert branch.calculate("source", "2021").tolist() == [3.0]
    assert branch.calculate("source", "2022").tolist() == [3.0]  # carried over


def test_direct_sum_is_taken_again_when_a_term_changes_an_earlier_input():
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        label = "source"

    class result(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        label = "result"

        def formula(person, period):
            if str(period) == "2020-02":
                january = person("source", "2020-01")
                if np.any(january == 1):
                    person.simulation.set_input(
                        "source", "2020-01", np.full(person.count, 3.0)
                    )
            return person("source", period)

    system = _one_person_system(source, result)
    root = SimulationBuilder().build_default_simulation(system)
    for month in periods.period("2020").get_subperiods(periods.MONTH):
        root.set_input("source", month, np.array([1.0]))
    branch = root.get_branch("branch")

    assert branch.calculate_add("result", "2020").tolist() == [14.0]


def test_input_a_formula_sets_wins_over_a_carried_over_default():
    """A formula returning None after setting its own period's input."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "source"

        def formula_2020(person, period):
            person.simulation.set_input("source", period, np.full(person.count, 3.0))
            return None

    system = _one_person_system(source)
    system.auto_carry_over_input_variables = True
    root = SimulationBuilder().build_default_simulation(system)
    root.set_input("source", "2021", np.array([9.0]))  # later: no carry-over
    branch = root.get_branch("branch")

    assert branch.calculate("source", "2020").tolist() == [3.0]


def test_input_set_under_the_default_key_during_a_branch_calculation_wins():
    """Holder.set_input stores under "default", which the branch reads."""

    def source(person, period):
        person.simulation.get_holder("source").set_input(
            period, np.full(person.count, 3.0)
        )
        return np.ones(person.count)

    system = _one_person_system(_yearly_variable("source", source))
    branch = SimulationBuilder().build_default_simulation(system).get_branch("b")

    assert branch.calculate("source", "2020").tolist() == [3.0]
    assert branch.calculate("source", "2020").tolist() == [3.0]


def test_direct_divide_keeps_ignoring_an_earlier_monthly_input():
    """Only an input set during the calculation wins; one stored before does not."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def store_whole_period(holder, period, array):
        holder._set(period, array)

    class flag(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "flag"

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "source"
        set_input = store_whole_period

        def formula(person, period):
            if np.any(person("flag", "2020") == 0):
                person.simulation.set_input("flag", "2020", np.ones(person.count))
            return np.full(person.count, 120.0)

    system = _one_person_system(flag, source)

    def branch(final_inputs):
        root = SimulationBuilder().build_default_simulation(system)
        root.set_input("flag", "2020", np.array([0.0]))
        root.set_input("source", "2020-01", np.array([99.0]))
        child = root.get_branch("b")
        if final_inputs:
            child.set_input("flag", "2020", np.array([1.0]))
        return child

    fresh = branch(final_inputs=True).calculate_divide("source", "2020-01")
    assert fresh.tolist() == [10.0]
    assert branch(final_inputs=False).calculate_divide(
        "source", "2020-01"
    ).tolist() == [10.0]


def test_reruns_do_not_multiply_through_nested_calculations():
    """Only the outermost calculation runs again; the inner ones run with it."""
    from collections import Counter

    calls = Counter()

    def leaf(person, period):
        calls["leaf"] += 1
        value = person("source", period)
        person.simulation.set_input("source", period, value + 1)  # every run
        return value

    def level(name, inner):
        def formula(person, period):
            calls[name] += 1
            return person(inner, period)

        return formula

    system = _one_person_system(
        _yearly_variable("source"),
        _yearly_variable("level1", leaf),
        _yearly_variable("level2", level("level2", "level1")),
        _yearly_variable("level3", level("level3", "level2")),
    )
    root = SimulationBuilder().build_default_simulation(system)
    root.set_input("source", "2020", np.array([0.0]))
    branch = root.get_branch("branch")
    branch.calculate("level3", "2020")

    reruns = simulation_module._RERUNS_AFTER_INPUT_CHANGE
    assert calls == {"leaf": reruns + 1, "level2": reruns + 1, "level3": reruns + 1}
    assert branch._calculations_in_flight == 0


def test_input_handler_calling_calculate_directly_still_drops():
    """A private _calculate (here a carry-over, no formula) also counts as calculating."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def two_months(holder, period, array):
        holder._set(period.first_month, array)
        holder.simulation._calculate("source", periods.period("2020-03"))
        holder._set(period.first_month.offset(1), array)

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        label = "source"
        set_input = two_months

    system = _one_person_system(source)
    system.auto_carry_over_input_variables = True

    def branch(handler_input):
        root = SimulationBuilder().build_default_simulation(system)
        root.set_input("source", "2020-01", np.array([1.0]))
        root.set_input("source", "2020-02", np.array([1.0]))
        child = root.get_branch("branch")
        if handler_input:
            child.set_input("source", "2020", np.array([3.0]))
        else:
            child.set_input("source", "2020-01", np.array([3.0]))
            child.set_input("source", "2020-02", np.array([3.0]))
        return child

    assert branch(handler_input=False).calculate("source", "2020-03").tolist() == [3.0]
    assert branch(handler_input=True).calculate("source", "2020-03").tolist() == [3.0]


@pytest.mark.parametrize("mode", ["calculate", "add", "divide"])
def test_result_refused_in_one_simulation_is_not_kept_by_its_caller_in_another(mode):
    """A parent formula calling into a branch whose formula changes its own input."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def variable(name, formula=None, definition_period=periods.YEAR):
        namespace = dict(
            value_type=float,
            entity=entities.Person,
            definition_period=definition_period,
            label=name,
        )
        if formula is not None:
            namespace["formula"] = formula
        return type(name, (Variable,), namespace)

    def leaf(person, period):
        previous = person("source", period.this_year)
        if np.any(previous != 3.0):
            person.simulation.set_input(
                "source", period.this_year, np.full(person.count, 3.0)
            )
        return previous * 2

    def from_child(person, period):
        worker = person.simulation.get_branch("worker")
        if mode == "add":
            return worker.calculate_add("leaf", period)
        if mode == "divide":
            return worker.calculate_divide("leaf", period.first_month)
        return worker.calculate("leaf", period)

    def child_outer(person, period):
        return person.simulation.parent_branch.calculate("from_child", period)

    def result_with(input_first):
        def result(person, period):
            simulation = person.simulation
            worker = simulation.get_branch("worker")
            try:
                if input_first:
                    worker.set_input("source", period, np.full(person.count, 3.0))
                return worker.calculate("child_outer", period)
            finally:
                del simulation.branches["worker"]

        return result

    def run(input_first):
        system = _one_person_system(
            variable("source"),
            variable("leaf", leaf, periods.MONTH if mode == "add" else periods.YEAR),
            variable("from_child", from_child),
            variable("child_outer", child_outer),
            variable("result", result_with(input_first)),
        )
        root = SimulationBuilder().build_default_simulation(system)
        root.set_input("source", "2020", np.array([1.0]))
        return (
            root.calculate("result", "2020").tolist(),
            root.calculate("from_child", "2020").tolist(),
        )

    assert run(input_first=False) == run(input_first=True)


def test_input_stored_with_an_earlier_number_still_wins():
    """Inputs are numbered when stored, whatever number a caller passes."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def january_only(holder, period, array):
        holder._set(period.first_month, array, sequence_number=1)

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        label = "source"
        set_input = january_only

        def formula(person, period):
            person.simulation.set_input("source", "2020", np.full(person.count, 42.0))
            return np.ones(person.count)

    system = _one_person_system(source)
    branch = SimulationBuilder().build_default_simulation(system).get_branch("b")

    assert branch.calculate("source", "2020-01").tolist() == [42.0]
    assert branch.get_holder("source")._is_input(periods.period("2020-01"), "b")


def test_direct_sum_runs_its_terms_again_only_as_a_whole():
    from collections import Counter

    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    calls = Counter()

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        label = "source"

    class counted(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        label = "counted"

        def formula(person, period):
            calls["counted"] += 1
            value = person("source", period)
            person.simulation.set_input("source", period, value + 1)  # every run
            return value

    system = _one_person_system(source, counted)
    root = SimulationBuilder().build_default_simulation(system)
    root.set_input("source", "2020-01", np.array([0.0]))
    branch = root.get_branch("branch")
    branch.calculate_add("counted", "2020-01")

    assert calls["counted"] == simulation_module._RERUNS_AFTER_INPUT_CHANGE + 1


def test_input_change_in_an_unrelated_simulation_leaves_this_one_caching():
    """A settled result from another family's simulation is kept, so carry-over finds it."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def leaf(person, period):
        if np.any(person("source", period) != 3):
            person.simulation.set_input("source", period, np.full(person.count, 3.0))
        return person("source", period) * 2

    other_root = SimulationBuilder().build_default_simulation(
        _one_person_system(_yearly_variable("source"), _yearly_variable("leaf", leaf))
    )
    other_root.set_input("source", "2020", np.array([1.0]))
    other = other_root.get_branch("worker")

    class result(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "result"

        def formula_2020(person, period):
            return other.calculate("leaf", "2020")

        def formula_2021(person, period):
            return None

    system = _one_person_system(result)
    system.auto_carry_over_input_variables = True
    branch = SimulationBuilder().build_default_simulation(system).get_branch("consumer")

    assert branch.calculate("result", "2020").tolist() == [6.0]
    assert branch.calculate("result", "2021").tolist() == [6.0]  # carried over


def test_refused_result_is_not_kept_by_a_caller_in_another_family():
    """A worker (its own family) calls back into the consumer, which calls the worker again."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def variable(name, formula=None):
        namespace = dict(
            value_type=float,
            entity=entities.Person,
            definition_period=periods.YEAR,
            label=name,
        )
        if formula is not None:
            namespace["formula"] = formula
        return type(name, (Variable,), namespace)

    def run(input_first):
        holders = {}

        def leaf(person, period):
            previous = person("source", period)
            if np.any(previous != 3.0):
                person.simulation.set_input(
                    "source", period, np.full(person.count, 3.0)
                )
            return previous * 2

        def worker_outer(person, period):
            return holders["consumer"].calculate("from_worker", period)

        worker_root = SimulationBuilder().build_default_simulation(
            _one_person_system(
                variable("source"),
                variable("leaf", leaf),
                variable("worker_outer", worker_outer),
            )
        )
        worker_root.set_input("source", "2020", np.array([1.0]))
        worker = worker_root.get_branch("worker")
        if input_first:
            worker.set_input("source", "2020", np.array([3.0]))

        consumer = (
            SimulationBuilder()
            .build_default_simulation(
                _one_person_system(
                    variable(
                        "from_worker",
                        lambda person, period: worker.calculate("leaf", period),
                    ),
                    variable(
                        "result",
                        lambda person, period: worker.calculate("worker_outer", period),
                    ),
                )
            )
            .get_branch("consumer")
        )
        holders["consumer"] = consumer
        return (
            consumer.calculate("result", "2020").tolist(),
            consumer.calculate("from_worker", "2020").tolist(),
        )

    assert run(input_first=False) == run(input_first=True) == ([6.0], [6.0])


def test_caller_in_the_same_family_does_not_keep_what_it_read_before_the_change():
    """A parent read its branch, then called a branch formula that changed the branch's input."""

    def run(input_first):
        def changes_source(person, period):
            if np.any(person("source", period) != 3.0):
                person.simulation.set_input(
                    "source", period, np.full(person.count, 3.0)
                )
            return np.zeros(person.count)

        def result(person, period):
            branch = person.simulation.branches["kept"]
            doubled = branch.calculate("doubled", period)  # read before the change
            branch.calculate("changes_source", period)
            return doubled

        system = _one_person_system(
            _yearly_variable("source"),
            _yearly_variable(
                "doubled", lambda person, period: person("source", period) * 2
            ),
            _yearly_variable("changes_source", changes_source),
            _yearly_variable("result", result),
        )
        root = SimulationBuilder().build_default_simulation(system)
        root.set_input("source", "2020", np.array([1.0]))
        kept = root.get_branch("kept")
        if input_first:
            kept.set_input("source", "2020", np.array([3.0]))
        first = root.calculate("result", "2020").tolist()
        return first, root.calculate("result", "2020").tolist()

    live_first, live_next = run(input_first=False)
    assert live_first == [2.0]  # documented: what it read stays read
    assert live_next == run(input_first=True)[1] == [6.0]  # but it was not kept


def test_input_a_formula_sets_for_its_own_period_wins():
    """As it would had the input been set before the calculation."""

    def source(person, period):
        person.simulation.set_input("source", period, np.full(person.count, 3.0))
        return np.ones(person.count)

    system = _one_person_system(_yearly_variable("source", source))
    branch = SimulationBuilder().build_default_simulation(system).get_branch("b")

    assert branch.calculate("source", "2020").tolist() == [3.0]
    assert branch.calculate("source", "2020").tolist() == [3.0]
    assert branch.get_holder("source")._is_input(periods.period("2020"), "b")


def test_input_handler_that_calculates_between_its_stores():
    """Values a handler calculates before its last store are dropped after it."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def two_months(holder, period, array):
        holder._set(period.first_month, array)
        holder.simulation.calculate("total", period)
        holder._set(period.first_month.offset(1), array)

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        label = "source"
        set_input = two_months

    class total(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "total"

        def formula(person, period):
            first = period.first_month
            return person("source", first) + person("source", first.offset(1))

    system = _one_person_system(source, total)
    root = SimulationBuilder().build_default_simulation(system)
    root.set_input("source", "2020-01", np.array([1.0]))
    root.set_input("source", "2020-02", np.array([1.0]))
    branch = root.get_branch("branch")
    branch.set_input("source", "2020", np.array([3.0]))

    assert branch.calculate("total", "2020").tolist() == [6.0]


def test_input_handler_that_fails_after_calculating():
    """The inputs it stored stay, so what it calculated before them drops."""
    from policyengine_core.country_template import entities
    from policyengine_core.variables import Variable

    def two_months_then_fail(holder, period, array):
        holder._set(period.first_month, array)
        holder.simulation.calculate("total", period)
        holder._set(period.first_month.offset(1), array)
        raise RuntimeError("handler failed")

    class source(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        label = "source"
        set_input = two_months_then_fail

    class total(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "total"

        def formula(person, period):
            first = period.first_month
            return person("source", first) + person("source", first.offset(1))

    system = _one_person_system(source, total)
    root = SimulationBuilder().build_default_simulation(system)
    root.set_input("source", "2020-01", np.array([1.0]))
    root.set_input("source", "2020-02", np.array([1.0]))
    branch = root.get_branch("branch")
    with pytest.raises(RuntimeError, match="handler failed"):
        branch.set_input("source", "2020", np.array([3.0]))

    assert branch.calculate("total", "2020").tolist() == [6.0]


def test_branch_input_stops_macro_reads_before_any_read(monkeypatch):
    """A macro file for the branch's name may come from a branch without the input."""
    import policyengine_core.simulations.simulation as simulation_module

    class CachedResult:
        def __init__(self, tax_benefit_system):
            pass

        def set_cache_path(self, *args):
            pass

        def get_cache_path(self):
            return type("Path", (), {"exists": lambda self: True})()

        def get_cache_value(self, path):
            return np.array([2.0])  # written by another simulation's "override"

        def set_cache_value(self, path, value):
            pass

    monkeypatch.setattr(simulation_module, "SimulationMacroCache", CachedResult)
    system = _one_person_system(
        _yearly_variable("source", lambda person, period: np.ones(person.count)),
        _yearly_variable("result", lambda person, period: person("source", period) * 2),
        _yearly_variable("total", lambda person, period: person("result", period) * 2),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    system.data_modified = False
    simulation.macro_cache_read = True
    simulation.dataset = type(
        "Dataset", (), {"file_path": simulation_module.Path("cache"), "name": "d"}
    )()
    monkeypatch.setattr(
        type(simulation),
        "check_macro_cache",
        lambda self, variable, period: variable == "result",
    )

    branch = simulation.get_branch("override")
    branch.set_input("source", "2020", np.array([3.0]))  # before any calculation

    assert branch.calculate("total", "2020").tolist() == [12.0]


def test_result_obsolete_after_its_own_input_change_is_not_written_to_the_macro_cache(
    monkeypatch,
):
    import policyengine_core.simulations.simulation as simulation_module

    written = []

    class RecordingCache:
        def __init__(self, tax_benefit_system):
            pass

        def set_cache_path(self, *args):
            pass

        def get_cache_path(self):
            return type("Path", (), {"exists": lambda self: False})()

        def get_cache_value(self, path):
            return None

        def set_cache_value(self, path, value):
            written.append(np.array(value).tolist())

    def result(person, period):
        source = person("source", period)
        if np.any(source == 1):
            person.simulation.set_input("source", period, np.full(person.count, 3.0))
        return source * 2

    monkeypatch.setattr(simulation_module, "SimulationMacroCache", RecordingCache)
    system = _one_person_system(
        _yearly_variable("source", lambda person, period: np.ones(person.count)),
        _yearly_variable("result", result),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.dataset = type(
        "Dataset", (), {"file_path": simulation_module.Path("cache"), "name": "d"}
    )()
    monkeypatch.setattr(
        type(simulation),
        "check_macro_cache",
        lambda self, variable, period: variable == "result",
    )
    branch = simulation.get_branch("b")

    assert branch.calculate("result", "2020").tolist() == [6.0]
    assert [2.0] not in written  # the first run's result, from the old input


def test_macro_cache_read_counts_as_depending_on_every_input(monkeypatch):
    """What a macro-cache value was calculated from was never calculated here."""
    import policyengine_core.simulations.simulation as simulation_module

    class CachedResult:
        def __init__(self, tax_benefit_system):
            pass

        def set_cache_path(self, *args):
            pass

        def get_cache_path(self):
            return type("Path", (), {"exists": lambda self: True})()

        def get_cache_value(self, path):
            return np.array([2.0])

        def set_cache_value(self, path, value):
            pass

    monkeypatch.setattr(simulation_module, "SimulationMacroCache", CachedResult)
    system = _one_person_system(
        _yearly_variable("source", lambda person, period: np.ones(person.count)),
        _yearly_variable("result", lambda person, period: person("source", period) * 2),
        _yearly_variable("total", lambda person, period: person("result", period) * 2),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    system.data_modified = False  # adding variables marks the system modified
    simulation.macro_cache_read = True
    simulation.dataset = type(
        "Dataset", (), {"file_path": simulation_module.Path("cache"), "name": "d"}
    )()
    monkeypatch.setattr(
        type(simulation),
        "check_macro_cache",
        lambda self, variable, period: variable == "result",
    )
    assert simulation.calculate("total", "2020").tolist() == [4.0]
    assert not simulation.get_holder("source").get_known_periods()  # never calculated

    branch = simulation.get_branch("branch")
    branch.set_input("source", "2020", np.array([3.0]))  # never calculated here

    assert branch.calculate("total", "2020").tolist() == [12.0]


def test_history_hand_back_failure_still_ends_the_calculation(monkeypatch):
    def fail(self):
        raise RuntimeError("hand-back failed")

    system = _one_person_system(
        _yearly_variable("source", lambda person, period: np.ones(person.count))
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    monkeypatch.setattr(type(simulation), "_share_store_history_with_caller", fail)

    with pytest.raises(RuntimeError, match="hand-back failed"):
        simulation.calculate("source", "2020")

    assert simulation._calculations_in_flight == 0
    assert not simulation.tracer.stack


def test_failed_calculation_in_another_simulation_hands_its_history_back():
    """Whether a calculation fails can depend on what it read."""

    def checked(person, period):
        source = person("source", period)
        if (source == 0).any():
            raise ValueError("source is zero")
        return source * 2

    def result(person, period):
        simulation = person.simulation
        child = simulation.get_branch("child")
        try:
            return child.calculate("checked", period)
        except ValueError:
            return np.full(person.count, 7.0)
        finally:
            del simulation.branches["child"]

    system = _one_person_system(
        _yearly_variable("source", lambda person, period: np.zeros(person.count)),
        _yearly_variable("checked", checked),
        _yearly_variable("result", result),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    assert simulation.calculate("result", "2020").tolist() == [7.0]

    branch = simulation.get_branch("branch")
    branch.set_input("source", "2020", np.array([3.0]))

    assert branch.calculate("result", "2020").tolist() == [6.0]


def test_restored_simulation_drops_restored_values_an_input_may_have_fed(tmp_path):
    from policyengine_core.tools.simulation_dumper import (
        dump_simulation,
        restore_simulation,
    )

    system = _one_person_system(
        _yearly_variable("source"),
        _yearly_variable("result", lambda person, period: person("source", period) * 2),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.set_input("source", "2020", np.array([1.0]))
    simulation.calculate("result", "2020")
    dump_simulation(simulation, str(tmp_path / "dump"))

    restored = restore_simulation(str(tmp_path / "dump"), system)
    branch = restored.get_branch("branch")
    branch.set_input("source", "2020", np.array([3.0]))

    assert branch.calculate("result", "2020").tolist() == [6.0]
    assert restored.calculate("source", "2020").tolist() == [1.0]  # still an input


@pytest.mark.parametrize("read", ["carried_over", "not_kept"])
def test_restored_values_drop_without_records_of_what_they_read(tmp_path, read):
    """A dump keeps values, not what each was calculated from."""
    from policyengine_core.tools.simulation_dumper import (
        dump_simulation,
        restore_simulation,
    )

    system = _one_person_system(
        _yearly_variable("source"),
        _yearly_variable(
            "result", lambda person, period: person("source", period) * 2 + 7
        ),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    if read == "carried_over":
        system.auto_carry_over_input_variables = True
        simulation.set_input("source", "2020", np.array([1.0]))
        override, fresh_inputs = "2021", {"2020": 1.0, "2021": 3.0}
    else:
        system.cache_blacklist = {"source"}
        simulation.opt_out_cache = True
        override, fresh_inputs = "2022", {"2022": 3.0}
    simulation.calculate("result", "2022")
    dump_simulation(simulation, str(tmp_path / "dump"))

    restored = restore_simulation(str(tmp_path / "dump"), system)
    branch = restored.get_branch("branch")
    branch.set_input("source", override, np.array([3.0]))

    fresh = SimulationBuilder().build_default_simulation(system)
    for period, value in fresh_inputs.items():
        fresh.set_input("source", period, np.array([value]))
    assert (
        branch.calculate("result", "2022").tolist()
        == fresh.calculate("result", "2022").tolist()
    )


def test_dump_of_a_branch_holds_the_branch_values(tmp_path):
    from policyengine_core.tools.simulation_dumper import (
        dump_simulation,
        restore_simulation,
    )

    system = _one_person_system(
        _yearly_variable("source"),
        _yearly_variable("result", lambda person, period: person("source", period) * 2),
    )
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.set_input("source", "2020", np.array([1.0]))
    simulation.set_input("source", "2021", np.array([5.0]))
    branch = simulation.get_branch("branch")
    branch.set_input("source", "2020", np.array([3.0]))
    branch.calculate("result", "2020")
    dump_simulation(branch, str(tmp_path / "dump"))

    restored = restore_simulation(str(tmp_path / "dump"), system)

    assert restored.calculate("source", "2020").tolist() == [3.0]
    assert restored.calculate("source", "2021").tolist() == [5.0]  # inherited
    assert restored.calculate("result", "2020").tolist() == [6.0]
    holder = restored.get_holder("source")
    assert holder._is_input(periods.period("2020"))
    assert holder._is_input(periods.period("2021"))


def test_merge_reads_again_what_was_recorded_while_it_merged():
    """A record added to the source during a merge (another thread) is not skipped."""
    source = StoreHistory()
    source.record_store("a", periods.period("2020"), 1)
    destination = StoreHistory()
    original = destination.record_store
    calls = []

    def record_store(variable_name, period, sequence_number):
        if not calls:  # the other thread records while this merge runs
            source.record_store("v", periods.period("2020"), 2)
        calls.append(variable_name)
        original(variable_name, period, sequence_number)

    destination.record_store = record_store
    destination.merge(source)
    destination.merge(source)

    assert destination.earliest_dependency("v", periods.period("2020")) == 2


def test_history_pickled_before_journals_still_records():
    import pickle

    history = StoreHistory()
    history.record_store("v", periods.period("2020"), 1)
    del history._journal, history._generation  # as pickled before journals
    restored = pickle.loads(pickle.dumps(history))

    restored.record_store("w", periods.period("2020"), 2)

    assert restored.earliest_dependency("w", periods.period("2020")) == 2


def test_disk_restore_reads_older_file_names_and_breaks_time_ties(tmp_path):
    import os

    import policyengine_core.data_storage.on_disk_storage as on_disk_storage
    from policyengine_core.data_storage import OnDiskStorage

    # A file named before process tokens, and one from another process.
    np.save(tmp_path / "default_2020.4.npy", np.array([1.0]))
    np.save(tmp_path / "default_2020.cccccccccccc.999999999999.npy", np.array([2.0]))
    storage = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    storage.put(np.array([3.0]), periods.period("2020"))  # this process, number lower
    for path in tmp_path.glob("*.npy"):
        os.utime(path, ns=(10**18, 10**18))  # a coarse clock: every time ties

    reader = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    reader.restore()

    assert set(reader._files) == {"default_2020"}  # one key, old name included
    assert reader.get(periods.period("2020")).tolist() == [3.0]
    assert on_disk_storage._PROCESS_TOKEN in reader._files["default_2020"]


def test_forked_process_gets_its_own_disk_file_token():
    import os
    import warnings

    import policyengine_core.data_storage.on_disk_storage as on_disk_storage

    if not hasattr(os, "fork"):
        pytest.skip("no fork on this platform")
    # A bare fork whose child only writes to a pipe: a process pool forked
    # from a multi-threaded test process can deadlock.
    read, write = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        try:
            os.write(write, on_disk_storage._PROCESS_TOKEN.encode())
        finally:
            os._exit(0)
    os.close(write)
    child_token = os.read(read, 64).decode()
    os.close(read)
    os.waitpid(pid, 0)

    assert child_token and child_token != on_disk_storage._PROCESS_TOKEN


# ----- Only what may depend on the input is dropped ----- #


def test_input_on_one_branch_does_not_make_another_drop(tax_benefit_system):
    simulation = _build(tax_benefit_system)
    sibling = simulation.get_branch("sibling")
    sibling.set_input("rent", "2017-02", np.array([500.0]))
    sibling.calculate("housing_allowance", "2017-02")
    simulation.calculate("disposable_income", JANUARY)
    held = {
        (name, key)
        for name in ("disposable_income", "income_tax")
        for key in _stored_keys(simulation, name)
    }

    branch = simulation.get_branch("branch")
    dropped = branch._drop_values_that_may_depend_on("rent", periods.period("2017-02"))

    assert dropped == 0
    assert held <= {
        (name, key)
        for name in ("disposable_income", "income_tax")
        for key in _stored_keys(branch, name)
    }


def test_branches_that_choose_between_overrides_calculate_shared_values_once():
    """Values the overridden variable cannot reach are calculated once per arm.

    ``choose`` compares ``tax`` in two branches that set it, as
    policyengine-us itemization does; ``agi`` does not depend on it.
    """
    # A marginal-rate branch after the parent has calculated everything.
    simulation = branching_simulation()
    simulation.calculate("net", "2020")
    FORMULA_RUNS["agi"] = 0
    rate = simulation.calculate("marginal_rate", "2020")
    assert FORMULA_RUNS["agi"] == 1
    assert np.allclose(rate, 0.85)

    # A baseline branch calculating after the reform arm has.
    FORMULA_RUNS["agi"] = 0
    simulation = branching_simulation()
    baseline = simulation.get_branch("baseline")
    simulation.calculate("net", "2020")
    baseline.calculate("net", "2020")
    assert FORMULA_RUNS["agi"] == 2

    fresh = branching_simulation()
    assert np.array_equal(
        baseline.calculate("net", "2020"), fresh.calculate("net", "2020")
    )


# ----- Other paths ----- #


def test_prerequisite_requested_before_a_branch_input_still_counts():
    """``requires_computation_after`` holds if the prerequisite was requested, even if dropped."""
    from policyengine_core.entities import build_entity
    from policyengine_core.taxbenefitsystems import TaxBenefitSystem
    from policyengine_core.variables import Variable

    person = build_entity("person", "persons", "Person", is_person=True)

    class source(Variable):
        label = "Source"
        value_type = float
        entity = person
        definition_period = periods.YEAR

    class prerequisite(Variable):
        label = "Prerequisite"
        value_type = float
        entity = person
        definition_period = periods.YEAR

        def formula(person, period):
            return person("source", period) * 0 + 1

    class result(Variable):
        label = "Result"
        value_type = float
        entity = person
        definition_period = periods.YEAR
        requires_computation_after = "prerequisite"

        def formula(person, period):
            return person("source", period) * 2

    system = TaxBenefitSystem([person])
    system.add_variables(source, prerequisite, result)
    simulation = SimulationBuilder().build_default_simulation(system)
    simulation.set_input("source", "2022", np.array([1.0]))
    simulation.calculate("prerequisite", "2022")
    simulation.calculate("result", "2022")

    branch = simulation.get_branch("branch")
    branch.set_input("source", "2022", np.array([2.0]))

    assert branch.calculate("result", "2022").tolist() == [4.0]


def test_values_a_custom_input_handler_calculates_are_not_inputs():
    from policyengine_core.entities import build_entity
    from policyengine_core.taxbenefitsystems import TaxBenefitSystem
    from policyengine_core.variables import Variable

    person = build_entity("person", "persons", "Person", is_person=True)

    def store_first_month_then_calculate(holder, period, array):
        holder._set(period.first_month, array)
        holder.simulation.calculate("dependent", period)

    class dispatched(Variable):
        label = "Dispatched"
        value_type = float
        entity = person
        definition_period = periods.MONTH
        set_input = store_first_month_then_calculate

    class dependent(Variable):
        label = "Dependent"
        value_type = float
        entity = person
        definition_period = periods.YEAR

        def formula(person, period):
            return 2 * person("dispatched", period.first_month)

    system = TaxBenefitSystem([person])
    system.add_variables(dispatched, dependent)
    branch = SimulationBuilder().build_default_simulation(system).get_branch("branch")
    branch.set_input("dispatched", "2020", np.array([3.0]))
    assert branch.calculate("dependent", "2020").tolist() == [6.0]

    branch.set_input("dispatched", "2020-01", np.array([4.0]))

    assert branch.calculate("dependent", "2020").tolist() == [8.0]
    assert not branch.get_holder("dependent")._memory_storage._input_keys


def test_year_input_after_calculating_one_of_its_months():
    """The drop comes before a yearly input is divided, so only input months count."""
    simulation = synthetic_simulation(ROOT_INPUTS)
    branch = simulation.get_branch("branch")
    branch.calculate("p_month", "2015-03")  # 2015 has no p_m input: default

    branch.set_input("p_m", "2015", np.array([120.0, 120.0, 120.0]))

    fresh = synthetic_simulation(
        {**ROOT_INPUTS, **{("p_m", f"2015-{m:02d}"): (10.0,) * 3 for m in range(1, 13)}}
    )
    assert np.allclose(
        branch.calculate("p_month", "2015-03"), fresh.calculate("p_month", "2015-03")
    )


def test_derivative_keeps_inputs_set_after_construction():
    """``derivative`` keeps every input of the simulation it differentiates."""
    simulation = synthetic_simulation({("p_a", "2013"): (1.0, 1.0, 1.0)})
    branch = simulation.get_branch("branch")
    branch.set_input("p_b", "2013", np.array([9.0, 9.0, 9.0]))

    assert np.allclose(branch.derivative("p_sum", "p_b", "2013"), 2.0)
    assert np.array_equal(branch.calculate("p_b", "2013"), [9.0, 9.0, 9.0])


def test_disk_branch_keeps_its_values_when_its_parent_recalculates():
    """Each disk store writes its own file, so a recalculation leaves children's files."""
    config = MemoryConfig(max_memory_occupation=0)
    simulation = synthetic_simulation(ROOT_INPUTS, memory_config=config)
    branch = simulation.get_branch("branch")
    before = branch.calculate("p_sum", "2013").copy()
    child = branch.get_branch("child")

    branch.set_input("p_a", "2013", np.zeros(3))
    branch.calculate("p_sum", "2013")  # recalculated: a new file

    assert np.array_equal(child.calculate("p_sum", "2013"), before)


def test_disk_branches_with_the_same_name_keep_their_own_values():
    config = MemoryConfig(max_memory_occupation=0)
    simulation = synthetic_simulation(ROOT_INPUTS, memory_config=config)
    first = simulation.get_branch("a").get_branch("leaf")
    second = simulation.get_branch("b").get_branch("leaf")
    first.set_input("p_a", "2013", np.full(3, 3.0))
    second.set_input("p_a", "2013", np.full(3, 4.0))
    # A name reused after its branch is forgotten.
    old = simulation.get_branch("reused")
    old.set_input("p_a", "2013", np.full(3, 7.0))
    del simulation.branches["reused"]
    new = simulation.get_branch("reused")
    new.set_input("p_a", "2013", np.full(3, 8.0))

    assert np.array_equal(first.calculate("p_a", "2013"), [3.0] * 3)
    assert np.array_equal(second.calculate("p_a", "2013"), [4.0] * 3)
    assert np.array_equal(old.calculate("p_a", "2013"), [7.0] * 3)
    assert np.array_equal(new.calculate("p_a", "2013"), [8.0] * 3)


def test_values_unpickled_from_another_process_stay_earlier(monkeypatch):
    """Sequence numbers restart in each process; unpickling moves this one past them."""
    import itertools
    import pickle

    import policyengine_core.data_storage.store_history as store_history

    storage = InMemoryStorage(is_eternal=False)
    history = StoreHistory()
    for _ in range(100):
        next_sequence_number()
    storage.put(np.array([1.0]), periods.period("2020"))
    history.record_store("v", periods.period("2020"), next_sequence_number())
    source = StoreHistory()  # kept alive: its weak read position must not stop pickling
    history.merge(source)
    assert len(history._merged) == 1
    payloads = pickle.dumps(storage), pickle.dumps(history)

    monkeypatch.setattr(store_history, "_sequence", itertools.count(1))  # a new process
    restored_storage, restored_history = (pickle.loads(p) for p in payloads)
    restored_storage.put(np.array([2.0]), periods.period("2021"))

    numbers = restored_storage._sequence_numbers
    assert numbers["default:2021"] > numbers["default:2020"]
    assert next_sequence_number() > restored_history.earliest_dependency(
        "v", periods.period("2020")
    )


def test_storage_pickled_before_stores_were_numbered_still_stores():
    import pickle

    storage = InMemoryStorage(is_eternal=False)
    storage.put(np.array([1.0]), periods.period("2020"))
    del storage._sequence_numbers, storage._input_keys  # as pickled before
    restored = pickle.loads(pickle.dumps(storage))

    restored.put(np.array([2.0]), periods.period("2021"), is_input=True)

    assert restored._input_keys == {"default:2021"}
