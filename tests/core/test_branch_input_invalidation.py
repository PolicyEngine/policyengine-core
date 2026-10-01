"""``set_input`` on a branch drops what may depend on the value it replaces.

A branch starts with every value its parent holds. ``set_input`` on a branch
used to store the new value and keep everything calculated from the old one,
so a value the parent calculated before branching answered for the branch,
whatever the branch's inputs said, and the branch's results depended on what
had been calculated before it was created.

Every stored value now carries a sequence number from one process-wide
counter, and a value is stored after everything it was calculated from. An
input set on a branch drops the branch's non-input values stored at or after
the first time anything in the family stored, or derived, the overridden
variable for an overlapping period (``StoreHistory.earliest_dependency``).

The invariant tested throughout (and by the property test at the end, over
random sequences of calculations, branches and inputs): whatever the
branch's family calculated before, and in whatever order, a calculation in a
branch equals the same calculation in a new simulation given the branch's
inputs, its own and those it inherited when it was created, before
calculating anything.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.data_storage.store_history import (
    StoreHistory,
    next_sequence_number,
)
from policyengine_core.experimental import MemoryConfig
from policyengine_core.holders import set_input_divide_by_period
from policyengine_core.model_api import Reform
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable

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

    assert _stored_keys(branch, "housing_tax") == {"default:2017"}
    assert _stored_keys(branch, "income_tax") == set()
    assert branch.calculate("income_tax", JANUARY) == pytest.approx(
        4000 * tax_benefit_system.parameters(JANUARY).taxes.income_tax_rate
    )


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

    assert branch.get_holder("housing_allowance")._disk_storage._files == {}
    rent_disk = branch.get_holder("rent")._disk_storage
    assert set(rent_disk._files) == {"default_2017-01", "cheaper_2017-01"}
    assert rent_disk._input_keys == {"default_2017-01", "cheaper_2017-01"}
    fresh = _build(tax_benefit_system, situation)
    fresh.set_input("rent", JANUARY, np.array([300.0]))
    assert np.array_equal(
        branch.calculate("housing_allowance", JANUARY),
        fresh.calculate("housing_allowance", JANUARY),
    )


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


# ----- A synthetic system for derived values and branches inside formulas ----- #


def _yearly(name, formula=None, **attributes):
    namespace = dict(
        value_type=float,
        entity=entities.Person,
        definition_period=periods.YEAR,
        label=name,
        **attributes,
    )
    if formula is not None:
        namespace["formula"] = formula
    return type(name, (Variable,), namespace)


def _inner_branch_formula(person, period):
    """Calculate a value in a branch that exists only while this formula runs."""
    simulation = person.simulation
    inner = simulation.get_branch("inner")
    try:
        inner.set_input("p_switch", period, np.ones(person.count))
        inner_value = inner.calculate("p_inner_only", period)
    finally:
        del simulation.branches["inner"]
    return inner_value + person("p_sum", period)


def _monthly_formula(person, period):
    return (
        person("p_a", period)
        + person("p_b", period.this_year) / 12
        + person("p_m", period)
    )


def _monthly(name, formula=None, **attributes):
    namespace = dict(
        value_type=float,
        entity=entities.Person,
        definition_period=periods.MONTH,
        label=name,
        **attributes,
    )
    if formula is not None:
        namespace["formula"] = formula
    return type(name, (Variable,), namespace)


SYNTHETIC_VARIABLES = [
    # Inputs. p_up is uprated by a parameter that changes every year from
    # 2012 to 2015; p_c is given for 2012 only, and carried over where the
    # system carries inputs over.
    _yearly("p_a"),
    _yearly("p_b"),
    _yearly("p_up", uprating="taxes.income_tax_rate"),
    _yearly("p_c"),
    _yearly(
        "p_sum",
        lambda person, period: person("p_a", period) + 2 * person("p_b", period),
    ),
    _yearly(
        "p_prod",
        lambda person, period: (
            person("p_sum", period) * (1 + person("p_up", period) / 100)
        ),
    ),
    _yearly(
        "p_cond",
        lambda person, period: np.where(
            person("p_sum", period) > 60,
            person("p_prod", period),
            person("p_c", period),
        ),
    ),
    _yearly(
        "p_lag",
        lambda person, period: (
            person("p_sum", period.last_year) + person("p_a", period)
        ),
    ),
    # A monthly input; an input for a year is divided between its months.
    _monthly("p_m", set_input=set_input_divide_by_period),
    _monthly("p_month", _monthly_formula),
    _yearly("p_months", lambda person, period: person("p_month", period)),
    _yearly("p_switch", lambda person, period: np.zeros(person.count)),
    _yearly(
        "p_inner_only",
        lambda person, period: 3 * person("p_a", period) + person("p_switch", period),
    ),
    _yearly("p_inner", _inner_branch_formula),
    # Reads itself a year earlier, back until core's spiral detection gives
    # up and returns the default.
    _yearly(
        "p_spiral", lambda person, period: person("p_spiral", period.last_year) + 1
    ),
]


def _synthetic_system(carry_over):
    class System(CountryTaxBenefitSystem):
        auto_carry_over_input_variables = carry_over

    system = System()
    system.add_variables(*SYNTHETIC_VARIABLES)
    return system


# Carry-over is left out of the random programs below: core's carry-over
# itself depends on calculation order (a later period calculated first stops
# an earlier input being carried into the years between), in any simulation.
SYNTHETIC_SYSTEM = _synthetic_system(carry_over=False)
CARRY_OVER_SYSTEM = _synthetic_system(carry_over=True)

PEOPLE = 3
YEARS = ["2012", "2013", "2014", "2015"]


def _synthetic(inputs, memory_config=None, system=SYNTHETIC_SYSTEM):
    """A simulation of ``PEOPLE`` people given ``inputs`` before anything else."""
    simulation = SimulationBuilder().build_default_simulation(system, count=PEOPLE)
    if memory_config is not None:
        simulation.memory_config = memory_config
        # Holders read the memory configuration when created; nothing is
        # stored yet, so create them again. Create them all here, so every
        # branch's holders are copies of these: a holder first created in a
        # branch would own (and remove on deletion) a directory the whole
        # family's disk storage shares.
        for population in simulation.populations.values():
            population._holders = {}
        for variable in system.variables:
            simulation.get_holder(variable)
    for (variable, period), values in sorted(
        inputs.items(),
        key=lambda item: periods.key_period_size(periods.period(item[0][1])),
    ):
        simulation.set_input(variable, period, np.asarray(values, dtype=float))
    return simulation


ROOT_INPUTS = {
    **{("p_a", year): (10.0 * i, 20.0, 35.0 + i) for i, year in enumerate(YEARS)},
    **{("p_b", year): (5.0, 15.0 + i, 25.0) for i, year in enumerate(YEARS)},
    ("p_up", "2012"): (100.0, 200.0, 300.0),
    ("p_c", "2012"): (7.0, 8.0, 9.0),
    **{
        ("p_m", f"{year}-{month:02d}"): (1.0 * month, 2.0, 0.5 * month)
        for year in (2013, 2014)
        for month in range(1, 13)
    },
}


def test_value_carried_over_from_an_earlier_input_is_dropped():
    simulation = _synthetic(ROOT_INPUTS, system=CARRY_OVER_SYSTEM)
    simulation.calculate("p_cond", "2015")  # p_c 2015 carried over from 2012

    branch = simulation.get_branch("branch")
    branch.set_input("p_c", "2014", np.array([70.0, 80.0, 90.0]))

    fresh = _synthetic(
        {**ROOT_INPUTS, ("p_c", "2014"): (70.0, 80.0, 90.0)},
        system=CARRY_OVER_SYSTEM,
    )
    assert np.array_equal(fresh.calculate("p_c", "2015"), [70.0, 80.0, 90.0])
    for variable in ("p_c", "p_cond"):
        assert np.array_equal(
            branch.calculate(variable, "2015"), fresh.calculate(variable, "2015")
        ), variable


def test_uprated_value_is_dropped():
    simulation = _synthetic(ROOT_INPUTS)
    simulation.calculate("p_prod", "2015")  # p_up 2015 uprated from 2012

    branch = simulation.get_branch("branch")
    branch.set_input("p_up", "2014", np.array([1.0, 2.0, 3.0]))

    fresh = _synthetic({**ROOT_INPUTS, ("p_up", "2014"): (1.0, 2.0, 3.0)})
    assert np.allclose(
        branch.calculate("p_prod", "2015"), fresh.calculate("p_prod", "2015")
    )


def test_value_calculated_only_inside_another_branch_is_tracked():
    """A formula's own branch stores the overridden variable; its result is dropped.

    ``p_inner`` calculates ``p_inner_only`` in a branch it deletes when done,
    so the parent and its later branches never store ``p_inner_only``. The
    history the whole family shares still records it.
    """
    simulation = _synthetic(ROOT_INPUTS)
    simulation.calculate("p_inner", "2013")
    assert not simulation.get_holder("p_inner_only")._memory_storage._arrays

    branch = simulation.get_branch("branch")
    branch.set_input("p_inner_only", "2013", np.array([1.0, 1.0, 1.0]))

    fresh = _synthetic({**ROOT_INPUTS, ("p_inner_only", "2013"): (1.0, 1.0, 1.0)})
    assert np.array_equal(
        branch.calculate("p_inner", "2013"), fresh.calculate("p_inner", "2013")
    )


def test_value_a_holder_does_not_keep_is_tracked():
    """``variables_to_drop`` values are not stored, but what reads them is."""
    config = MemoryConfig(max_memory_occupation=1, variables_to_drop=["p_sum"])
    simulation = _synthetic(ROOT_INPUTS, memory_config=config)
    simulation.calculate("p_prod", "2013")
    assert not simulation.get_holder("p_sum")._memory_storage._arrays

    branch = simulation.get_branch("branch")
    branch.set_input("p_sum", "2013", np.array([0.0, 0.0, 0.0]))

    assert np.array_equal(branch.calculate("p_prod", "2013"), [0.0, 0.0, 0.0])


def test_value_a_spiral_defaults_is_tracked():
    """The default a spiral returns is not stored, but what reads it is."""
    simulation = _synthetic(ROOT_INPUTS)
    branch = simulation.get_branch("branch")
    branch.calculate("p_spiral", "2015")
    deepest = min(
        periods.period(key.split(":", 1)[1])
        for key in branch.get_holder("p_spiral")._memory_storage._arrays
    ).last_year

    branch.set_input("p_spiral", deepest, np.array([100.0, 100.0, 100.0]))

    fresh = _synthetic({**ROOT_INPUTS, ("p_spiral", str(deepest)): (100.0,) * 3})
    assert np.array_equal(
        branch.calculate("p_spiral", "2015"), fresh.calculate("p_spiral", "2015")
    )


# ----- Property: branch results do not depend on what was calculated before ----- #

SETTABLE = [
    "p_a",
    "p_b",
    "p_up",
    "p_c",
    "p_m",
    "p_sum",
    "p_prod",
    "p_switch",
    "p_inner_only",
]
CALCULABLE = [
    "p_sum",
    "p_prod",
    "p_cond",
    "p_lag",
    "p_months",
    "p_inner",
    "p_inner_only",
    "p_up",
    "p_c",
]
MONTHS = ["2013-01", "2013-07", "2014-12", "2015-03"]
MEMORY_CONFIGS = {
    "memory": lambda: None,
    "disk": lambda: MemoryConfig(max_memory_occupation=0),
    "not_kept": lambda: MemoryConfig(
        max_memory_occupation=1, variables_to_drop=["p_sum", "p_inner_only"]
    ),
}

# Indices pick a simulation modulo how many exist (-1: the newest); small ones
# keep most operations on the root and the first few branches, where they
# interact.
simulation_index = st.integers(0, 3)
values = st.tuples(*[st.integers(0, 100).map(float)] * PEOPLE)
calculate = st.one_of(
    st.tuples(
        st.just("calculate"),
        simulation_index,
        st.sampled_from(CALCULABLE),
        st.sampled_from(YEARS),
    ),
    st.tuples(
        st.just("calculate"),
        simulation_index,
        st.just("p_month"),
        st.sampled_from(MONTHS),
    ),
)
set_input = st.one_of(
    st.tuples(
        st.just("set"),
        simulation_index,
        st.sampled_from(SETTABLE),
        st.sampled_from(YEARS),
        values,
    ),
    st.tuples(
        st.just("set"),
        simulation_index,
        st.sampled_from(["p_m", "p_month"]),
        st.sampled_from(MONTHS),
        values,
    ),
)
single_operation = st.one_of(
    calculate,
    calculate,
    set_input,
    set_input,
    st.tuples(st.just("branch"), simulation_index),
    st.tuples(st.just("drop"), simulation_index),
)
# Pairs of a calculated value and an input it depends on, each reaching it a
# different way: through a formula's own branch, uprating from an earlier
# year, a month of a year, a value the holder may not keep, a lagged year,
# and a year summed from months.
DEPENDENCIES = [
    ("p_inner", "2013", "p_inner_only", "2013"),
    ("p_prod", "2015", "p_up", "2014"),
    ("p_month", "2013-07", "p_m", "2013"),
    ("p_prod", "2013", "p_sum", "2013"),
    ("p_lag", "2014", "p_a", "2013"),
    ("p_cond", "2014", "p_b", "2014"),
    ("p_months", "2014", "p_m", "2014-12"),
]
# Calculate a value, branch, override an input it depends on in the new
# branch (index -1), and calculate the value again there; or branch first and
# calculate the value in the branch before and after overriding the input.
override_after_calculating = st.tuples(
    simulation_index, st.sampled_from(DEPENDENCIES), values, st.booleans()
).map(
    lambda drawn: (
        [
            ("calculate", drawn[0], drawn[1][0], drawn[1][1]),
            ("branch", drawn[0]),
            ("set", -1, drawn[1][2], drawn[1][3], drawn[2]),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
        ]
        if drawn[3]
        else [
            ("branch", drawn[0]),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
            ("set", -1, drawn[1][2], drawn[1][3], drawn[2]),
            ("calculate", -1, drawn[1][0], drawn[1][1]),
        ]
    )
)
operations = st.lists(
    st.one_of(
        single_operation.map(lambda operation: [operation]),
        override_after_calculating,
    ),
    min_size=1,
    max_size=15,
).map(lambda chunks: [operation for chunk in chunks for operation in chunk])


def _stored_periods(variable, period):
    """The periods ``set_input`` stores ``variable`` at, given ``period``."""
    period = periods.period(period)
    if SYNTHETIC_SYSTEM.get_variable(variable).definition_period == periods.MONTH:
        return [str(month) for month in period.get_subperiods(periods.MONTH)]
    return [str(period)]


def _own_inputs(simulation, variable, period):
    """The values stored on the branch itself for ``variable`` over ``period``."""
    holder = simulation.get_holder(variable)
    return {
        (variable, stored_period): tuple(
            holder._get_array_from_storage(stored_period, simulation.branch_name)
        )
        for stored_period in _stored_periods(variable, period)
    }


def _run(program, memory_config):
    """Run ``program``; return each calculation with the inputs it should reflect.

    A branch's inputs are those of the simulation it was created from, as
    they were then, and those set on it since, as stored (an input for a
    year is divided between the months the branch has not set itself).
    """
    root = _synthetic(ROOT_INPUTS, memory_config=memory_config)
    simulations = [root]
    inputs = [dict(ROOT_INPUTS)]
    own_inputs = [set()]
    results = []
    for operation in program:
        kind, index = operation[0], operation[1] % len(simulations)
        if operation[1] == -1:
            index = len(simulations) - 1  # The newest simulation.
        simulation = simulations[index]
        if kind == "branch":
            simulations.append(simulation.get_branch(f"b{len(simulations)}"))
            inputs.append(dict(inputs[index]))
            own_inputs.append(set())
        elif kind == "set":
            if index == 0:
                continue  # Inputs on the root after calculating are out of scope.
            _, _, variable, period, value = operation
            stored = {(variable, p) for p in _stored_periods(variable, period)}
            if len(stored) > 1 and stored <= own_inputs[index]:
                # Dividing a year between months the branch has all set
                # itself is an error unless the totals match.
                continue
            simulation.set_input(variable, period, np.asarray(value))
            stored = _own_inputs(simulation, variable, period)
            inputs[index].update(stored)
            own_inputs[index].update(stored)
        elif kind == "calculate":
            _, _, variable, period = operation
            value = np.array(simulation.calculate(variable, period), copy=True)
            results.append((dict(inputs[index]), variable, period, value))
        else:
            simulation.drop_computed_arrays()
    return results


@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(program=operations, mode=st.sampled_from(sorted(MEMORY_CONFIGS)))
def test_branch_calculations_match_a_simulation_given_its_inputs_first(program, mode):
    for branch_inputs, variable, period, value in _run(program, MEMORY_CONFIGS[mode]()):
        expected = _synthetic(branch_inputs).calculate(variable, period)
        # Uprating chains may round differently in float32 depending on which
        # earlier periods were calculated first.
        np.testing.assert_allclose(
            value, expected, rtol=1e-5, err_msg=f"{variable} {period}"
        )
