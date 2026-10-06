"""Auto-carry-over depends only on the inputs, not on what was calculated first.

With ``auto_carry_over_input_variables``, a variable with no formula result
for a period takes its value from another period. Before this fix,
``Simulation._calculate`` took the latest-starting stored period other than
a ``calculate_add``/``calculate_divide`` cache (#557), and returned the
default if that period started after the requested one. Two things made the
result depend on calculation order:

* Any later stored period blocked carry-over. A later input did (inputs for
  2012 and 2014 gave the default for 2013), and so did a value the simulation
  had itself calculated: once 2014 had been calculated, carrying the 2012
  input into it, 2013 came out as the default instead of the 2012 input.
* Values the simulation calculated were carried like inputs. A value masked by
  ``defined_for`` carried its mask into later periods, a default cached for a
  period where ``defined_for`` was false everywhere replaced the input, and a
  formula result carried past the formula's ``end``.

The rule now is the reference rule in ``tests/fixtures/carry_over.py``. Every
value the simulation caches is marked derived for the (branch, period) key it
is stored under, ``Holder.is_derived`` reports the mark of the value a branch
reads, and only values that are not derived carry. A derived value never
replaces an input a branch can read.

Each regression below checks a simulation that calculated other periods first
against a fresh simulation that calculates only the period in question; the
properties are in ``test_carry_over_order_property.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage, OnDiskStorage
from policyengine_core.experimental import MemoryConfig
from policyengine_core.reforms import Reform
from policyengine_core.tools.simulation_dumper import (
    dump_simulation,
    restore_simulation,
)
from tests.fixtures.carry_over import alone, build_system, simulation


@pytest.fixture(scope="module")
def system():
    return build_system()


def _after(system, inputs, requests, variable, period):
    built = simulation(system, inputs)
    for requested_variable, requested_period in requests:
        built.calculate(requested_variable, requested_period)
    return built.calculate(variable, period)


@pytest.mark.parametrize(
    "variable,earlier_input,later_period,between",
    [
        ("carried", "2012", "2014", "2013"),
        ("carried_count", "2012", "2014", "2013"),
        ("carried_monthly", "2012-01", "2012-06", "2012-03"),
    ],
)
def test_calculated_later_period_does_not_hide_earlier_input(
    system, variable, earlier_input, later_period, between
):
    inputs = {variable: {earlier_input: [7, 8]}}
    result = _after(system, inputs, [(variable, later_period)], variable, between)
    np.testing.assert_array_equal(result, [7, 8])
    np.testing.assert_array_equal(result, alone(system, inputs, variable, between))


@pytest.mark.parametrize(
    "variable,earlier,later,between",
    [
        ("carried", "2012", "2014", "2013"),
        ("carried_monthly", "2012-01", "2012-06", "2012-03"),
    ],
)
def test_later_input_does_not_hide_earlier_input(
    system, variable, earlier, later, between
):
    inputs = {variable: {earlier: [7, 8], later: [9, 10]}}
    np.testing.assert_array_equal(alone(system, inputs, variable, between), [7, 8])


def test_inputs_are_stored_in_any_order(system):
    inputs = {"carried": {"2014": [9, 10], "2012": [7, 8]}}
    for year, expected in [("2013", [7, 8]), ("2014", [9, 10]), ("2016", [9, 10])]:
        np.testing.assert_array_equal(alone(system, inputs, "carried", year), expected)


@pytest.mark.parametrize(
    "first,second", [("2013", "2013-01"), ("2013-01", "2013")], ids=["year", "month"]
)
def test_inputs_starting_together_carry_the_one_ending_last(system, first, second):
    values = {"2013": [1, 1], "2013-01": [2, 2]}
    inputs = {
        "year_input_without_helper": {first: values[first], second: values[second]}
    }
    np.testing.assert_array_equal(
        alone(system, inputs, "year_input_without_helper", "2014"), [1, 1]
    )


@pytest.mark.parametrize(
    "first,second", [("2013", "year:2013:2"), ("year:2013:2", "2013")]
)
def test_inputs_at_one_unit_starting_together_carry_the_one_ending_last(
    system, first, second
):
    values = {"2013": [1, 1], "year:2013:2": [2, 2]}
    inputs = {
        "year_input_without_helper": {first: values[first], second: values[second]}
    }
    np.testing.assert_array_equal(
        alone(system, inputs, "year_input_without_helper", "2015"), [2, 2]
    )


@pytest.mark.parametrize("reverse", [False, True], ids=["month-first", "day-first"])
def test_inputs_with_the_same_extent_carry_the_larger_unit(system, reverse):
    """Two inputs covering the same days in other units than the variable's
    resolve the same way whichever was stored first."""
    values = {"month:2013-01:2": [10, 10], "day:2013-01-01:59": [20, 20]}
    order = list(values)[::-1] if reverse else list(values)
    inputs = {"year_input_without_helper": {key: values[key] for key in order}}
    np.testing.assert_array_equal(
        alone(system, inputs, "year_input_without_helper", "2014"), [10, 10]
    )


@pytest.mark.parametrize(
    "stored,asked",
    [("day:9999-12-30:3", "9999-12-31"), ("day:2012-01-01:4000000", "2013-01-01")],
)
def test_an_input_ending_after_the_last_date_carries(system, stored, asked):
    """A period that ends after 9999-12-31 has no ``stop`` (it raises
    ``OverflowError``); its input carries all the same."""
    inputs = {"day_input_without_helper": {stored: [10, 20]}}
    np.testing.assert_array_equal(
        alone(system, inputs, "day_input_without_helper", asked), [10, 20]
    )


@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_an_input_ending_after_the_last_date_ends_last(system, longer_first):
    values = {"day:2012-01-01:4000000": [10, 20], "day:2012-01-01:2": [1, 2]}
    order = list(values) if longer_first else list(values)[::-1]
    inputs = {"day_input_without_helper": {key: values[key] for key in order}}
    np.testing.assert_array_equal(
        alone(system, inputs, "day_input_without_helper", "2013-01-01"), [10, 20]
    )


@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_of_two_inputs_ending_after_the_last_date_the_later_carries(
    system, longer_first
):
    """Both end after 9999-12-31. As strings, ``:10`` sorts before ``:9``."""
    values = {"day:9999-12-30:10": [10, 10], "day:9999-12-30:9": [9, 9]}
    order = list(values) if longer_first else list(values)[::-1]
    inputs = {"day_input_without_helper": {key: values[key] for key in order}}
    np.testing.assert_array_equal(
        alone(system, inputs, "day_input_without_helper", "9999-12-31"), [10, 10]
    )


@pytest.mark.parametrize("months_first", [True, False], ids=["months", "days"])
def test_inputs_in_two_units_ending_after_the_last_date_carry_the_later(
    system, months_first
):
    """800 days from 1 February 9999 end after 24 months from the same day.
    The larger unit and the string order both favour the months, so only
    the end picks the days."""
    values = {"day:9999-02-01:800": [8, 8], "month:9999-02:24": [24, 24]}
    order = list(values)[::-1] if months_first else list(values)
    inputs = {"year_input_without_helper": {key: values[key] for key in order}}
    np.testing.assert_array_equal(
        alone(system, inputs, "year_input_without_helper", "year:9999-02"), [8, 8]
    )


def test_later_input_does_not_carry_backwards(system):
    built = simulation(system, {"carried": {"2012": [7, 8], "2014": [9, 10]}})
    np.testing.assert_array_equal(built.calculate("carried", "2011"), [0, 0])
    np.testing.assert_array_equal(built.calculate("carried", "2015"), [9, 10])
    np.testing.assert_array_equal(built.calculate("carried", "2011"), [0, 0])


def test_value_masked_by_defined_for_does_not_carry_its_mask(system):
    inputs = {
        "carried_if_eligible": {"2012": [7, 8]},
        "eligible": {"2013": [False, True], "2014": [True, True]},
    }
    built = simulation(system, inputs)
    np.testing.assert_array_equal(
        built.calculate("carried_if_eligible", "2013"), [0, 8]
    )
    np.testing.assert_array_equal(
        built.calculate("carried_if_eligible", "2014"), [7, 8]
    )


def test_default_cached_where_defined_for_is_false_does_not_carry(system):
    inputs = {
        "carried_if_eligible": {"2012": [7, 8]},
        "eligible": {"2013": [False, False], "2014": [True, True]},
    }
    built = simulation(system, inputs)
    np.testing.assert_array_equal(
        built.calculate("carried_if_eligible", "2013"), [0, 0]
    )
    np.testing.assert_array_equal(
        built.calculate("carried_if_eligible", "2014"), [7, 8]
    )


@pytest.mark.parametrize("own_input", [None, "2012"])
def test_formula_result_does_not_carry_past_the_formula_end(system, own_input):
    inputs = {"carried": {"2012": [1, 2]}}
    if own_input is not None:
        inputs["formula_until_2013"] = {own_input: [40, 50]}
    expected = alone(system, inputs, "formula_until_2013", "2014")
    np.testing.assert_array_equal(expected, [0, 0] if own_input is None else [40, 50])
    result = _after(
        system, inputs, [("formula_until_2013", "2013")], "formula_until_2013", "2014"
    )
    np.testing.assert_array_equal(result, expected)


def test_input_set_after_a_period_defaulted_still_carries(system):
    built = simulation(system, {})
    np.testing.assert_array_equal(built.calculate("carried", "2013"), [0, 0])
    built.set_input("carried", "2012", np.array([7.0, 8.0]))
    np.testing.assert_array_equal(built.calculate("carried", "2014"), [7, 8])


def test_defaults_are_cached_as_before(system):
    """With nothing stored, or only earlier calculated values, the default is
    cached (as master cached the value it carried); before a later stored
    period it is not (as master returned it)."""
    built = simulation(system, {})
    holder = built.get_holder("carried_monthly")
    for month in range(1, 13):
        for _ in range(2):
            built.calculate("carried_monthly", f"2012-{month:02d}")
    for month in range(1, 13):
        assert holder.get_array(periods.period(f"2012-{month:02d}")) is not None
    built = simulation(system, {"carried_monthly": {"2013-06": [1, 2]}})
    holder = built.get_holder("carried_monthly")
    np.testing.assert_array_equal(built.calculate("carried_monthly", "2012-03"), [0, 0])
    assert holder.get_array(periods.period("2012-03")) is None


def test_branch_input_is_not_hidden_by_a_value_the_parent_carried(system):
    built = simulation(system, {"carried": {"2012": [7, 8]}})
    built.calculate("carried", "2013")
    branch = built.get_branch("reform")
    branch.set_input("carried", "2012", np.array([70.0, 80.0]))
    np.testing.assert_array_equal(branch.calculate("carried", "2014"), [70, 80])
    np.testing.assert_array_equal(built.calculate("carried", "2014"), [7, 8])


def test_derived_mark_belongs_to_the_branch_that_stored_the_value(system):
    """A value calculated on one branch does not make another branch's input
    for the same period derived."""

    def build():
        built = simulation(system, {"carried": {"2011": [7, 7]}})
        built.get_holder("formula_until_2013").set_input(
            periods.period("2012"), np.array([30.0, 30.0]), "other"
        )
        return built

    expected = build().get_branch("other").calculate("formula_until_2013", "2014")
    np.testing.assert_array_equal(expected, [30, 30])
    built = build()
    np.testing.assert_array_equal(
        built.calculate("formula_until_2013", "2012"), [12, 12]
    )
    other = built.get_branch("other")
    holder = other.get_holder("formula_until_2013")
    assert holder.is_derived(periods.period("2012"), "default")
    assert not holder.is_derived(periods.period("2012"), "other")
    np.testing.assert_array_equal(
        other.calculate("formula_until_2013", "2014"), expected
    )


def test_add_over_a_calculated_period_keeps_it_derived(system):
    inputs = {
        "carried_if_eligible": {"2012": [7, 8]},
        "eligible": {"2013": [False, True], "2014": [True, True]},
    }
    built = simulation(system, inputs)
    np.testing.assert_array_equal(
        built.calculate_add("carried_if_eligible", "2013"), [0, 8]
    )
    np.testing.assert_array_equal(
        built.calculate("carried_if_eligible", "2014"), [7, 8]
    )


def test_add_over_an_input_period_keeps_it_an_input(system):
    built = simulation(system, {"carried": {"2012": [7, 8]}})
    np.testing.assert_array_equal(built.calculate_add("carried", "2012"), [7, 8])
    np.testing.assert_array_equal(built.calculate("carried", "2013"), [7, 8])


@pytest.mark.parametrize(
    "variable,input_period,kind,request_period,later_period",
    [
        # The sum over the two years would replace the two-year input.
        ("year_input_without_helper", "year:2012:2", "add", "year:2012:2", "2014"),
        # The twelfth of 2012 would replace the January input.
        ("year_input_without_helper", "2012-01", "divide", "2012-01", "2014"),
        # The sum of the months would replace the year input.
        ("month_input_without_helper", "2013", "add", "2013", "2014-01"),
    ],
)
@pytest.mark.parametrize("on_branch", [False, True], ids=["simulation", "branch"])
def test_add_or_divide_never_replaces_an_input(
    system, variable, input_period, kind, request_period, later_period, on_branch
):
    inputs = {variable: {input_period: [120, 24]}}
    expected = alone(system, inputs, variable, later_period)
    np.testing.assert_array_equal(expected, [120, 24])
    built = simulation(system, inputs)
    if on_branch:
        built = built.get_branch("reform")
    if kind == "add":
        built.calculate_add(variable, request_period)
    else:
        built.calculate_divide(variable, request_period)
    holder = built.get_holder(variable)
    np.testing.assert_array_equal(
        holder.get_array(periods.period(input_period), built.branch_name), [120, 24]
    )
    np.testing.assert_array_equal(built.calculate(variable, later_period), expected)


def test_add_over_an_anchored_year_keeps_the_anchored_input(system):
    """``set_input`` stores a month of a yearly flow at the year starting
    then; ``ADD`` over that year is a single calendar-year sub-period."""
    inputs = {"carried": {"2012": [1, 2], "2013-06": [5, 6]}}
    expected = alone(system, inputs, "carried", "2014")
    np.testing.assert_array_equal(expected, [5, 6])
    built = simulation(system, inputs)
    built.calculate_add("carried", "year:2013-06")
    np.testing.assert_array_equal(built.calculate("carried", "2014"), expected)


def test_month_cache_of_a_year_input_does_not_carry_into_the_next_year(system):
    """policyengine-us computes ``monthly_age`` from ``age`` one month at a
    time; each month of a yearly flow caches a twelfth of the year (#557)."""
    inputs = {"carried": {"2024": [40, 6]}}
    built = simulation(system, inputs)
    for month in range(1, 13):
        built.calculate("carried", f"2024-{month:02d}")
    np.testing.assert_allclose(built.calculate("carried", "2024-12"), [40 / 12, 6 / 12])
    np.testing.assert_array_equal(built.calculate("carried", "2025"), [40, 6])


def test_own_unit_input_wins_over_an_input_at_another_unit(system):
    """As before 3.24.0, and as in #557: the year input, not the later month
    input of a variable with no ``set_input`` helper."""
    inputs = {"year_input_without_helper": {"2024": [120, 12], "2024-12": [9, 9]}}
    np.testing.assert_array_equal(
        alone(system, inputs, "year_input_without_helper", "2025"), [120, 12]
    )


# Lifecycle paths that write or drop stored values outside ``Holder._set``
# (the cases #557's review found to leave per-period marks stale).

FLOW = "year_input_without_helper"


def test_reform_replay_restores_an_input_at_another_unit(system):
    built = simulation(system, {FLOW: {"2024-06": [600, 60]}})
    branch = built.get_branch("child")
    branch.calculate_divide(FLOW, "2024-06")
    branch.apply_reform(_noop)
    np.testing.assert_array_equal(branch.calculate(FLOW, "2025"), [600, 60])


def test_dump_and_restore_keep_carrying_the_input(system, tmp_path):
    built = simulation(system, {FLOW: {"2024": [120, 12]}})
    built.calculate_divide(FLOW, "2024-06")
    dump_simulation(built, str(tmp_path / "dump"))
    restored = restore_simulation(str(tmp_path / "dump"), system)
    np.testing.assert_array_equal(built.calculate(FLOW, "2025"), [120, 12])
    np.testing.assert_array_equal(restored.calculate(FLOW, "2025"), [120, 12])


def test_deleting_a_branch_cache_exposes_the_ancestor_input(system):
    built = simulation(system, {FLOW: {"2024-06": [600, 60]}})
    branch = built.get_branch("child")
    branch.calculate_divide(FLOW, "2024-06")
    branch.get_holder(FLOW).delete_arrays(periods.period("2024-06"), "child")
    np.testing.assert_array_equal(branch.calculate(FLOW, "2025"), [600, 60])
    np.testing.assert_array_equal(built.calculate(FLOW, "2025"), [600, 60])


def test_deleting_a_branch_input_does_not_expose_the_ancestor_twelfth(system):
    built = simulation(system, {FLOW: {"2024": [120, 12]}})
    built.calculate_divide(FLOW, "2024-06")
    branch = built.get_branch("child")
    branch.set_input(FLOW, "2024-06", np.array([600.0, 60.0]))
    branch.get_holder(FLOW).delete_arrays(periods.period("2024-06"), "child")
    np.testing.assert_array_equal(branch.calculate(FLOW, "2025"), [120, 12])
    np.testing.assert_array_equal(built.calculate(FLOW, "2025"), [120, 12])


@pytest.mark.parametrize("delete_first", [True, False], ids=["deleted", "replaced"])
def test_a_value_written_straight_to_storage_is_not_derived(system, delete_first):
    """The mark is stored with the value: a value written later at the same
    key, even straight to the storage as ``apply_reform``'s input replay
    does, is not taken for a derived one."""
    built = simulation(system, {"carried": {"2012": [7, 8]}})
    built.calculate("carried", "2013")
    holder = built.get_holder("carried")
    assert holder.is_derived(periods.period("2013"))
    if delete_first:
        holder.delete_arrays(periods.period("2013"))
        assert not holder.is_derived(periods.period("2013"))
    holder._memory_storage.put(np.array([5.0, 5.0]), periods.period("2013"))
    assert not holder.is_derived(periods.period("2013"))
    np.testing.assert_array_equal(built.calculate("carried", "2014"), [5, 5])


def test_carry_over_in_a_branch_copies_only_the_array_it_reads(system):
    """Deciding which periods are inputs reads no shared array (#556): the
    branch copies only the input it carries."""
    built = simulation(
        system, {"carried": {"2012": [1, 1], "2014": [2, 2], "2016": [3, 3]}}
    )
    built.calculate("carried", "2017")
    branch = built.get_branch("reform")
    storage = branch.get_holder("carried")._memory_storage
    shared = len(storage._shared)
    np.testing.assert_array_equal(branch.calculate("carried", "2020"), [3, 3])
    assert len(storage._shared) == shared - 1


def _marks(storage):
    """A storage's own record of which values are which: an in-memory
    storage records its inputs, an on-disk one its derived values."""
    return storage._derived if isinstance(storage, OnDiskStorage) else storage._inputs


@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
def test_storages_keep_the_derived_mark_with_the_value(on_disk, tmp_path):
    storage = (
        OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
        if on_disk
        else InMemoryStorage(is_eternal=False)
    )
    year = periods.period("2013")
    value = np.array([1.0, 2.0])
    assert not storage.has(year) and not storage.is_derived(year)
    storage.put(value, year, derived=True)
    assert storage.has(year) and storage.is_derived(year)
    assert not storage.has(year, "reform") and not storage.is_derived(year, "reform")
    clone = storage.clone()
    assert clone.is_derived(year)
    storage.put(value, year)
    assert storage.has(year) and not storage.is_derived(year)
    assert clone.is_derived(year)
    storage.put(value, year, derived=True)
    storage.delete(year)
    assert not storage.has(year) and not storage.is_derived(year)
    assert not _marks(storage)
    storage.put(value, year, derived=True)
    storage.put(value, year, "reform", derived=True)
    storage.delete(branch_name="reform")
    assert storage.is_derived(year) and not storage.has(year, "reform")
    storage.delete()
    assert not _marks(storage)
    storage.put(value, year)
    assert not storage.is_derived(year)


def test_carry_over_reads_derived_marks_from_disk_storage(system):
    built = simulation(system, {})
    built.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = built.get_holder("carried")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    built.set_input("carried", "2012", np.array([7.0, 8.0]))
    np.testing.assert_array_equal(built.calculate("carried", "2014"), [7, 8])
    assert holder._disk_storage.is_derived(periods.period("2014"))
    np.testing.assert_array_equal(built.calculate("carried", "2013"), [7, 8])
    holder.delete_arrays(periods.period("2014"))
    built.set_input("carried", "2014", np.array([9.0, 10.0]))
    assert not holder.is_derived(periods.period("2014"))
    np.testing.assert_array_equal(built.calculate("carried", "2015"), [9, 10])


def test_a_default_cached_before_an_input_does_not_become_an_uprating_base(system):
    """With no input at or before 2012 the default is not cached for an
    uprated variable, so 2014 still carries the 2013-01 input."""
    inputs = {"uprated_input_without_helper": {"2013-01": [10, 20]}}
    expected = alone(system, inputs, "uprated_input_without_helper", "2014")
    np.testing.assert_array_equal(expected, [10, 20])
    built = simulation(system, inputs)
    np.testing.assert_array_equal(
        built.calculate("uprated_input_without_helper", "2012"), [0, 0]
    )
    np.testing.assert_array_equal(
        built.calculate("uprated_input_without_helper", "2014"), expected
    )


def test_a_value_calculated_while_an_input_is_set_stays_on_its_own_branch(system):
    """A ``set_input`` helper that calculates writes the calculated value to
    the simulation's own branch, as a derived value, not to the input's."""
    built = simulation(system, {"carried": {"2012": [1, 2]}})
    holder = built.get_holder("formula_until_2013")
    holder.set_input(periods.period("2012"), np.array([20.0, 20.0]), "other")
    built.get_holder("input_with_calculating_helper").set_input(
        periods.period("2013-06"), np.array([1.0, 1.0]), "other"
    )
    np.testing.assert_array_equal(
        holder.get_array(periods.period("2012"), "other"), [20, 20]
    )
    np.testing.assert_array_equal(
        holder.get_array(periods.period("2012"), "default"), [6, 7]
    )
    assert holder.is_derived(periods.period("2012"), "default")
    assert not holder.is_derived(periods.period("2012"), "other")
    assert ("formula_until_2013", "default", periods.period("2012")) not in (
        built._user_input_keys
    )
    other = built.get_branch("other")
    np.testing.assert_array_equal(
        other.calculate("formula_until_2013", "2014"), [20, 20]
    )


def test_carry_over_reads_provenance_in_one_pass(system, monkeypatch):
    """With many calculated periods and deep branches, a decision makes one
    pass over the stored keys instead of walking the branches per period."""
    from policyengine_core.holders import Holder

    built = simulation(system, {"carried": {"1900": [1, 1]}})
    for year in range(1901, 2010):
        built.calculate("carried", str(year))
    branch = built
    for depth in range(20):
        branch = branch.get_branch(f"b{depth}")
    calls = {"stores": 0, "passes": 0}
    stores, passes = Holder._stores, Holder.get_input_periods

    def counting_stores(self, *args, **kwargs):
        calls["stores"] += 1
        return stores(self, *args, **kwargs)

    def counting_passes(self, *args, **kwargs):
        calls["passes"] += 1
        return passes(self, *args, **kwargs)

    monkeypatch.setattr(Holder, "_stores", counting_stores)
    monkeypatch.setattr(Holder, "get_input_periods", counting_passes)
    np.testing.assert_array_equal(branch.calculate("carried", "2020"), [1, 1])
    # One pass for the decision; caching the result checks the period on each
    # branch in the chain once (21), not every stored period on each (2,310).
    assert calls["passes"] == 1
    assert calls["stores"] <= 21


def test_a_period_stored_only_on_another_branch_does_not_carry(system):
    """A branch never takes a period it cannot read for an input."""
    built = simulation(system, {"year_input_without_helper": {"2013-01": [20, 20]}})
    built.get_holder("year_input_without_helper").set_input(
        periods.period("2012"), np.array([10.0, 10.0]), "other"
    )
    other = built.get_branch("other")
    np.testing.assert_array_equal(
        built.calculate("year_input_without_helper", "2014"), [20, 20]
    )
    np.testing.assert_array_equal(
        other.calculate("year_input_without_helper", "2014"), [10, 10]
    )


def test_marks_are_cleared_with_the_values_apply_reform_wipes(system):
    built = simulation(system, {})
    built.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = built.get_holder("carried")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    for year in range(2010, 2020):
        built.calculate("carried", str(year))
        built.apply_reform(_noop)
    assert not holder._memory_storage._inputs
    assert not holder._disk_storage._derived
    # A file written for a period whose value was derived, then read back
    # by rebuilding the index, is an input.
    built = simulation(system, {})
    built.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = built.get_holder("carried")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    writer = OnDiskStorage(holder._disk_storage.storage_dir, preserve_storage_dir=True)
    built.calculate("carried", "2012")
    assert holder.is_derived(periods.period("2012"))
    built.apply_reform(_noop)
    writer.put(np.array([10.0, 10.0]), periods.period("2012"))
    holder._disk_storage.restore()
    assert not holder.is_derived(periods.period("2012"))
    np.testing.assert_array_equal(built.calculate("carried", "2013"), [10, 10])


def test_marks_are_cleared_with_in_memory_values_apply_reform_wipes(system):
    built = simulation(system, {})
    holder = built.get_holder("carried")
    for year in range(2010, 2020):
        built.calculate("carried", str(year))
        built.apply_reform(_noop)
    assert not holder._memory_storage._inputs


def test_rebuilding_a_disk_index_reads_files_as_inputs(system):
    """A derived file replaced by an input, then re-indexed, is an input."""
    built = simulation(system, {})
    built.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = built.get_holder("carried")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    built.calculate("carried", "2012")
    assert holder.is_derived(periods.period("2012"))
    writer = OnDiskStorage(holder._disk_storage.storage_dir, preserve_storage_dir=True)
    writer.put(np.array([10.0, 10.0]), periods.period("2012"))
    holder._disk_storage.restore()
    assert not holder.is_derived(periods.period("2012"))
    np.testing.assert_array_equal(built.calculate("carried", "2013"), [10, 10])


def test_a_value_in_memory_takes_precedence_over_one_on_disk(system):
    """``get_array`` reads memory before disk for the same key, and so does
    the input test."""
    built = simulation(system, {})
    holder = built.get_holder("carried")
    holder._disk_storage = holder.create_disk_storage()
    year = periods.period("2012")
    holder._disk_storage.put(np.array([0.0, 0.0]), year, derived=True)
    holder._memory_storage.put(np.array([7.0, 8.0]), year)
    assert year in holder.get_input_periods()
    np.testing.assert_array_equal(built.calculate("carried", "2013"), [7, 8])


@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
def test_storages_pickled_without_marks_still_work(on_disk, tmp_path):
    import pickle

    storage = (
        OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
        if on_disk
        else InMemoryStorage(is_eternal=False)
    )
    year = periods.period("2012")
    storage.put(np.array([10.0]), year)
    # The state an older version pickled: its arrays or files, and no marks.
    state = dict(storage.__dict__)
    for name in ("_derived", "_inputs", "_shared"):
        state.pop(name, None)
    old = type(storage).__new__(type(storage))
    old.__setstate__(state)
    restored = pickle.loads(pickle.dumps(old))
    assert restored.has(year) and not restored.is_derived(year)
    restored.clone()
    restored.put(np.array([20.0]), periods.period("2013"), derived=True)
    assert restored.is_derived(periods.period("2013"))
    restored.delete(periods.period("2013"))
    np.testing.assert_array_equal(restored.get(year), [10.0])


def test_input_starting_with_the_requested_period_carries(system):
    """An input in another unit that starts when the period does carries."""
    built = simulation(system, {"year_input_without_helper": {"2013-01": [3, 4]}})
    np.testing.assert_array_equal(
        built.calculate("year_input_without_helper", "2013"), [3, 4]
    )


def test_carried_value_is_derived_and_the_input_is_not(system):
    built = simulation(system, {"carried": {"2012": [7, 8]}})
    built.calculate("carried", "2013")
    holder = built.get_holder("carried")
    assert holder.is_derived(periods.period("2013"))
    assert not holder.is_derived(periods.period("2012"))
    assert not holder.is_derived(periods.period("2015"))
    branch = built.get_branch("reform")
    assert branch.get_holder("carried").is_derived(periods.period("2013"), "reform")


def test_restored_simulation_keeps_derived_marks(system, tmp_path):
    inputs = {"carried": {"2012": [1, 2]}}
    built = simulation(system, inputs)
    np.testing.assert_array_equal(built.calculate("formula_until_2013", "2012"), [6, 7])
    dump_simulation(built, str(tmp_path / "dump"))
    restored = restore_simulation(str(tmp_path / "dump"), system)
    expected = alone(system, inputs, "formula_until_2013", "2014")
    np.testing.assert_array_equal(expected, [0, 0])
    np.testing.assert_array_equal(
        built.calculate("formula_until_2013", "2014"), expected
    )
    np.testing.assert_array_equal(
        restored.calculate("formula_until_2013", "2014"), expected
    )
    np.testing.assert_array_equal(restored.calculate("carried", "2014"), [1, 2])


class _noop(Reform):
    def apply(self):
        pass


def test_inputs_still_carry_after_apply_reform(system):
    inputs = {"carried": {"2012": [7, 8]}}
    built = simulation(system, inputs)
    built.calculate("carried", "2014")
    built.apply_reform(_noop)
    holder = built.get_holder("carried")
    assert not any(holder.is_derived(p) for p in holder.get_known_periods())
    np.testing.assert_array_equal(built.calculate("carried", "2013"), [7, 8])
    np.testing.assert_array_equal(built.calculate("carried", "2014"), [7, 8])
