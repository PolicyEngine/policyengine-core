"""Writing to or deleting from a holder drops the fast-cache entries it changes.

``Simulation.calculate`` answers a repeated request from ``_fast_cache``
before it reads the holder. ``Simulation.set_input`` and
``Simulation.delete_arrays`` dropped the matching entry, but the holder's own
methods did not: after ``holder.set_input(period, [300, 400])`` the holder
held ``[300, 400]`` while ``calculate`` still returned the value it had
calculated before.

Every holder write (``set_input``, ``put_in_cache``, a ``set_input`` helper's
stores) and delete now drops the entries it makes stale, in the holder's own
simulation only. The property test is
``test_holder_write_fast_cache_property.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.experimental import MemoryConfig
from tests.fixtures.uprated_inputs import build_simulation, build_system


@pytest.fixture(scope="module")
def system():
    return build_system()


def _fast_cached(simulation, variable):
    return sorted(
        str(period) for name, period in simulation._fast_cache if name == variable
    )


def test_holder_set_input_replaces_a_calculated_value(system):
    # The reported case: master kept returning the uprated [1038, 79].
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    assert simulation.calculate("uprated_count", "2013").tolist() == [1038, 79]
    holder = simulation.get_holder("uprated_count")

    holder.set_input(periods.period("2013"), [300, 400])

    assert holder.get_array(periods.period("2013")).tolist() == [300, 400]
    assert simulation.calculate("uprated_count", "2013").tolist() == [300, 400]


def test_holder_set_input_split_into_months_replaces_each_month(system):
    simulation = build_simulation(system)
    simulation.calculate("monthly_doubled", "2013-02")
    assert _fast_cached(simulation, "monthly_doubled") == ["2013-02"]
    holder = simulation.get_holder("monthly_doubled")

    # The helper stores twelve months; February is one of them.
    holder.delete_arrays(periods.period("2013"))
    holder.set_input(periods.period("2013"), [1200, 2400])

    assert simulation.calculate("monthly_doubled", "2013-02").tolist() == [100, 200]


def test_holder_set_input_of_an_eternity_variable_replaces_every_period(system):
    simulation = build_simulation(system)
    # One stored value answers every period; each request is cached apart.
    assert simulation.calculate("eternal_code_plus_one", "2012").tolist() == [1, 1]
    assert _fast_cached(simulation, "eternal_code_plus_one") == ["2012"]

    simulation.get_holder("eternal_code_plus_one").set_input(
        periods.period("2015"), [8, 9]
    )

    assert simulation.calculate("eternal_code_plus_one", "2012").tolist() == [8, 9]
    assert simulation.calculate("eternal_code_plus_one", "2015").tolist() == [8, 9]


def test_holder_put_in_cache_replaces_a_calculated_value(system):
    simulation = build_simulation(system, [("uprated_amount", "2012", [100.0, 50.0])])
    simulation.calculate("doubled_amount", "2012")

    simulation.get_holder("doubled_amount").put_in_cache(
        np.array([7.0, 8.0], dtype=np.float32), periods.period("2012")
    )

    assert simulation.calculate("doubled_amount", "2012").tolist() == [7.0, 8.0]


def test_holder_delete_arrays_drops_the_periods_it_deletes(system):
    simulation = build_simulation(system, [("monthly_amount", "2013", [1200, 2400])])
    simulation.calculate("monthly_doubled", "2013-02")
    simulation.calculate("monthly_doubled", "2014-06")
    simulation.calculate("doubled_amount", "2013")

    # Deleting a year deletes the months in it, and nothing else.
    simulation.get_holder("monthly_doubled").delete_arrays(periods.period("2013"))

    assert _fast_cached(simulation, "monthly_doubled") == ["2014-06"]
    assert _fast_cached(simulation, "doubled_amount") == ["2013"]
    simulation.get_holder("monthly_amount").delete_arrays(periods.period("2013"))
    simulation.set_input("monthly_amount", "2013", [2400, 4800])
    assert simulation.calculate("monthly_doubled", "2013-02").tolist() == [400, 800]


def test_holder_delete_arrays_without_a_period_drops_every_period(system):
    simulation = build_simulation(system, [("monthly_amount", "2013", [1200, 2400])])
    simulation.calculate("monthly_doubled", "2013-02")
    simulation.calculate("monthly_doubled", "2014-06")

    simulation.get_holder("monthly_doubled").delete_arrays()

    assert _fast_cached(simulation, "monthly_doubled") == []


def test_write_under_a_branch_the_simulation_does_not_read_keeps_the_entry(system):
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    calculated = simulation.calculate("uprated_count", "2013")

    simulation.get_holder("uprated_count").set_input(
        periods.period("2013"), [300, 400], "sibling"
    )

    # Still answered from the fast cache: the same array object.
    assert simulation.calculate("uprated_count", "2013") is calculated
    assert calculated.tolist() == [1038, 79]


def test_write_under_default_replaces_what_a_branch_read_from_it(system):
    simulation = build_simulation(system)
    branch = simulation.get_branch("reform")
    # No value under the branch's own name, so it reads (and caches) the
    # default; a later store under ``default`` in its holder replaces that.
    assert branch.calculate("doubled_amount", "2013").tolist() == [0, 0]
    holder = branch.get_holder("doubled_amount")
    holder.delete_arrays(periods.period("2013"), "reform")

    holder.put_in_cache(
        np.array([7.0, 8.0], dtype=np.float32), periods.period("2013"), "default"
    )

    assert branch.calculate("doubled_amount", "2013").tolist() == [7.0, 8.0]


def test_a_branch_write_leaves_the_parent_and_the_parent_write_leaves_the_branch(
    system,
):
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    parent_value = simulation.calculate("uprated_count", "2013")
    branch = simulation.get_branch("reform")

    # A branch has its own holders and its own fast cache.
    branch.get_holder("uprated_count").set_input(
        periods.period("2013"), [300, 400], "reform"
    )
    assert simulation.calculate("uprated_count", "2013") is parent_value
    assert branch.calculate("uprated_count", "2013").tolist() == [300, 400]

    branch_value = branch.calculate("uprated_count", "2014")
    simulation.get_holder("uprated_count").set_input(periods.period("2014"), [5, 6])
    assert simulation.calculate("uprated_count", "2014").tolist() == [5, 6]
    assert branch.calculate("uprated_count", "2014") is branch_value
    assert branch_value.tolist() == [311, 414]


def test_storing_a_calculated_value_keeps_it_in_the_fast_cache(system):
    # ``calculate`` stores its result in the holder and then caches it: the
    # store must not drop the entry the same request is about to rely on.
    simulation = build_simulation(system, [("uprated_amount", "2012", [100.0, 50.0])])

    first = simulation.calculate("doubled_amount", "2012")

    assert _fast_cached(simulation, "doubled_amount") == ["2012"]
    assert simulation.calculate("doubled_amount", "2012") is first


def test_write_under_an_ancestor_branch_name_replaces_what_a_nested_branch_read():
    # A nested branch reads its own key, then each ancestor's, then the
    # default. With the variable kept out of storage (cache blacklist), the
    # nested branch's fast cache is all that holds its calculated value, so
    # a write under the parent branch's name must drop it.
    system = build_system()
    system.cache_blacklist = ["doubled_amount"]
    simulation = build_simulation(system)
    simulation.opt_out_cache = True
    nested = simulation.get_branch("a").get_branch("b")
    assert nested.calculate("doubled_amount", "2013").tolist() == [0, 0]
    assert _fast_cached(nested, "doubled_amount") == ["2013"]

    nested.get_holder("doubled_amount").set_input(
        periods.period("2013"), [7.0, 8.0], "a"
    )

    assert nested.calculate("doubled_amount", "2013").tolist() == [7.0, 8.0]


def test_a_write_to_disk_storage_replaces_a_calculated_value(system):
    simulation = build_simulation(system)
    with pytest.warns(Warning):
        simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = simulation.get_holder("doubled_amount")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    assert simulation.calculate("doubled_amount", "2013").tolist() == [0, 0]
    assert holder._memory_storage.get(periods.period("2013")) is None

    holder.put_in_cache(np.array([7.0, 8.0], dtype=np.float32), periods.period("2013"))

    assert holder._disk_storage.get(periods.period("2013")).tolist() == [7.0, 8.0]
    assert simulation.calculate("doubled_amount", "2013").tolist() == [7.0, 8.0]


def test_a_write_replaces_an_eternity_value_requested_without_a_period(system):
    # ``calculate`` with no period caches the result under ``None``.
    simulation = build_simulation(system)
    assert simulation.calculate("eternal_code").tolist() == [0, 0]
    assert ("eternal_code", None) in simulation._fast_cache

    simulation.get_holder("eternal_code").set_input(periods.period("2015"), [8, 9])

    assert simulation.calculate("eternal_code").tolist() == [8, 9]
