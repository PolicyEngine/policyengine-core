import shutil
import tempfile

import numpy as np
import pytest
from numpy import testing

from policyengine_core.country_template import situation_examples
from policyengine_core.entities import build_entity
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.tools import simulation_dumper


def test_dump(tax_benefit_system):
    directory = tempfile.mkdtemp(prefix="openfisca_")
    simulation = SimulationBuilder().build_from_entities(
        tax_benefit_system, situation_examples.couple
    )
    calculated_value = simulation.calculate("disposable_income", "2018-01")
    simulation_dumper.dump_simulation(simulation, directory)

    simulation_2 = simulation_dumper.restore_simulation(directory, tax_benefit_system)

    # Check entities structure have been restored

    testing.assert_array_equal(simulation.person.ids, simulation_2.person.ids)
    testing.assert_array_equal(simulation.person.count, simulation_2.person.count)
    testing.assert_array_equal(simulation.household.ids, simulation_2.household.ids)
    testing.assert_array_equal(simulation.household.count, simulation_2.household.count)
    testing.assert_array_equal(
        simulation.household.members_position,
        simulation_2.household.members_position,
    )
    testing.assert_array_equal(
        simulation.household.members_entity_id,
        simulation_2.household.members_entity_id,
    )
    testing.assert_array_equal(
        simulation.household.members_role, simulation_2.household.members_role
    )

    # Check calculated values are in cache

    disposable_income_holder = simulation_2.person.get_holder("disposable_income")
    cached_value = disposable_income_holder.get_array("2018-01")
    assert cached_value is not None
    testing.assert_array_equal(cached_value, calculated_value)

    shutil.rmtree(directory)


# A population's count is not always given by its ids and memberships: a
# group with no members holds no membership, and a simulation with no persons
# holds none at all. ``restore_simulation`` used to take a group's count as
# one past its highest member index, so it raised on ``max()`` of no members
# and gave too small a count when the last groups had none.

EMPTY_GROUPS = {
    "persons": {
        "a": {"salary": {"2018-01": 1000}},
        "b": {"age": {"2018-01": 10}},
        "c": {},
    },
    # Households with no members first, in between and last.
    "households": {
        "h0": {},
        "h1": {"parents": ["a"], "children": ["b"], "rent": {"2018-01": 500}},
        "h2": {},
        "h3": {"parents": ["c"]},
        "h4": {"rent": {"2018-01": 700}},
    },
}


def _round_trip(simulation, tax_benefit_system, directory, *, without_counts=False):
    simulation_dumper.dump_simulation(simulation, str(directory))
    if without_counts:
        # As dumps written before counts were recorded.
        for path in directory.glob(f"__entities__/*/{simulation_dumper.COUNT_FILE}"):
            path.unlink()
    return simulation_dumper.restore_simulation(str(directory), tax_benefit_system)


def _assert_same_structure(simulation, restored):
    for key, population in simulation.populations.items():
        restored_population = restored.populations[key]
        assert restored_population.count == population.count, key
        testing.assert_array_equal(restored_population.ids, population.ids)
        if population.entity.is_person:
            continue
        for name in ("members_entity_id", "members_position", "members_role"):
            original = np.asarray(getattr(population, name))
            restored_array = np.asarray(getattr(restored_population, name))
            testing.assert_array_equal(restored_array, original, err_msg=name)
            if original.dtype.kind in "iu":
                assert restored_array.dtype.kind in "iu", name


def test_dump_restore_with_no_persons(tax_benefit_system, tmp_path):
    simulation = SimulationBuilder().build_default_simulation(
        tax_benefit_system, count=0
    )
    simulation.set_input("salary", "2018-01", np.array([]))
    simulation.set_input("rent", "2018-01", np.array([]))
    calculated = simulation.calculate("disposable_income", "2018-01")

    restored = _round_trip(simulation, tax_benefit_system, tmp_path / "dump")

    _assert_same_structure(simulation, restored)
    assert [population.count for population in restored.populations.values()] == [
        0,
        0,
    ]
    holder = restored.person.get_holder("disposable_income")
    testing.assert_array_equal(holder.get_array("2018-01"), calculated)
    assert holder.is_derived("2018-01")
    assert restored.calculate("household_income", "2018-01").tolist() == []
    assert restored.household.max(np.array([])).tolist() == []


def test_dump_restore_with_empty_groups_among_populated_ones(
    tax_benefit_system, tmp_path
):
    simulation = SimulationBuilder().build_from_entities(
        tax_benefit_system, EMPTY_GROUPS
    )
    household_income = simulation.calculate("household_income", "2018-01")
    assert simulation.household.count == 5
    assert simulation.household.members_entity_id.tolist() == [1, 1, 3]

    restored = _round_trip(simulation, tax_benefit_system, tmp_path / "dump")

    _assert_same_structure(simulation, restored)
    testing.assert_array_equal(
        restored.household.get_holder("household_income").get_array("2018-01"),
        household_income,
    )
    testing.assert_array_equal(
        restored.calculate("rent", "2018-01"), [0, 500, 0, 0, 700]
    )
    testing.assert_array_equal(
        restored.household.nb_persons(), simulation.household.nb_persons()
    )


def test_restore_infers_the_counts_of_a_dump_without_them(tax_benefit_system, tmp_path):
    # Every simulation built by ``SimulationBuilder`` has one id per instance.
    simulation = SimulationBuilder().build_from_entities(
        tax_benefit_system, EMPTY_GROUPS
    )
    simulation.calculate("household_income", "2018-01")

    restored = _round_trip(
        simulation, tax_benefit_system, tmp_path / "groups", without_counts=True
    )

    _assert_same_structure(simulation, restored)
    testing.assert_array_equal(
        restored.calculate("household_income", "2018-01"),
        simulation.calculate("household_income", "2018-01"),
    )

    no_one = SimulationBuilder().build_default_simulation(tax_benefit_system, count=0)
    restored = _round_trip(
        no_one, tax_benefit_system, tmp_path / "no_one", without_counts=True
    )
    _assert_same_structure(no_one, restored)


def test_restore_infers_counts_from_memberships_without_group_ids(
    tax_benefit_system, tmp_path
):
    # As ``scripts/simulation_generator.py`` builds one: counts and
    # memberships, but no group ids. Each group has members.
    simulation = Simulation(
        tax_benefit_system, tax_benefit_system.instantiate_entities()
    )
    simulation.persons.ids = np.arange(4)
    simulation.persons.count = 4
    simulation.household.count = 3
    simulation.household.members_entity_id = np.array([0, 1, 1, 2])
    simulation.set_input("rent", "2018-01", np.array([1.0, 2.0, 3.0]))
    assert len(simulation.household.ids) == 0

    restored = _round_trip(
        simulation, tax_benefit_system, tmp_path / "dump", without_counts=True
    )

    assert restored.person.count == 4
    assert restored.household.count == 3
    testing.assert_array_equal(
        restored.household.members_entity_id, simulation.household.members_entity_id
    )
    testing.assert_array_equal(restored.calculate("rent", "2018-01"), [1, 2, 3])


def test_dump_keeps_a_count_its_ids_and_memberships_do_not_give(
    tax_benefit_system, tmp_path
):
    # No group ids, and the last household has no members: only the dumped
    # count says there are three households.
    simulation = Simulation(
        tax_benefit_system, tax_benefit_system.instantiate_entities()
    )
    simulation.persons.ids = np.arange(2)
    simulation.persons.count = 2
    simulation.household.count = 3
    simulation.household.members_entity_id = np.array([0, 1])
    simulation.set_input("rent", "2018-01", np.array([1.0, 2.0, 3.0]))

    restored = _round_trip(simulation, tax_benefit_system, tmp_path / "dump")

    assert restored.household.count == 3
    testing.assert_array_equal(restored.calculate("rent", "2018-01"), [1, 2, 3])


class _IntegerLike:
    """An integer only through ``__index__``, as ``range`` takes it."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


@pytest.mark.parametrize(
    "count",
    [np.uint8(2), np.array(2), np.ma.array(2, dtype=np.int64), _IntegerLike(2)],
    ids=["uint8", "0-d array", "masked array", "__index__"],
)
def test_dump_writes_any_integer_count_as_int64(tax_benefit_system, tmp_path, count):
    # ``np.save`` of the count itself failed for a masked array and wrote an
    # object array, which restoring refuses, for an ``__index__`` object.
    simulation = SimulationBuilder().build_default_simulation(
        tax_benefit_system, count=count
    )

    restored = _round_trip(simulation, tax_benefit_system, tmp_path / "dump")

    count_files = list(
        (tmp_path / "dump").glob(f"__entities__/*/{simulation_dumper.COUNT_FILE}")
    )
    assert len(count_files) == 2
    for path in count_files:
        assert np.load(path).dtype == np.int64
    for population in restored.populations.values():
        assert type(population.count) is int and population.count == 2
        testing.assert_array_equal(population.ids, [0, 1])


def test_dump_of_a_count_that_is_not_an_integer_infers_it(tax_benefit_system, tmp_path):
    # Only set by hand: the dump records no count, and restoring infers it
    # from the ids, as for an older dump.
    simulation = SimulationBuilder().build_default_simulation(
        tax_benefit_system, count=2
    )
    simulation.household.count = 2.0
    simulation.set_input("rent", "2018-01", np.array([1.0, 2.0]))

    restored = _round_trip(simulation, tax_benefit_system, tmp_path / "dump")

    assert not (
        tmp_path / "dump" / "__entities__" / "household" / simulation_dumper.COUNT_FILE
    ).exists()
    assert restored.household.count == 2
    testing.assert_array_equal(restored.calculate("rent", "2018-01"), [1, 2])


def test_dump_restore_without_group_entities(tmp_path):
    # With no group to read the number of persons from, restoring raised
    # UnboundLocalError.
    system = TaxBenefitSystem(
        [build_entity("person", "persons", "Person", is_person=True)]
    )
    for count in (0, 3):
        simulation = SimulationBuilder().build_default_simulation(system, count=count)
        for without_counts in (False, True):
            restored = _round_trip(
                simulation,
                system,
                tmp_path / f"dump-{count}-{without_counts}",
                without_counts=without_counts,
            )
            assert restored.persons.count == count
            testing.assert_array_equal(restored.persons.ids, simulation.persons.ids)
