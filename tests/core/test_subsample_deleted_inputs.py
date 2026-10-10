"""Deleted source inputs must not turn later calculations into source data.

Subsampling after deleting an override must match subsampling without reading
the deleted variable first. This holds for branches and both storage backends;
new overrides are still preserved.
"""

import numpy as np
import pytest

from policyengine_core.country_template.entities import Person
from policyengine_core.experimental import MemoryConfig
from policyengine_core.periods import ETERNITY, YEAR, period
from policyengine_core.variables import Variable
from tests.fixtures.subsample_inputs import DATASET_YEAR, build, stored


class deleted_leaf(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "An input-only variable whose deleted value becomes a default"
    default_value = 17.0


class deleted_eternal(Variable):
    value_type = float
    entity = Person
    definition_period = ETERNITY
    label = "An eternal formula variable with a deleted override"

    def formula(person, period):
        return person("person_id", period) * 2


def prepare(variable, on_disk=False, branch=False):
    simulation = build()
    simulation.tax_benefit_system.add_variables(deleted_leaf, deleted_eternal)
    holder = simulation.get_holder(variable)
    if on_disk:
        # Exercise this variable's provenance on disk, including after the
        # rebuild, without spilling unrelated object-valued role arrays.
        simulation.memory_config = MemoryConfig(
            max_memory_occupation=0,
            priority_variables=[
                name
                for name in simulation.tax_benefit_system.variables
                if name != variable
            ],
        )
        holder._disk_storage = holder.create_disk_storage()
        holder._on_disk_storable = True
    simulation.set_input(variable, DATASET_YEAR, np.full(simulation.persons.count, 99))
    if on_disk:
        assert holder._disk_storage.has(period(DATASET_YEAR))
    if branch:
        simulation = simulation.get_branch("child")
    return simulation


def delete_override(simulation, variable, through_holder):
    if through_holder:
        holder = simulation.get_holder(variable)
        for branch_name in simulation._get_visible_branch_names():
            holder.delete_arrays(DATASET_YEAR, branch_name)
    else:
        simulation.delete_arrays(variable, DATASET_YEAR)


@pytest.mark.parametrize("variable", ["doubled", "deleted_leaf", "deleted_eternal"])
@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
@pytest.mark.parametrize("through_holder", [False, True], ids=["simulation", "holder"])
def test_deleted_input_recalculated_as_derived_is_not_exportable(
    variable, on_disk, through_holder
):
    simulation = prepare(variable, on_disk)
    delete_override(simulation, variable, through_holder)
    simulation.calculate(variable, DATASET_YEAR)

    assert simulation.get_holder(variable).is_derived(
        period(DATASET_YEAR), simulation.branch_name
    )
    assert simulation._get_set_input_periods(variable) == []


@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
@pytest.mark.parametrize("branch", [False, True], ids=["default", "child"])
def test_deleted_formula_override_does_not_freeze_after_subsample(on_disk, branch):
    simulation = prepare("doubled", on_disk, branch)
    simulation.delete_arrays("doubled", DATASET_YEAR)
    simulation.calculate("doubled", DATASET_YEAR)

    simulation.subsample(n=3, seed="deleted-input", time_period=DATASET_YEAR)
    simulation.set_input("base", DATASET_YEAR, np.full(simulation.persons.count, 5))

    np.testing.assert_array_equal(
        simulation.calculate("doubled", DATASET_YEAR),
        np.full(simulation.persons.count, 10),
    )


@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
@pytest.mark.parametrize("branch", [False, True], ids=["default", "child"])
def test_new_override_after_deletion_is_still_exportable(on_disk, branch):
    simulation = prepare("doubled", on_disk, branch)
    simulation.delete_arrays("doubled", DATASET_YEAR)
    simulation.calculate("doubled", DATASET_YEAR)
    simulation.set_input("doubled", DATASET_YEAR, np.full(simulation.persons.count, 7))

    assert simulation._get_set_input_periods("doubled") == [period(DATASET_YEAR)]
    simulation.subsample(n=3, seed="deleted-input", time_period=DATASET_YEAR)
    np.testing.assert_array_equal(
        simulation.calculate("doubled", DATASET_YEAR),
        np.full(simulation.persons.count, 7),
    )


def test_reading_deleted_overrides_does_not_change_subsample():
    """Property: storage and all later formulas ignore pre-sample reads.

    Vary the deleted override, storage, branch, deletion API, sample size,
    and subsequent leaf input. A never-read twin is the differential oracle.
    """
    hypothesis = pytest.importorskip("hypothesis")
    st = hypothesis.strategies

    @hypothesis.settings(max_examples=40, deadline=None)
    @hypothesis.given(
        override=st.integers(-100, 100),
        on_disk=st.booleans(),
        branch=st.booleans(),
        through_holder=st.booleans(),
        n=st.integers(1, 6),
        later_input=st.integers(-100, 100),
    )
    def check(override, on_disk, branch, through_holder, n, later_input):
        fresh = prepare("doubled", on_disk, branch)
        used = prepare("doubled", on_disk, branch)
        for simulation in (fresh, used):
            simulation.set_input(
                "doubled", DATASET_YEAR, np.full(simulation.persons.count, override)
            )
            delete_override(simulation, "doubled", through_holder)
        used.calculate("doubled", DATASET_YEAR)
        for simulation in (fresh, used):
            simulation.subsample(n=n, seed="deleted-input", time_period=DATASET_YEAR)

        assert stored(used) == stored(fresh)
        for simulation in (fresh, used):
            simulation.set_input(
                "base", DATASET_YEAR, np.full(simulation.persons.count, later_input)
            )
        np.testing.assert_array_equal(
            used.calculate("doubled", DATASET_YEAR),
            fresh.calculate("doubled", DATASET_YEAR),
        )
        assert stored(used) == stored(fresh)

    check()
