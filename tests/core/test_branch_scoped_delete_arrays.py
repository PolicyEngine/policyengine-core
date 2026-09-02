"""Regression tests for deleting caches from nested simulation branches."""

from __future__ import annotations

import gc

import numpy as np

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.simulations import SimulationBuilder


PERIOD = "2017-01"


def _salary_simulation():
    return SimulationBuilder().build_from_entities(
        CountryTaxBenefitSystem(),
        {
            "persons": {"bill": {"salary": {PERIOD: 3_000}}},
            "households": {"household": {"parents": ["bill"]}},
        },
    )


def _purge_formula_outputs(simulation):
    for variable_name in simulation.tax_benefit_system.variables:
        if variable_name not in simulation.input_variables:
            simulation.delete_arrays(variable_name)


def test_nested_branch_recomputes_after_formula_output_purge():
    """A perturbed nested branch must not reuse its parent's formula output."""
    simulation = _salary_simulation()
    measurement = simulation.get_branch("measurement")
    parent_value = measurement.calculate("income_tax", PERIOD).copy()
    perturbed = measurement.get_branch("perturbed")

    _purge_formula_outputs(perturbed)
    perturbed.set_input("salary", PERIOD, np.asarray([6_000.0]))
    recomputed_value = perturbed.calculate("income_tax", PERIOD)

    np.testing.assert_allclose(parent_value, [450.0], rtol=1e-6)
    np.testing.assert_allclose(recomputed_value, [900.0], rtol=1e-6)
    np.testing.assert_allclose(
        measurement.calculate("income_tax", PERIOD),
        parent_value,
    )


def test_delete_arrays_purges_only_visible_branch_names():
    """Root and nested purges must preserve unrelated dispatched values."""
    simulation = _salary_simulation()
    root_holder = simulation.get_holder("salary")
    dispatched_value = np.asarray([7_777.0])
    root_holder._memory_storage.put(dispatched_value, PERIOD, "dispatched")

    simulation.delete_arrays("salary", PERIOD)

    assert root_holder._memory_storage.get(PERIOD, "default") is None
    np.testing.assert_array_equal(
        root_holder._memory_storage.get(PERIOD, "dispatched"),
        dispatched_value,
    )

    measurement = simulation.get_branch("measurement")
    perturbed = measurement.get_branch("perturbed")
    child_holder = perturbed.get_holder("salary")
    child_holder._memory_storage.put(np.asarray([1.0]), PERIOD, "default")
    child_holder._memory_storage.put(np.asarray([2.0]), PERIOD, "measurement")
    child_holder._memory_storage.put(np.asarray([3.0]), PERIOD, "perturbed")

    perturbed.delete_arrays("salary", PERIOD)

    assert child_holder._memory_storage.get(PERIOD, "default") is None
    assert child_holder._memory_storage.get(PERIOD, "measurement") is None
    assert child_holder._memory_storage.get(PERIOD, "perturbed") is None
    np.testing.assert_array_equal(
        child_holder._memory_storage.get(PERIOD, "dispatched"),
        dispatched_value,
    )


def test_on_disk_storage_clone_copies_metadata_without_owning_directory(tmp_path):
    """A cloned disk view must own metadata, but not the shared directory."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    storage = OnDiskStorage(str(storage_dir), is_eternal=True)
    storage.put(np.asarray([12.0]), PERIOD, "default")
    storage._enums["sentinel"] = "enum"

    try:
        assert storage.preserve_storage_dir is False
        clone = storage.clone()

        assert clone is not storage
        assert clone.storage_dir == storage.storage_dir
        assert clone.is_eternal == storage.is_eternal
        assert clone.preserve_storage_dir is True
        assert clone._files == storage._files
        assert clone._files is not storage._files
        assert clone._enums == storage._enums
        assert clone._enums is not storage._enums
        assert clone._storage_dir_owner is storage
        assert clone.clone()._storage_dir_owner is storage
    finally:
        # Let pytest's tmp_path cleanup remove the directory even if an
        # assertion fails before the clone can prove its ownership setting.
        storage.preserve_storage_dir = True


def test_on_disk_storage_clone_keeps_cleanup_owner_alive(tmp_path):
    """A detached clone must remain readable after its source is released."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    storage = OnDiskStorage(str(storage_dir))
    storage.put(np.asarray([12.0]), PERIOD, "default")
    clone = storage.clone()

    del storage
    gc.collect()

    assert storage_dir.is_dir()
    np.testing.assert_array_equal(clone.get(PERIOD, "default"), [12.0])


def test_child_disk_delete_keeps_parent_view_intact(tmp_path):
    """Deleting through a child must rewire only the child's disk mappings."""
    simulation = _salary_simulation()
    parent_holder = simulation.get_holder("salary")
    parent_holder._memory_storage.delete(PERIOD, "default")
    parent_holder._disk_storage = parent_holder.create_disk_storage(str(tmp_path))
    parent_storage = parent_holder._disk_storage
    parent_storage.put(np.asarray([3_000.0]), PERIOD, "default")
    disk_key = f"default_{PERIOD}"

    try:
        child = simulation.get_branch("measurement")
        child_holder = child.get_holder("salary")
        child_storage = child_holder._disk_storage
        assert child_storage._files[disk_key] == parent_storage._files[disk_key]
        inherited_value = child_holder.get_array(PERIOD, child.branch_name)

        child.delete_arrays("salary", PERIOD)

        assert child_storage is not parent_storage
        assert child_storage.preserve_storage_dir is True
        np.testing.assert_array_equal(inherited_value, [3_000.0])
        assert disk_key in parent_storage._files
        assert disk_key not in child_storage._files
        assert (tmp_path / "salary" / f"{disk_key}.npy").is_file()
        np.testing.assert_array_equal(
            parent_holder._disk_storage.get(PERIOD, "default"),
            [3_000.0],
        )
    finally:
        # The path belongs to pytest for this test; avoid a later destructor
        # racing tmp_path cleanup if the assertion above fails.
        parent_storage.preserve_storage_dir = True


def test_derivative_recomputes_inside_named_branch():
    """A derivative clone must purge formula outputs visible to its branch."""
    simulation = _salary_simulation()
    measurement = simulation.get_branch("measurement")
    measurement.calculate("income_tax", PERIOD)

    derivative = measurement.derivative(
        "income_tax",
        "salary",
        PERIOD,
        delta=1,
    )

    np.testing.assert_allclose(derivative, [0.15], rtol=1e-4, atol=1e-6)
