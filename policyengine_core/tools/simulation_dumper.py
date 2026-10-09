# -*- coding: utf-8 -*-


import os
import warnings

import numpy as np

from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.periods import ETERNITY
from policyengine_core.simulations import Simulation

# Next to each variable's arrays: the periods, one per line, whose dumped
# value the simulation calculated (see ``Holder.is_derived``). The file is
# left out when there are none. ``restore_simulation`` restores these values
# as calculated ones, which auto-carry-over and uprating never read and
# ``apply_reform`` drops, and every other value as an input.
DERIVED_PERIODS_FILE = "derived_periods.txt"

# In ``__entities__``, written last by every dump: it says the dump lists
# every calculated value in ``DERIVED_PERIODS_FILE``, so a variable without
# that file has none. Dumps written before policyengine-core 3.32.16 list no
# calculated value, whatever they hold, and those written before this file
# existed have no marker. (``restore_simulation`` reads only named files in
# ``__entities__``, so earlier versions restore a dump with the marker.)
DUMP_FORMAT_FILE = "dump_format.txt"
DUMP_FORMAT_VERSION = 1


def dump_simulation(simulation, directory):
    """
    Write simulation data to directory, so that it can be restored later.

    Writes the default branch's values, listing the periods of those the
    simulation calculated (see ``restore_simulation``).
    """
    parent_directory = os.path.abspath(os.path.join(directory, os.pardir))
    if not os.path.isdir(parent_directory):  # To deal with reforms
        os.mkdir(parent_directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)

    if os.listdir(directory):
        raise ValueError("Directory '{}' is not empty".format(directory))

    entities_dump_dir = os.path.join(directory, "__entities__")
    os.mkdir(entities_dump_dir)

    for entity in simulation.populations.values():
        # Dump entity structure
        _dump_entity(entity, entities_dump_dir)

        # Dump variable values
        for holder in entity._holders.values():
            _dump_holder(holder, directory)

    with open(os.path.join(entities_dump_dir, DUMP_FORMAT_FILE), "w") as file:
        file.write(f"{DUMP_FORMAT_VERSION}\n")


def restore_simulation(directory, tax_benefit_system, **kwargs):
    """
    Restore simulation from directory

    Each value is restored as the dumped simulation stored it: one it
    calculated (listed in the variable's ``derived_periods.txt``) as a
    calculated value, and every other value as an input, recorded in
    ``_user_input_keys`` as ``set_input`` records one, so ``apply_reform``
    keeps the inputs and drops the calculated values, as it would have in
    the dumped simulation.

    A dump lists which values were calculated, not where an input came
    from: a value the dumped simulation stored with ``put_in_cache``
    without ``derived=True`` (an input to auto-carry-over and uprating that
    ``set_input`` did not record, so that ``apply_reform`` dropped it there)
    is restored as a recorded input too.

    A dump written before policyengine-core 3.32.16 lists no calculated
    value, so every value in it is restored as an input. A dump without a
    format marker that lists none may be one of those, so restoring it
    warns: ``apply_reform`` then keeps any calculated value in it as dumped
    instead of recalculating it.
    """
    simulation = Simulation(
        tax_benefit_system, tax_benefit_system.instantiate_entities()
    )

    entities_dump_dir = os.path.join(directory, "__entities__")
    for population in simulation.populations.values():
        if population.entity.is_person:
            continue
        person_count = _restore_entity(population, entities_dump_dir)

    for population in simulation.populations.values():
        if not population.entity.is_person:
            continue
        _restore_entity(population, entities_dump_dir)
        population.count = person_count

    variables_to_restore = [
        variable for variable in os.listdir(directory) if variable != "__entities__"
    ]
    lists_calculated_values = os.path.exists(
        os.path.join(entities_dump_dir, DUMP_FORMAT_FILE)
    )
    for variable in variables_to_restore:
        if _restore_holder(simulation, variable, directory):
            lists_calculated_values = True
    if variables_to_restore and not lists_calculated_values:
        warnings.warn(
            f"The simulation dump in {directory} does not say which of its "
            f"values were calculated: it has no {DUMP_FORMAT_FILE} and lists "
            "no calculated value, as dumps written by policyengine-core "
            "before 3.32.16 never do. Every value in it is restored as an "
            "input, so apply_reform keeps any calculated value in it as "
            "dumped instead of recalculating it. Dump the simulation again "
            "to record which values were calculated.",
            stacklevel=2,
        )

    return simulation


def _dump_holder(holder, directory):
    disk_storage = holder.create_disk_storage(directory, preserve=True)
    derived_periods = set()
    for period in holder.get_known_periods():
        value = holder.get_array(period)
        disk_storage.put(value, period)
        # Read the mark of exactly the value dumped: the same period on the
        # same branch as ``get_array``.
        if holder.is_derived(period):
            derived_periods.add(str(period))
    if derived_periods:
        path = os.path.join(disk_storage.storage_dir, DERIVED_PERIODS_FILE)
        with open(path, "w") as file:
            file.write("\n".join(sorted(derived_periods)) + "\n")


def _dump_entity(population, directory):
    path = os.path.join(directory, population.entity.key)
    os.mkdir(path)
    np.save(os.path.join(path, "id.npy"), population.ids)

    if population.entity.is_person:
        return

    np.save(os.path.join(path, "members_position.npy"), population.members_position)
    np.save(
        os.path.join(path, "members_entity_id.npy"),
        population.members_entity_id,
    )

    flattened_roles = population.entity.flattened_roles
    if len(flattened_roles) == 0:
        encoded_roles = np.int64(0)
    else:
        encoded_roles = np.select(
            [population.members_role == role for role in flattened_roles],
            [str(role.key) for role in flattened_roles],
            default="unknown",
        )
    np.save(os.path.join(path, "members_role.npy"), encoded_roles)


def _restore_entity(population, directory):
    path = os.path.join(directory, population.entity.key)

    population.ids = np.load(os.path.join(path, "id.npy"))

    if population.entity.is_person:
        return

    population.members_position = np.load(os.path.join(path, "members_position.npy"))
    population.members_entity_id = np.load(os.path.join(path, "members_entity_id.npy"))
    encoded_roles = np.load(os.path.join(path, "members_role.npy"))

    flattened_roles = population.entity.flattened_roles
    if len(flattened_roles) == 0:
        population.members_role = np.int64(0)
    else:
        population.members_role = np.select(
            [encoded_roles == role.key for role in flattened_roles],
            [role for role in flattened_roles],
        )
    person_count = len(population.members_entity_id)
    population.count = max(population.members_entity_id) + 1
    return person_count


def _restore_holder(simulation, variable, directory):
    """Restore one variable's values: those its ``DERIVED_PERIODS_FILE``
    lists as calculated values, every other one as an input. Return whether
    it lists any."""
    storage_dir = os.path.join(directory, variable)
    is_variable_eternal = (
        simulation.tax_benefit_system.get_variable(variable).definition_period
        == ETERNITY
    )
    disk_storage = OnDiskStorage(
        storage_dir, is_eternal=is_variable_eternal, preserve_storage_dir=True
    )
    disk_storage.restore()

    holder = simulation.get_holder(variable)

    derived_periods_path = os.path.join(storage_dir, DERIVED_PERIODS_FILE)
    derived_periods = set()
    # Legacy dumps have no provenance record; their calculated values may restore as inputs.
    if os.path.exists(derived_periods_path):
        with open(derived_periods_path) as file:
            derived_periods = set(file.read().split())

    for period in disk_storage.get_known_periods():
        value = disk_storage.get(period)
        if str(period) in derived_periods:
            holder.put_in_cache(value, period, derived=True)
        else:
            _restore_input(simulation, holder, period, value)
    return bool(derived_periods)


def _restore_input(simulation, holder, period, value):
    """Store ``value`` as an input, recorded as ``set_input`` records one.

    ``Holder.set_input`` would also run the variable's ``set_input`` helper,
    but a dump holds values already split into the variable's own periods.
    """
    if not hasattr(simulation, "_user_input_contexts"):
        simulation._user_input_contexts = []
    simulation._user_input_contexts.append(simulation.branch_name)
    try:
        holder._set(period, value, simulation.branch_name)
    finally:
        simulation._user_input_contexts.pop()
