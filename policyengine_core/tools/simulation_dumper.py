# -*- coding: utf-8 -*-


import os
import warnings

import numpy as np

from policyengine_core.data_storage import OnDiskStorage
from policyengine_core import periods
from policyengine_core.periods import ETERNITY
from policyengine_core.simulations import Simulation

# Next to each variable's arrays: the periods, one per line, whose dumped
# value was an input (stored through ``set_input``). ``restore_simulation``
# registers exactly these as inputs, so ``apply_reform``, which keeps inputs
# and drops calculated values, keeps the same values in the restored
# simulation as in the dumped one.
INPUT_PERIODS_FILE = "inputs.txt"

# Periods, one per line, whose dumped value the simulation calculated (see
# ``Holder.is_derived``), so a restored simulation does not carry them over.
DERIVED_PERIODS_FILE = "derived_periods.txt"


def dump_simulation(simulation, directory):
    """
    Write simulation data to directory, so that it can be restored later.
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

    input_keys = _input_storage_keys(simulation)
    for entity in simulation.populations.values():
        # Dump entity structure
        _dump_entity(entity, entities_dump_dir)

        # Dump variable values
        for holder in entity._holders.values():
            _dump_holder(holder, directory, input_keys)


def restore_simulation(directory, tax_benefit_system, **kwargs):
    """
    Restore simulation from directory

    Values the dumped simulation stored as inputs are restored as inputs
    (recorded in ``_user_input_keys``, as ``set_input`` records them), and
    every other value as a calculated one, so ``apply_reform`` keeps and drops
    the same values it would have in the dumped simulation. A dump written
    before inputs were recorded (no ``inputs.txt``) does not say which values
    were inputs, so every value in it is restored as an input, with a
    warning: ``apply_reform`` then keeps its calculated values as dumped.
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

    variables_to_restore = (
        variable for variable in os.listdir(directory) if variable != "__entities__"
    )
    without_input_record = [
        variable
        for variable in variables_to_restore
        if not _restore_holder(simulation, variable, directory)
    ]
    if without_input_record:
        warnings.warn(
            f"The simulation dump in {directory} does not record which values "
            f"were inputs ({len(without_input_record)} variables have no "
            f"{INPUT_PERIODS_FILE}; it was written by an earlier version of "
            "policyengine-core). Every value in it is restored as an input, "
            "so apply_reform keeps the calculated values as dumped instead of "
            "recalculating them. Dump the simulation again to record its "
            "inputs.",
            stacklevel=2,
        )

    return simulation


def _dump_holder(holder, directory, input_keys=frozenset()):
    disk_storage = holder.create_disk_storage(directory, preserve=True)
    input_periods = []
    derived_periods = set()
    for period in holder.get_known_periods():
        value = holder.get_array(period)
        disk_storage.put(value, period)
        # The input record and derived mark of exactly the value dumped:
        # ``get_array`` above reads the default branch.
        if (holder.variable.name, "default", str(period)) in input_keys:
            input_periods.append(str(period))
        if holder.is_derived(period):
            derived_periods.add(str(period))
    path = os.path.join(disk_storage.storage_dir, INPUT_PERIODS_FILE)
    with open(path, "w") as file:
        file.write("".join(f"{period}\n" for period in dict.fromkeys(input_periods)))
    if derived_periods:
        path = os.path.join(disk_storage.storage_dir, DERIVED_PERIODS_FILE)
        with open(path, "w") as file:
            file.write("\n".join(sorted(derived_periods)) + "\n")


def _input_storage_keys(simulation):
    """The storage keys ``_user_input_keys`` records as inputs.

    Each record entry becomes ``(variable, branch, period)`` with the period
    as storage writes it, as ``Simulation._invalidate_all_caches`` reads the
    record back through the storage: an ETERNITY variable's one value is an
    input whatever period its entry names.
    """
    input_keys = set()
    for name, branch_name, period in getattr(simulation, "_user_input_keys", ()):
        variable = simulation.tax_benefit_system.get_variable(name)
        if variable is not None and variable.definition_period == ETERNITY:
            period = ETERNITY
        elif period is None:
            continue
        input_keys.add((name, branch_name, str(periods.period(period))))
    return input_keys


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
    """Restore one variable's values; return whether its inputs were recorded."""
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

    input_periods_path = os.path.join(storage_dir, INPUT_PERIODS_FILE)
    if os.path.exists(input_periods_path):
        with open(input_periods_path) as file:
            input_periods = set(file.read().split())
    else:
        # Dumped before inputs were recorded: nothing says which values were
        # calculated, so keep every value as an input.
        input_periods = None

    derived_periods_path = os.path.join(storage_dir, DERIVED_PERIODS_FILE)
    derived_periods = set()
    if os.path.exists(derived_periods_path):
        with open(derived_periods_path) as file:
            derived_periods = set(file.read().split())

    for period in disk_storage.get_known_periods():
        value = disk_storage.get(period)
        if str(period) in derived_periods:
            holder.put_in_cache(value, period, derived=True)
        elif input_periods is None or str(period) in input_periods:
            _restore_input(simulation, holder, period, value)
        else:
            holder.put_in_cache(value, period)
    return input_periods is not None


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
