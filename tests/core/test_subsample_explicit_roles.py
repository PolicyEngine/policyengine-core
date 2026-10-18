"""``subsample`` keeps the roles members were loaded with.

The rebuild reads each member's role from a ``person_<entity>_role`` column.
That column used to be carried only when the system defined a variable of
that name, so a system without one rebuilt every member with the default
role: silently on master, and as a capacity error once implicit defaults
are capacity-checked (review P3-1 on PolicyEngine/policyengine-core#595).
Person and membership IDs had the same gap.
"""

import logging

import pandas as pd
import pytest

from policyengine_core.data import Dataset
from policyengine_core.periods import ETERNITY
from tests.fixtures.subsample_roles import (
    DATASET_YEAR,
    NO_ROLE,
    ROLE,
    households_by_members,
    load,
    load_dataframe,
    role_counts,
    role_keys,
    role_system,
    roles_by_person,
    source_households,
    source_roles,
)

HOUSEHOLDS = [
    ["first_parent", "second_parent", "child", "child"],
    ["first_parent", "child"],
    ["first_parent", "lodger"],
    ["second_parent", "first_parent", "child", "parent"],
    ["child"],
]
# Shuffled so that a household's members are not adjacent rows.
ORDER = [7, 0, 12, 3, 9, 1, 5, 10, 2, 8, 11, 4, 6]
UNKNOWN_COLUMN_WARNING = "do not match any variable"


def assert_roles_kept(simulation, households):
    """Every kept household is a whole source household with its roles."""
    source = source_households(households)
    kept = households_by_members(simulation)
    assert kept, "subsample kept no household"
    for members, roles in kept.items():
        assert members in source, f"{sorted(members)} is not a source household"
        assert roles == source[members]
    expected = source_roles(households)
    assert roles_by_person(simulation) == {
        person: expected[person] for person in simulation.persons.ids.tolist()
    }
    # The projector counts the same roles the members hold.
    household_order = [
        frozenset(
            simulation.persons.ids[
                simulation.household.members_entity_id == index
            ].tolist()
        )
        for index in range(simulation.household.count)
    ]
    for key, counts in role_counts(simulation).items():
        assert counts == [kept[members][key] for members in household_order]


def test_review_reproduction_keeps_explicit_roles():
    """The reviewed case: no role variable and a bounded default role."""
    data = pd.DataFrame(
        {
            f"person_id__{DATASET_YEAR}": [0, 1, 2, 3],
            f"person_household_id__{DATASET_YEAR}": [0, 0, 0, 1],
            f"{ROLE}__{DATASET_YEAR}": [
                "first_parent",
                "child",
                "child",
                "first_parent",
            ],
            f"household_weight__{DATASET_YEAR}": [1.0, 1.0, 1.0, 1.0],
        }
    )
    simulation = load_dataframe(data, default_role="first_parent")
    assert role_counts(simulation) == {
        "first_parent": [1, 1],
        "second_parent": [0, 0],
        "child": [2, 0],
    }

    simulation.subsample(n=2, seed=0)

    # Master rebuilt everyone as first_parent: first_parent [3, 1], child [0, 0].
    assert simulation.household.count == 2
    assert role_counts(simulation) == {
        "first_parent": [1, 1],
        "second_parent": [0, 0],
        "child": [2, 0],
    }
    assert roles_by_person(simulation) == {
        0: "first_parent",
        1: "child",
        2: "child",
        3: "first_parent",
    }


@pytest.mark.parametrize("role_variable", ["absent", "string"])
@pytest.mark.parametrize("role_column", [ROLE, "role"])
@pytest.mark.parametrize("default_role", ["first_parent", "child", "parent", None])
def test_explicit_roles_survive_subsample(role_variable, role_column, default_role):
    simulation = load(
        HOUSEHOLDS,
        role_variable=role_variable,
        role_column=role_column,
        default_role=default_role,
        order=ORDER,
    )
    assert_roles_kept(simulation, HOUSEHOLDS)
    assert simulation.household.count == len(HOUSEHOLDS)

    simulation.subsample(n=8, seed="roles", time_period=DATASET_YEAR)

    assert simulation.household.count >= 2
    assert_roles_kept(simulation, HOUSEHOLDS)


def test_roles_survive_repeated_subsamples():
    simulation = load(HOUSEHOLDS, order=ORDER, default_role="child")
    for n, seed in [(8, "first"), (3, "second"), (2, "third")]:
        simulation.subsample(n=n, seed=seed, time_period=DATASET_YEAR)
        assert_roles_kept(simulation, HOUSEHOLDS)


def test_rebuilt_dataset_carries_role_keys():
    simulation = load(HOUSEHOLDS, order=ORDER)
    simulation.subsample(n=8, seed="roles", time_period=DATASET_YEAR)

    data = simulation.dataset.load()
    [role_column] = [column for column in data if column.split("__")[0] == ROLE]
    [id_column] = [column for column in data if column.split("__")[0] == "person_id"]
    assert role_column == f"{ROLE}__{ETERNITY}"
    expected = source_roles(HOUSEHOLDS)
    people = data[id_column].tolist()
    assert dict(zip(people, data[role_column].tolist())) == {
        person: expected[person] for person in people
    }


@pytest.mark.parametrize("missing", ["person_id", "person_household_id"])
def test_structural_ids_without_variables_survive_subsample(missing):
    """Master raised: ``KeyError`` mid-rebuild, or ``IndexError``."""
    system = role_system("absent")
    del system.variables[missing]
    simulation = load(HOUSEHOLDS, order=ORDER, system=system)
    source_ids = sorted(simulation.persons.ids.tolist())

    simulation.subsample(n=8, seed="roles", time_period=DATASET_YEAR)

    assert set(simulation.persons.ids.tolist()) <= set(source_ids)
    assert_roles_kept(simulation, HOUSEHOLDS)


@pytest.mark.parametrize("unmatched", ["lodger", "parent"])
def test_integer_role_variable_rejects_roleless_members_before_changing_anything(
    unmatched,
):
    """An integer role column cannot say "no role"; master wrote role 0."""
    households = [["first_parent", "child"], ["first_parent", unmatched]]
    simulation = load(households, role_variable="integer", role_column="role")
    dataset = simulation.dataset
    persons = simulation.persons.ids.tolist()
    roles = role_keys(simulation).tolist()
    weights = simulation.calculate("household_weight", DATASET_YEAR).tolist()

    with pytest.raises(ValueError, match="integer role variable.*no role"):
        simulation.subsample(n=2, seed="roles", time_period=DATASET_YEAR)

    assert simulation.dataset is dataset
    assert simulation.persons.ids.tolist() == persons
    assert role_keys(simulation).tolist() == roles
    assert simulation.calculate("household_weight", DATASET_YEAR).tolist() == weights


def test_integer_role_variable_carries_flattened_roles():
    households = [["first_parent", "second_parent", "child"], ["child", "first_parent"]]
    simulation = load(households, role_variable="integer", role_column="role")

    simulation.subsample(n=4, seed="roles", time_period=DATASET_YEAR)

    assert_roles_kept(simulation, households)


def unknown_column_warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if UNKNOWN_COLUMN_WARNING in record.getMessage()
    ]


@pytest.mark.parametrize("role_column", [ROLE, "role"])
def test_structural_columns_are_not_reported_as_ignored(caplog, role_column):
    """The loader reads these columns, though no variable has their name."""
    system = role_system("absent")
    del system.variables["person_household_id"]
    with caplog.at_level(logging.WARNING):
        simulation = load(HOUSEHOLDS, role_column=role_column, system=system)

    assert unknown_column_warnings(caplog) == []
    assert_roles_kept(simulation, HOUSEHOLDS)


@pytest.mark.parametrize("role_column", [ROLE, "role", None])
def test_carried_role_columns_are_not_reported_as_ignored(caplog, role_column):
    """Like a system with no role variables (policyengine-us, for instance),
    whose subsample now carries a role column per group entity."""
    households = HOUSEHOLDS if role_column else [["child"] * 3, ["child"], ["child"]]
    simulation = load(households, role_column=role_column, default_role="child")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        simulation.subsample(n=8, seed="roles", time_period=DATASET_YEAR)

    assert unknown_column_warnings(caplog) == []
    assert_roles_kept(simulation, households)


def test_unknown_columns_are_still_reported_beside_structural_ones(caplog):
    data = load(HOUSEHOLDS).dataset.load()
    data[f"a_removed_input__{DATASET_YEAR}"] = 1.0
    with caplog.at_level(logging.WARNING):
        load_dataframe(data)

    [warning] = unknown_column_warnings(caplog)
    assert "1 column(s)" in warning
    assert f"a_removed_input__{DATASET_YEAR}" in warning
    assert ROLE not in warning


def test_time_period_array_role_column_is_used_and_not_reported(caplog):
    frame = load(HOUSEHOLDS).dataset.load()
    data = {column.split("__")[0]: {ETERNITY: frame[column].values} for column in frame}
    data["household_id"] = {ETERNITY: [10, 20, 30, 40, 50]}
    data["household_weight"] = {DATASET_YEAR: [1.0, 2.0, 3.0, 4.0, 5.0]}
    dataset = type(
        "RoleArrays",
        (Dataset,),
        {
            "name": "role_arrays",
            "label": "Role arrays",
            "data_format": Dataset.TIME_PERIOD_ARRAYS,
            "file_path": "dataframe",
            "time_period": DATASET_YEAR,
            "load": lambda self: data,
        },
    )()
    simulation_class = type(load(HOUSEHOLDS))
    with caplog.at_level(logging.WARNING):
        simulation = simulation_class(
            tax_benefit_system=role_system("absent"), dataset=dataset
        )

    assert unknown_column_warnings(caplog) == []
    assert_roles_kept(simulation, HOUSEHOLDS)
    assert NO_ROLE in role_keys(simulation).tolist()
