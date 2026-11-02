"""Dataset defaults preserve literal role matching and person-level cardinality.

The role table below specifies expected assignments independently of the loader:
only a literal flattened-role key matches; unknown and compound keys remain 0.
For accepted defaults, role counts, aggregates, projections and unique lookups
are computed from the original dataset memberships. A capacity-limited default
must be rejected if assigning it uniformly would exceed its per-group maximum.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.data import Dataset  # noqa: E402
from policyengine_core.entities import build_entity  # noqa: E402
from policyengine_core.periods import ETERNITY  # noqa: E402
from policyengine_core.projectors import UniqueRoleToEntityProjector  # noqa: E402
from policyengine_core.simulations import SimulationBuilder  # noqa: E402
from policyengine_core.taxbenefitsystems import TaxBenefitSystem  # noqa: E402
from tests.core.test_dataset_default_role import (  # noqa: E402
    in_memory_dataset,
    simulate,
)

PERSON = build_entity(key="person", plural="persons", label="Person", is_person=True)
HOUSEHOLD = build_entity(
    key="household",
    plural="households",
    label="Household",
    roles=[
        {
            "key": "parent",
            "plural": "parents",
            "subroles": ["first_parent", "second_parent"],
        },
        {"key": "child", "plural": "children"},
    ],
)
FAMILY = build_entity(
    key="family",
    plural="families",
    label="Family",
    roles=[
        {"key": "child", "plural": "children"},
        {"key": "parent", "plural": "parents", "subroles": ["mother", "father"]},
    ],
)
UNIT = build_entity(
    key="unit",
    plural="units",
    label="Unit",
    roles=[{"key": "member", "plural": "members"}],
)
COMMITTEE = build_entity(
    key="committee",
    plural="committees",
    label="Committee",
    roles=[{"key": "delegate", "plural": "delegates", "max": 2}],
)
GROUP_ENTITIES = [HOUSEHOLD, FAMILY, UNIT, COMMITTEE]
SYSTEM = TaxBenefitSystem([PERSON, *GROUP_ENTITIES])

# Deliberately explicit: this oracle does not reproduce a resolution algorithm.
EXPECTED_DEFAULT_ROLES = {
    "household": {
        "child": HOUSEHOLD.CHILD,
        "first_parent": HOUSEHOLD.FIRST_PARENT,
        "second_parent": HOUSEHOLD.SECOND_PARENT,
    },
    "family": {"child": FAMILY.CHILD, "mother": FAMILY.MOTHER, "father": FAMILY.FATHER},
    "unit": {"member": UNIT.MEMBER},
    "committee": {"delegate": COMMITTEE.DELEGATE},
}
DEFAULT_ROLES = [
    "member",
    "parent",
    "child",
    "first_parent",
    "second_parent",
    "mother",
    "father",
    "delegate",
    "nobody",
]


class DatasetCase(NamedTuple):
    dataset: Dataset
    memberships: dict[str, np.ndarray]
    values: np.ndarray


def _case(columns, data_format, values):
    memberships = {
        entity.key: columns[f"person_{entity.key}_id"].copy()
        for entity in GROUP_ENTITIES
    }
    if data_format == Dataset.FLAT_FILE:
        dataset = Dataset.from_dataframe(pd.DataFrame(columns), "2024")
    elif data_format == Dataset.TIME_PERIOD_ARRAYS:
        dataset = in_memory_dataset(
            {name: {ETERNITY: array} for name, array in columns.items()}, data_format
        )
    else:
        dataset = in_memory_dataset(columns, data_format)
    return DatasetCase(dataset, memberships, np.array(values, dtype=float))


def _uniform_case(persons, group_ids, memberships):
    columns = {"person_id": np.arange(1, persons + 1)}
    for entity in GROUP_ENTITIES:
        columns[f"{entity.key}_id"] = np.array(group_ids, dtype=int)
        columns[f"person_{entity.key}_id"] = np.array(memberships, dtype=int)
    return _case(columns, Dataset.ARRAYS, np.arange(1, persons + 1) * 100)


@st.composite
def _datasets(draw):
    persons = draw(st.integers(min_value=0, max_value=12))
    data_format = draw(
        st.sampled_from([Dataset.ARRAYS, Dataset.TIME_PERIOD_ARRAYS, Dataset.FLAT_FILE])
    )
    columns = {"person_id": np.arange(100, 100 + persons)}
    for entity in GROUP_ENTITIES:
        if data_format == Dataset.FLAT_FILE:
            # The loader synthesizes group IDs 0..n-1 for a flat file. Choose
            # consecutive occupied IDs to isolate roles from ID renumbering.
            group_ids = list(range(max(persons, 1)))
        else:
            group_ids = draw(
                st.lists(
                    st.integers(min_value=-50, max_value=50),
                    min_size=1 if persons else 0,
                    max_size=6,
                    unique=True,
                )
            )
        if persons:
            occupied = draw(
                st.integers(min_value=1, max_value=min(persons, len(group_ids)))
            )
            # Keep empty groups after occupied ones in sorted ID order. The
            # existing join's treatment of other empty-group positions is
            # separate from default-role assignment.
            occupied_ids = sorted(group_ids)[:occupied]
            memberships = occupied_ids + draw(
                st.lists(
                    st.sampled_from(occupied_ids),
                    min_size=persons - occupied,
                    max_size=persons - occupied,
                )
            )
            memberships = draw(st.permutations(memberships))
        else:
            memberships = []
        if data_format != Dataset.FLAT_FILE:
            columns[f"{entity.key}_id"] = np.array(group_ids, dtype=int)
        columns[f"person_{entity.key}_id"] = np.array(memberships, dtype=int)
    values = draw(
        st.lists(
            st.integers(min_value=-1000, max_value=1000),
            min_size=persons,
            max_size=persons,
        )
    )
    return _case(columns, data_format, values)


@settings(max_examples=200, deadline=None)
@given(_datasets(), st.sampled_from(DEFAULT_ROLES))
@example(_uniform_case(2, [10, 20], [10, 10]), "member")
@example(_uniform_case(3, [10, 20], [10, 10, 10]), "first_parent")
@example(_uniform_case(3, [10, 20], [10, 10, 10]), "delegate")
@example(_uniform_case(2, [10, 20], [10, 10]), "delegate")
@example(_uniform_case(3, [20, 10, 30], [20, 10, 30]), "first_parent")
def test_default_roles_preserve_literal_matching_and_capacity(case, default_role):
    dataset, memberships, values = case
    persons = len(values)
    violations = []
    for entity in GROUP_ENTITIES:
        role = EXPECTED_DEFAULT_ROLES[entity.key].get(default_role)
        if role is not None and role.max is not None:
            if any(
                count > role.max for count in Counter(memberships[entity.key]).values()
            ):
                violations.append(entity)
    if violations:
        with pytest.raises(ValueError, match="default role.*at most") as error:
            simulate(SYSTEM, dataset, default_role=default_role)
        assert f"person_{violations[0].key}_role" in str(error.value)
        return

    simulation = simulate(SYSTEM, dataset, default_role=default_role)

    for entity in GROUP_ENTITIES:
        group = simulation.populations[entity.key]
        role = EXPECTED_DEFAULT_ROLES[entity.key].get(default_role)
        raw_memberships = memberships[entity.key]

        assert group.members_role.shape == (persons,)
        assert len(group.members_entity_id) == persons
        if role is None:
            assert (group.members_role == 0).all()
        else:
            assert all(member_role is role for member_role in group.members_role)
            if role.max is not None:
                assert all(
                    count <= role.max for count in Counter(raw_memberships).values()
                )

        group_values = np.arange(group.count, dtype=float)
        group_indices = {group_id: index for index, group_id in enumerate(group.ids)}
        counts = np.array(
            [(raw_memberships == group_id).sum() for group_id in group.ids]
        )
        totals = np.array(
            [values[raw_memberships == group_id].sum() for group_id in group.ids]
        )
        projected_values = np.array(
            [group_values[group_indices[group_id]] for group_id in raw_memberships]
        )

        for query_role in [*entity.roles, *entity.flattened_roles]:
            holds = role is not None and (
                query_role is role or role in (query_role.subroles or [])
            )
            has_role = simulation.persons.has_role(query_role)
            nb_persons = group.nb_persons(role=query_role)
            total = group.sum(values, role=query_role)
            projected = group.project(group_values, role=query_role)

            assert has_role.shape == projected.shape == (persons,)
            assert nb_persons.shape == total.shape == (group.count,)
            assert np.array_equal(has_role, np.full(persons, holds))
            assert np.array_equal(
                nb_persons, counts if holds else np.zeros(group.count)
            )
            if query_role.max is not None:
                assert (nb_persons <= query_role.max).all()
            assert np.array_equal(total, totals if holds else np.zeros(group.count))
            assert np.array_equal(
                projected, projected_values if holds else np.zeros(persons)
            )
            if persons:
                maximum = group.max(values, role=query_role)
                expected_maximum = np.array(
                    [
                        max(values[raw_memberships == group_id], default=-np.inf)
                        if holds
                        else -np.inf
                        for group_id in group.ids
                    ]
                )
                assert maximum.shape == (group.count,)
                assert np.array_equal(maximum, expected_maximum)

        for unique_role in entity.flattened_roles:
            if unique_role.max != 1:
                continue
            # The accepted-capacity invariant guarantees zero or one selected
            # person per raw group ID. Unmatched roles select nobody.
            expected_lookup = np.array(
                [
                    next(iter(values[raw_memberships == group_id]), -123)
                    if role is unique_role
                    else -123
                    for group_id in group.ids
                ]
            )
            lookup = group.value_from_person(values, unique_role, default=-123)
            assert lookup.shape == (group.count,)
            assert np.array_equal(lookup, expected_lookup)

            expected_projector = np.array(
                [
                    next(iter(values[raw_memberships == group_id]), 0)
                    if role is unique_role
                    else 0
                    for group_id in group.ids
                ]
            )
            projector = UniqueRoleToEntityProjector(group, unique_role)
            assert np.array_equal(projector.transform(values), expected_projector)

        for compound_role in entity.roles:
            if not compound_role.subroles or len(compound_role.subroles) != 2:
                continue
            # Uniform defaults cannot supply both partner subroles. Pass the
            # public person-to-group projector so group values project back to
            # person shape, as in existing partner contract tests.
            partners = simulation.persons.value_from_partner(
                values, getattr(simulation.persons, entity.key), compound_role
            )
            assert partners.shape == (persons,)
            assert not partners.any()


@settings(max_examples=50, deadline=None)
@given(
    st.integers(min_value=1, max_value=50), st.sampled_from(["short", "long", "matrix"])
)
@example(0, "matrix")
def test_join_requires_a_one_dimensional_role_per_person(persons, shape):
    builder = SimulationBuilder()
    builder.create_entities(SYSTEM)
    builder.declare_person_entity("person", np.arange(persons))
    household = builder.declare_entity("household", [10])
    if shape == "matrix":
        roles = np.zeros((persons, 1), dtype=int)
    else:
        roles = np.zeros(persons + (1 if shape == "long" else -1), dtype=int)

    with pytest.raises(ValueError, match="give one role per person"):
        builder.join_with_persons(household, np.repeat(10, persons), roles)


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=12),
    st.booleans(),
    st.booleans(),
    st.sampled_from(["2024", "eternity"]),
)
@example([0, 2, 2], False, False, "2024")
@example([0, 0, 0], True, True, "eternity")
def test_suffixed_entity_roles_preserve_explicit_assignment(
    role_indices, generic_role, literal_keys, suffix
):
    # Independent encoding table: do not derive expected roles from the loader.
    role_table = {
        0: ("first_parent", HOUSEHOLD.FIRST_PARENT),
        1: ("second_parent", HOUSEHOLD.SECOND_PARENT),
        2: ("child", HOUSEHOLD.CHILD),
    }
    persons = len(role_indices)
    explicit_roles = (
        [role_table[index][0] for index in role_indices]
        if literal_keys
        else role_indices
    )
    columns = {
        "person_id": np.arange(1, persons + 1),
        # Group zero matches the flat-file loader's synthetic group IDs.
        "person_household_id": np.zeros(persons, dtype=int),
        f"person_household_role__{suffix}": explicit_roles,
    }
    if generic_role:
        # Every generic entry conflicts with its entity-specific counterpart.
        columns["role"] = [(index + 1) % 3 for index in role_indices]
    simulation = simulate(
        TaxBenefitSystem([PERSON, HOUSEHOLD]),
        Dataset.from_dataframe(pd.DataFrame(columns), "2024"),
        default_role="first_parent",
    )
    household = simulation.household
    assert list(household.members_role) == [
        role_table[index][1] for index in role_indices
    ]

    # Explicit columns retain their existing capacity handling: repeated unique
    # roles remain accepted. Counts and sums still follow the declared roles.
    values = np.arange(1, persons + 1)
    query_roles = [
        (HOUSEHOLD.FIRST_PARENT, {0}),
        (HOUSEHOLD.SECOND_PARENT, {1}),
        (HOUSEHOLD.CHILD, {2}),
        (HOUSEHOLD.PARENT, {0, 1}),
    ]
    for role, matching_indices in query_roles:
        expected_count = sum(index in matching_indices for index in role_indices)
        expected_sum = sum(
            value
            for value, index in zip(values, role_indices)
            if index in matching_indices
        )
        assert household.nb_persons(role=role).tolist() == [expected_count]
        assert household.sum(values, role=role).tolist() == [expected_sum]
