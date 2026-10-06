"""Property: a dataset without role columns gives every person one role.

For any persons, any assignment of them to the groups of several group
entities (empty groups included), any of the three dataset formats and any
``Simulation.default_role``, building the simulation must give, for every
group entity:

* ``members_role`` with one entry per person, like ``members_entity_id``;
* every member holding the same role of that entity: the role the default
  names, the first subrole of a role with subroles it names, or else the
  entity's first role;
* role queries shaped like their non-role counterparts (one cell per group
  for ``nb_persons``, ``sum`` and ``max``; one per person for ``project`` and
  ``has_role``), equal to them for every role that contains the default role
  and to the empty result for every other role.

Examples, including the audit witness, are in ``test_dataset_default_role.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.data import Dataset  # noqa: E402
from policyengine_core.entities import build_entity  # noqa: E402
from policyengine_core.periods import ETERNITY  # noqa: E402
from policyengine_core.taxbenefitsystems import TaxBenefitSystem  # noqa: E402
from tests.core.test_dataset_default_role import (  # noqa: E402
    in_memory_dataset,
    simulate,
)

PERSON = build_entity(key="person", plural="persons", label="Person", is_person=True)
GROUP_ENTITIES = [
    # Shaped like the country template's household.
    build_entity(
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
    ),
    # A role with subroles that is not the first role.
    build_entity(
        key="family",
        plural="families",
        label="Family",
        roles=[
            {"key": "child", "plural": "children"},
            {"key": "parent", "plural": "parents", "subroles": ["mother", "father"]},
        ],
    ),
    # Shaped like the country models' groups: one role, "member".
    build_entity(
        key="unit",
        plural="units",
        label="Unit",
        roles=[{"key": "member", "plural": "members"}],
    ),
]
SYSTEM = TaxBenefitSystem([PERSON, *GROUP_ENTITIES])
DEFAULT_ROLES = [
    "member",
    "parent",
    "child",
    "first_parent",
    "second_parent",
    "mother",
    "father",
    "nobody",
]


def expected_default_role(entity, default_role):
    for role in entity.flattened_roles:
        if role.key == default_role:
            return role
    for role in entity.roles:
        if role.key == default_role and role.subroles:
            return role.subroles[0]
    return entity.flattened_roles[0]


@st.composite
def _datasets(draw):
    persons = draw(st.integers(min_value=0, max_value=12))
    data_format = draw(
        st.sampled_from([Dataset.ARRAYS, Dataset.TIME_PERIOD_ARRAYS, Dataset.FLAT_FILE])
    )
    columns = {"person_id": np.arange(100, 100 + persons)}
    for entity in GROUP_ENTITIES:
        if data_format == Dataset.FLAT_FILE:
            # A flat file declares groups only through their members.
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
            columns[f"{entity.key}_id"] = np.array(group_ids, dtype=int)
        columns[f"person_{entity.key}_id"] = np.array(
            draw(
                st.lists(
                    st.sampled_from(group_ids) if group_ids else st.nothing(),
                    min_size=persons,
                    max_size=persons,
                )
            ),
            dtype=int,
        )
    if data_format == Dataset.FLAT_FILE:
        dataset = Dataset.from_dataframe(pd.DataFrame(columns), "2024")
    elif data_format == Dataset.TIME_PERIOD_ARRAYS:
        dataset = in_memory_dataset(
            {name: {ETERNITY: values} for name, values in columns.items()},
            data_format,
        )
    else:
        dataset = in_memory_dataset(columns, data_format)
    values = draw(
        st.lists(
            st.integers(min_value=-1000, max_value=1000),
            min_size=persons,
            max_size=persons,
        )
    )
    return dataset, persons, np.array(values, dtype=float)


@settings(max_examples=200, deadline=None)
@given(_datasets(), st.sampled_from(DEFAULT_ROLES))
def test_every_person_holds_one_default_role(case, default_role):
    dataset, persons, values = case
    simulation = simulate(SYSTEM, dataset, default_role=default_role)

    for entity in GROUP_ENTITIES:
        group = simulation.populations[entity.key]
        role = expected_default_role(entity, default_role)

        # One role per person, and it is a role of this entity.
        assert len(group.members_role) == persons == len(group.members_entity_id)
        assert all(member_role is role for member_role in group.members_role)

        group_values = np.arange(group.count, dtype=float)
        for query_role in [*entity.roles, *entity.flattened_roles]:
            holds = query_role is role or role in (query_role.subroles or [])
            has_role = simulation.persons.has_role(query_role)
            nb_persons = group.nb_persons(role=query_role)
            total = group.sum(values, role=query_role)
            projected = group.project(group_values, role=query_role)

            assert has_role.shape == (persons,)
            assert nb_persons.shape == total.shape == (group.count,)
            assert projected.shape == (persons,)
            if holds:
                assert has_role.all()
                assert np.array_equal(nb_persons, group.nb_persons())
                assert np.array_equal(total, group.sum(values))
                assert np.array_equal(projected, group.project(group_values))
            else:
                assert not has_role.any()
                assert not nb_persons.any()
                assert not total.any()
                assert not projected.any()
            if persons:
                # ``reduce`` needs at least one person to size its loop.
                maximum = group.max(values, role=query_role)
                assert maximum.shape == (group.count,)
                if holds:
                    assert np.array_equal(maximum, group.max(values))
                else:
                    assert (maximum == -np.inf).all()
