"""Property: joining persons to groups keeps each person in their named group.

For any distinct group IDs (integers or strings) declared in any order, any
set of empty groups anywhere in that order, and any assignment of persons to
the occupied groups, ``join_with_persons`` must give a population where:

* each person's ``members_entity_id`` points at the group their ID names;
* per-group sums and member counts, in declared order, equal a plain-Python
  dict oracle;
* projecting group values gives each person their own group's value.

Numeric IDs also match by value when the declared and person IDs have
different dtypes (int32, int64, uint64, float64), including negative values
and values near 2**31, 2**53, 2**62 and beyond the int64 limit, where a
common float64 would merge neighbours.

Flat-file group values match each group's first member, including empty
frames. Default-role masks have one entry per person, and role-filtered
sums and projections match independent membership oracles.

Examples, including the audit witness, are in
``test_join_with_persons_identity.py``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

import hypothesis  # noqa: E402
from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.data import Dataset  # noqa: E402
from policyengine_core.entities import build_entity  # noqa: E402
from policyengine_core.simulations import Simulation  # noqa: E402
from policyengine_core.simulations.simulation_builder import (  # noqa: E402
    group_positions,
)
from policyengine_core.taxbenefitsystems import TaxBenefitSystem  # noqa: E402
from tests.core.test_join_with_persons_identity import _join  # noqa: E402


@st.composite
def _groups_and_members(draw):
    use_text = draw(st.booleans())
    distinct = draw(
        st.lists(
            st.text(alphabet="abcxyz", min_size=1, max_size=3)
            if use_text
            else st.integers(min_value=-1000, max_value=1000),
            min_size=1,
            max_size=10,
            unique=True,
        )
    )
    group_ids = draw(st.permutations(distinct))
    occupied = draw(st.lists(st.sampled_from(group_ids), min_size=1, unique=True))
    persons_group_ids = draw(st.lists(st.sampled_from(occupied), max_size=25))
    incomes = draw(
        st.lists(
            st.integers(min_value=-10_000, max_value=10_000),
            min_size=len(persons_group_ids),
            max_size=len(persons_group_ids),
        )
    )
    group_values = draw(
        st.lists(
            st.integers(min_value=-10_000, max_value=10_000),
            min_size=len(group_ids),
            max_size=len(group_ids),
        )
    )
    return group_ids, persons_group_ids, incomes, group_values


@settings(max_examples=300, deadline=None)
@given(_groups_and_members())
def test_join_matches_dict_oracle(tax_benefit_system, case):
    group_ids, persons_group_ids, incomes, group_values = case
    household = _join(tax_benefit_system, group_ids, persons_group_ids)

    # Identity: each person sits in the group their ID names.
    members = household.members_entity_id
    assert [group_ids[i] for i in members] == persons_group_ids

    # Per-group sums and counts, in declared order, match the oracle.
    totals = {group_id: 0 for group_id in group_ids}
    for group_id, income in zip(persons_group_ids, incomes):
        totals[group_id] += income
    counts = Counter(persons_group_ids)
    assert household.sum(np.array(incomes, dtype=float)).tolist() == [
        float(totals[group_id]) for group_id in group_ids
    ]
    assert household.nb_persons().tolist() == [
        counts[group_id] for group_id in group_ids
    ]

    # Projection gives each person their own group's value.
    value_of = dict(zip(group_ids, group_values))
    assert household.project(np.array(group_values)).tolist() == [
        value_of[group_id] for group_id in persons_group_ids
    ]


DTYPE_PAIRS = [
    (np.int64, np.int64),
    (np.int32, np.int64),
    (np.int64, np.uint64),
    (np.uint64, np.int64),
    (np.uint64, np.uint64),
    (np.int64, np.float64),
    (np.float64, np.int64),
    (np.uint64, np.float64),
    (np.float64, np.uint64),
]
#: Where runs of 32 consecutive IDs start: around 0, the int32 limit, the
#: last integers float64 holds exactly, and beyond the int64 limit.
BASES = [-(2**53) - 16, -16, 0, 2**31 - 16, 2**53 - 16, 2**62, 2**63 + 8]


def _exact_in(dtype, value):
    """Whether ``value`` survives a round trip through ``dtype``."""
    try:
        return int(np.array([value], dtype=dtype)[0]) == value
    except (OverflowError, ValueError):
        return False


@settings(max_examples=300, deadline=None)
@given(
    dtypes=st.sampled_from(DTYPE_PAIRS),
    offsets=st.lists(st.integers(0, 31), min_size=1, max_size=8, unique=True),
    data=st.data(),
)
def test_numeric_ids_match_by_value_across_dtypes(dtypes, offsets, data):
    group_dtype, person_dtype = dtypes
    base = data.draw(
        st.sampled_from(
            [
                base
                for base in BASES
                if all(
                    _exact_in(dtype, base) and _exact_in(dtype, base + 31)
                    for dtype in dtypes
                    if np.dtype(dtype).kind != "f"
                )
            ]
        )
    )
    group_values = [base + offset for offset in offsets]
    persons_values = data.draw(st.lists(st.sampled_from(group_values), max_size=12))
    hypothesis.assume(
        all(_exact_in(group_dtype, value) for value in group_values)
        and all(_exact_in(person_dtype, value) for value in persons_values)
    )

    positions = group_positions(
        np.array(group_values, dtype=group_dtype),
        np.array(persons_values, dtype=person_dtype),
    )

    assert [group_values[i] for i in positions] == persons_values


@st.composite
def _flat_file_rows(draw):
    memberships = draw(st.lists(st.integers(-10, 10), max_size=12))
    rents = draw(
        st.lists(
            st.integers(0, 10_000),
            min_size=len(memberships),
            max_size=len(memberships),
        )
    )
    return memberships, rents, draw(st.booleans()), draw(st.booleans())


@settings(max_examples=100, deadline=None)
@example(case=([], [], False, True))
@given(case=_flat_file_rows())
def test_flat_file_group_values_match_first_member_oracle(tax_benefit_system, case):
    memberships, rents, suffixed, explicit_memberships = case
    suffix = "__2024" if suffixed else ""
    columns = {
        f"person_id{suffix}": np.arange(len(memberships)),
        f"household_id{suffix}": np.array(memberships, dtype=np.int64),
        "rent__2024-01": np.array(rents, dtype=float),
    }
    if explicit_memberships:
        columns[f"person_household_id{suffix}"] = np.array(memberships, dtype=np.int64)
    simulation = Simulation(
        tax_benefit_system=tax_benefit_system,
        dataset=Dataset.from_dataframe(pd.DataFrame(columns), "2024"),
    )

    first_rent = {}
    for membership, rent in zip(memberships, rents):
        first_rent.setdefault(membership, rent)
    group_ids = sorted(first_rent)
    assert simulation.persons.count == len(memberships)
    assert simulation.populations["household"].count == len(group_ids)
    assert simulation.calculate("rent", "2024-01").tolist() == [
        first_rent[group_id] for group_id in group_ids
    ]
    assert simulation.calculate("rent", "2024-01", map_to="person").tolist() == [
        first_rent[membership] for membership in memberships
    ]


@settings(max_examples=100, deadline=None)
@example(memberships=[])
@example(memberships=[7, 7])
@given(memberships=st.lists(st.integers(-10, 10), max_size=20))
def test_default_role_masks_cover_persons_and_preserve_role_projection(memberships):
    person = build_entity("person", "persons", "", is_person=True)
    household = build_entity(
        "household", "households", "", roles=[{"key": "member", "plural": "members"}]
    )
    dataframe = pd.DataFrame(
        {
            "person_id__2024": np.arange(len(memberships)),
            "person_household_id__2024": np.array(memberships, dtype=np.int64),
        }
    )
    simulation = Simulation(
        tax_benefit_system=TaxBenefitSystem([person, household]),
        dataset=Dataset.from_dataframe(dataframe, "2024"),
    )

    population = simulation.populations["household"]
    assert population.members_role.shape == (len(memberships),)
    assert simulation.persons.has_role(household.MEMBER).tolist() == [
        True for _ in memberships
    ]
    counts = Counter(memberships)
    group_ids = sorted(counts)
    assert population.sum(
        np.ones(len(memberships)), role=household.MEMBER
    ).tolist() == [counts[group_id] for group_id in group_ids]
    values = np.arange(len(group_ids)) + 10
    value_of = dict(zip(group_ids, values.tolist()))
    assert population.project(values, role=household.MEMBER).tolist() == [
        value_of[membership] for membership in memberships
    ]
