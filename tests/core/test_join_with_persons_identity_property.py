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

Examples, including the audit witness, are in
``test_join_with_persons_identity.py``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

import hypothesis  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.simulations.simulation_builder import (  # noqa: E402
    group_positions,
)
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
