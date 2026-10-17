"""Property: group reductions and ranks match a plain-Python oracle.

For any number of groups (1 to 6), any assignment of 0 to 30 persons to them
(so groups may be empty, and there may be no persons at all), given as an
integer array or a plain list, and any values:

* ``sum``, ``nb_persons``, ``any``, ``all``, ``max`` and ``min`` equal the
  oracle, with 0, False, True, -inf and inf for groups with no members;
* ``Population.get_rank`` gives each person the position of their value among
  their group's values, ties going to the person listed first.

Examples, including the empty-population witness, are in
``test_group_population_empty.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tests.core.test_group_population_empty import populations  # noqa: E402


@st.composite
def _groups(draw):
    count = draw(st.integers(1, 6))
    members_entity_id = draw(st.lists(st.integers(0, count - 1), max_size=30))
    values = draw(
        st.lists(
            st.integers(-5, 5),
            min_size=len(members_entity_id),
            max_size=len(members_entity_id),
        )
    )
    as_list = draw(st.booleans())
    return count, members_entity_id, values, as_list


@settings(max_examples=300, deadline=None)
@given(_groups())
def test_reductions_and_ranks_match_oracle(case):
    count, members_entity_id, values, as_list = case
    persons, households = populations(members_entity_id, count)
    if as_list:
        # Memberships may come as a plain list; an empty one is float.
        households.members_entity_id = list(members_entity_id)
    array = np.array(values, dtype=float)

    members = {group: [] for group in range(count)}
    for person, group in enumerate(members_entity_id):
        members[group].append(person)
    group_values = [[values[person] for person in members[g]] for g in range(count)]

    assert households.sum(array).tolist() == [sum(v) for v in group_values]
    assert households.nb_persons().tolist() == [len(v) for v in group_values]
    assert households.any(array > 0).tolist() == [
        any(x > 0 for x in v) for v in group_values
    ]
    assert households.all(array > 0).tolist() == [
        all(x > 0 for x in v) for v in group_values
    ]
    assert households.max(array).tolist() == [
        max(v, default=-np.inf) for v in group_values
    ]
    assert households.min(array).tolist() == [
        min(v, default=np.inf) for v in group_values
    ]

    expected_rank = [0] * len(values)
    for group in range(count):
        ordered = sorted(members[group], key=lambda person: values[person])
        for rank, person in enumerate(ordered):
            expected_rank[person] = rank
    assert persons.get_rank(households, array).tolist() == expected_rank
