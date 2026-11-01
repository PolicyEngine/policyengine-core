"""Properties of flat-file export and of mapping text between entities."""

import numpy as np
import pandas as pd
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from tests.core.test_flat_file_datasets import build as build_flat
from tests.core.test_map_result_categorical import build as build_groups

memberships = st.integers(min_value=1, max_value=8).flatmap(
    lambda size: st.tuples(
        st.lists(st.integers(1, 4), min_size=size, max_size=size),
        st.lists(st.integers(1, 4), min_size=size, max_size=size),
    )
)


@hypothesis.given(memberships)
@hypothesis.settings(deadline=None, max_examples=60)
def test_a_person_string_maps_to_a_group_exactly_when_its_members_share_it(partition):
    """Members who agree give their value, and members who disagree raise."""
    households, families = partition
    simulation = build_groups(households, families)
    # Each household's own label, so families that cross households disagree.
    labels = np.array([f"h{household}" for household in households], dtype=object)
    family_ids = np.asarray(families)
    agrees = all(
        len(set(labels[family_ids == family])) == 1 for family in set(families)
    )
    if agrees:
        mapped = simulation.map_result(labels, "person", "family")
        expected = {family: label for family, label in zip(families, labels)}
        assert list(mapped) == [expected[family] for family in sorted(set(families))]
    else:
        with pytest.raises(ValueError, match="different values"):
            simulation.map_result(labels, "person", "family")


@hypothesis.given(
    st.lists(
        st.tuples(
            st.integers(1, 3),
            st.floats(0, 120, width=32),
            st.booleans(),
            st.booleans(),
        ),
        min_size=1,
        max_size=8,
    ),
    st.integers(1, 1000),
)
@hypothesis.settings(deadline=None, max_examples=60)
def test_an_export_reloads_to_the_same_answers(rows, first_id):
    """Export then reload changes neither the people nor what they were given."""
    households, ages, heads, roles = map(list, zip(*rows))
    data = pd.DataFrame(
        {
            "person_id": [first_id + index for index in range(len(rows))],
            "household_id": households,
            "family_id": households,
            "age": ages,
            "is_head": heads,
            "observed_role": roles,
        }
    )
    source = build_flat(data)
    reloaded = build_flat(source.to_input_dataframe())
    assert reloaded.persons.ids.tolist() == source.persons.ids.tolist()
    assert (
        reloaded.populations["household"].count == source.populations["household"].count
    )
    for name in ("age", "is_head", "observed_role"):
        assert (
            reloaded.calculate(name, 2024).tolist()
            == source.calculate(name, 2024).tolist()
        )
