"""Subsampling preserves a flat-file partition recorded by membership IDs."""

from unittest.mock import patch

import numpy as np
import pytest

from policyengine_core.country_template import Simulation
from policyengine_core.data import Dataset
from tests.fixtures.subsample_inputs import DATASET_YEAR, build, dataframe, stored


def partition(simulation):
    """Person IDs grouped by the population's actual household membership."""
    members = simulation.household.members_entity_id
    return {
        frozenset(simulation.persons.ids[members == index].tolist())
        for index in range(simulation.household.count)
    }


@pytest.mark.parametrize("nonconsecutive_ids", [False, True])
@pytest.mark.parametrize("quantize_weights", [False, True])
@pytest.mark.parametrize("n", [1, 4])
def test_missing_input_household_id_uses_recorded_membership(
    nonconsecutive_ids, quantize_weights, n
):
    data = dataframe().drop(columns=[f"household_id__{DATASET_YEAR}"])
    membership_column = f"person_household_id__{DATASET_YEAR}"
    if nonconsecutive_ids:
        data[membership_column] = data[membership_column].map(
            {1: 901, 2: -20, 3: 33, 4: 88, 5: 205, 6: 411}
        )
        data = data.iloc[[9, 0, 4, 6, 2, 8, 1, 5, 3, 7]].reset_index(drop=True)

    def make_simulation():
        return Simulation(
            tax_benefit_system=build().tax_benefit_system,
            dataset=Dataset.from_dataframe(data, DATASET_YEAR),
        )

    fresh, used = make_simulation(), make_simulation()
    source_partition = partition(fresh)
    assert fresh.household.count == 6
    used.calculate("household_id", DATASET_YEAR)

    choice = np.random.choice
    for simulation in (fresh, used):
        selected = {}

        def choose_households(*args, **kwargs):
            selected["ids"] = choice(*args, **kwargs)
            return selected["ids"]

        with patch("numpy.random.choice", side_effect=choose_households):
            simulation.subsample(
                n=n,
                seed="recorded-memberships",
                time_period=DATASET_YEAR,
                quantize_weights=quantize_weights,
            )
        # The resulting inputs select whole recorded groups, without adding
        # the calculated default household ID.
        sampled = simulation.dataset.load()
        selected_ids = set(selected["ids"])
        assert set(sampled["person_household_id__ETERNITY"]) == selected_ids
        expected_people = set(
            data.loc[
                data[membership_column].isin(selected_ids), f"person_id__{DATASET_YEAR}"
            ]
        )
        assert set(simulation.persons.ids) == expected_people
        assert simulation.household.count == len(selected_ids)
        assert 1 <= len(selected_ids) <= n
        assert partition(simulation) <= source_partition
        assert not any(column.startswith("household_id__") for column in sampled)
        assert not any(name == "household_id" for name, _, _ in stored(simulation))
        assert simulation.default_calculation_period == DATASET_YEAR
        expected_totals = 2 * data.groupby(membership_column)[
            f"base__{DATASET_YEAR}"
        ].transform("sum")
        totals_by_person = dict(
            zip(data[f"person_id__{DATASET_YEAR}"], expected_totals)
        )
        np.testing.assert_array_equal(
            simulation.calculate("household_total", DATASET_YEAR, map_to="person"),
            [totals_by_person[person_id] for person_id in simulation.persons.ids],
        )

    assert stored(used) == stored(fresh)
    assert partition(used) == partition(fresh)
    np.testing.assert_array_equal(
        used.calculate("household_weight", DATASET_YEAR),
        fresh.calculate("household_weight", DATASET_YEAR),
    )
