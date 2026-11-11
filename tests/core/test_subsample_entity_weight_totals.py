"""Every recorded weight column conserves its own entity's total.

The flat-file representation repeats group weights per person. Those repeated
values must not determine the total conserved by subsampling. The independent
reference below groups by the first surviving member, matching the loader even
when a second group partition crosses households.
"""

from collections import Counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from policyengine_core.data import Dataset
from policyengine_core.entities import build_entity
from policyengine_core.periods import ETERNITY, YEAR, period
from policyengine_core.simulations import Simulation
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.variables import Variable
from tests.core.test_user_input_keys import _store_on_disk

YEAR_INPUT = "2022"
Person = build_entity(key="person", plural="persons", label="Person", is_person=True)
Household = build_entity(
    key="household",
    plural="households",
    label="Household",
    roles=[{"key": "member", "plural": "members", "label": "Members"}],
)
Family = build_entity(
    key="family",
    plural="families",
    label="Family",
    roles=[{"key": "member", "plural": "members", "label": "Members"}],
)
ENTITIES = {"person": Person, "household": Household, "family": Family}
WEIGHT_YEARS = (YEAR_INPUT, "2023")


def weight_simulation(data):
    """Load only leaf inputs into a small three-entity system."""
    system = TaxBenefitSystem(list(ENTITIES.values()))
    for name, entity, period in (
        ("person_id", Person, ETERNITY),
        ("household_id", Household, ETERNITY),
        ("person_household_id", Person, ETERNITY),
        ("family_id", Family, ETERNITY),
        ("person_family_id", Person, ETERNITY),
        ("person_weight", Person, YEAR),
        ("household_weight", Household, YEAR),
        ("family_weight", Family, YEAR),
    ):
        system.add_variable(
            type(
                name,
                (Variable,),
                {
                    "value_type": float if period == YEAR else int,
                    "entity": entity,
                    "definition_period": period,
                    "label": name.replace("_", " "),
                },
            )
        )
    return Simulation(
        tax_benefit_system=system,
        dataset=Dataset.from_dataframe(data, YEAR_INPUT),
    )


def weight_data(
    households, families, household_weights, family_weights, person_weights
):
    data = pd.DataFrame(
        {
            "person_id__ETERNITY": np.arange(len(households)) + 101,
            "household_id__ETERNITY": households,
            "person_household_id__ETERNITY": households,
            "family_id__ETERNITY": families,
            "person_family_id__ETERNITY": families,
        }
    )
    for year_index, year in enumerate(WEIGHT_YEARS):
        # Distinct period distributions expose accidental reuse of one column's
        # total or scale factor when rescaling all recorded weight inputs.
        data[f"household_weight__{year}"] = data["household_id__ETERNITY"].map(
            {
                group: weight * (1 + year_index * (index % 3 + 1))
                for index, (group, weight) in enumerate(
                    sorted(household_weights.items())
                )
            }
        )
        data[f"family_weight__{year}"] = data["family_id__ETERNITY"].map(
            {
                group: weight * (1 + year_index * (index % 2 + 1))
                for index, (group, weight) in enumerate(sorted(family_weights.items()))
            }
        )
        data[f"person_weight__{year}"] = np.asarray(person_weights) * (
            1 + year_index * (np.arange(len(households)) % 4 + 1)
        )
    return data


def assert_weight_totals_and_reference(data, chosen, quantize_weights):
    simulation = weight_simulation(data)
    originals = {
        (entity, year): simulation.calculate(f"{entity}_weight", year).sum()
        for entity in ENTITIES
        for year in WEIGHT_YEARS
    }
    with patch("numpy.random.choice", return_value=np.asarray(chosen)):
        simulation.subsample(
            n=len(chosen),
            seed="entity-weight-totals",
            time_period=YEAR_INPUT,
            quantize_weights=quantize_weights,
        )

    retained = data[data["household_id__ETERNITY"].isin(chosen)].copy()
    counts = retained["household_id__ETERNITY"].map(Counter(chosen))
    for entity in ENTITIES:
        # Native entity order is independent of the flat-file person order.
        entity_ids = simulation.calculate(f"{entity}_id", ETERNITY)
        for year in WEIGHT_YEARS:
            column = f"{entity}_weight__{year}"
            actual = simulation.calculate(f"{entity}_weight", year)
            target = originals[(entity, year)]
            assert actual.sum() == pytest.approx(target, rel=2e-6)

            # This reference uses pandas group labels, rather than simulation
            # population indices or member counts. A family can lose its first
            # source member and retain members with unequal household counts.
            retained["candidate_weight"] = (
                counts if quantize_weights else retained[column] * counts
            )
            reference = retained.groupby(f"{entity}_id__ETERNITY", sort=True)[
                "candidate_weight"
            ].first()
            reference *= target / reference.sum()
            np.testing.assert_allclose(
                actual, reference.loc[entity_ids].to_numpy(), rtol=2e-6
            )


@pytest.mark.parametrize("quantize_weights", [False, True])
@pytest.mark.parametrize("chosen", [[20, 40, 20], [30, 20, 20]])
def test_subsample_preserves_native_entity_weight_totals(chosen, quantize_weights):
    data = weight_data(
        households=[30, 10, 20, 20, 30, 40, 40, 20, 40],
        families=[110, 110, 220, 110, 330, 330, 220, 220, 440],
        household_weights={10: 100.0, 20: 200.0, 30: 300.0, 40: 400.0},
        family_weights={110: 5.0, 220: 11.0, 330: 17.0, 440: 23.0},
        person_weights=np.arange(2.0, 11.0),
    )
    assert_weight_totals_and_reference(data, chosen, quantize_weights)


@pytest.mark.parametrize("quantize_weights", [False, True])
def test_subsample_normalizes_using_exported_membership(quantize_weights):
    data = weight_data(
        households=[30, 10, 20, 20, 30, 40, 40, 20, 40],
        families=[110, 110, 220, 110, 330, 330, 220, 220, 440],
        household_weights={10: 100.0, 20: 200.0, 30: 300.0, 40: 400.0},
        family_weights={110: 5.0, 220: 11.0, 330: 17.0, 440: 23.0},
        person_weights=np.arange(2.0, 11.0),
    )
    simulation = weight_simulation(data)
    exported_memberships = np.array([110, 110, 220, 220, 330, 330, 330, 220, 440])
    simulation.set_input("person_family_id", ETERNITY, exported_memberships)

    # set_input changes the recorded column without changing loaded populations.
    # The retained rows contain three exported families, but four old families.
    chosen = [20, 40, 20]
    retained = data[data["household_id__ETERNITY"].isin(chosen)].copy()
    retained["person_family_id__ETERNITY"] = exported_memberships[retained.index]
    counts = retained["household_id__ETERNITY"].map(Counter(chosen))
    np.testing.assert_array_equal(
        simulation.to_input_dataframe()["person_family_id__ETERNITY"],
        exported_memberships,
    )
    with patch("numpy.random.choice", return_value=np.asarray(chosen)):
        simulation.subsample(
            n=len(chosen),
            seed="exported-membership",
            time_period=YEAR_INPUT,
            quantize_weights=quantize_weights,
        )

    for entity in ENTITIES:
        entity_ids = simulation.calculate(f"{entity}_id", ETERNITY)
        membership_column = (
            "person_id__ETERNITY"
            if entity == "person"
            else f"person_{entity}_id__ETERNITY"
        )
        for year in WEIGHT_YEARS:
            column = f"{entity}_weight__{year}"
            # Each period's target comes independently from the source's native
            # entity groups, before the membership override or subsampling.
            target = data.groupby(f"{entity}_id__ETERNITY")[column].first().sum()
            candidate = (
                counts.astype(float) if quantize_weights else retained[column] * counts
            )
            reference = candidate.groupby(retained[membership_column]).first()
            reference *= target / reference.sum()
            actual = simulation.calculate(f"{entity}_weight", year)
            assert actual.sum() == pytest.approx(target, rel=2e-6)
            np.testing.assert_allclose(
                actual, reference.loc[entity_ids].to_numpy(), rtol=2e-6
            )


@pytest.mark.parametrize("quantize_weights", [False, True])
def test_subsample_conserves_recorded_disk_weight_total_under_memory_cache(
    quantize_weights,
):
    data = weight_data(
        households=[10, 10, 20, 30, 30],
        families=[110, 110, 220, 330, 110],
        household_weights={10: 100.0, 20: 200.0, 30: 300.0},
        family_weights={110: 7.0, 220: 11.0, 330: 13.0},
        person_weights=np.arange(2.0, 7.0),
    )
    simulation = weight_simulation(data)
    simulation.delete_arrays("family_weight")
    _store_on_disk(simulation, "family_weight")
    source_weights = data.groupby("family_id__ETERNITY").first()
    holder = simulation.get_holder("family_weight")
    for year in WEIGHT_YEARS:
        values = source_weights[f"family_weight__{year}"].to_numpy()
        simulation.set_input("family_weight", year, values)
        np.testing.assert_array_equal(holder._disk_storage.get(period(year)), values)

    simulation.memory_config.max_memory_occupation_pc = 101
    for year in WEIGHT_YEARS:
        # An unmarked memory cache can shadow a surviving disk input in reads.
        # Normalization must preserve the recorded source total from disk.
        shadow = np.full(len(source_weights), 999.0)
        holder._set(period(year), shadow, derived=False, is_input=False)
        assert not holder._stores_user_input(period(year), "default", "memory")
        assert holder._stores_user_input(period(year), "default", "disk")
        np.testing.assert_array_equal(
            simulation.calculate("family_weight", year), shadow
        )

    chosen = [20, 30, 20]
    with patch("numpy.random.choice", return_value=np.asarray(chosen)):
        simulation.subsample(
            n=len(chosen),
            seed="recorded-disk-weight-total",
            time_period=YEAR_INPUT,
            quantize_weights=quantize_weights,
        )

    retained = data[data["household_id__ETERNITY"].isin(chosen)]
    counts = retained["household_id__ETERNITY"].map(Counter(chosen))
    entity_ids = simulation.calculate("family_id", ETERNITY)
    for year in WEIGHT_YEARS:
        column = f"family_weight__{year}"
        target = source_weights[column].sum()
        candidate = (
            counts.astype(float) if quantize_weights else retained[column] * counts
        )
        reference = candidate.groupby(retained["person_family_id__ETERNITY"]).first()
        reference *= target / reference.sum()
        actual = simulation.calculate("family_weight", year)
        assert actual.sum() == pytest.approx(target, rel=2e-6)
        np.testing.assert_allclose(
            actual, reference.loc[entity_ids].to_numpy(), rtol=2e-6
        )


@pytest.mark.parametrize("quantize_weights", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_entity_total_invariant_across_deterministic_random_partitions(
    seed, quantize_weights
):
    rng = np.random.default_rng(seed)
    households = np.repeat([10, 20, 30, 40], rng.integers(1, 5, size=4))
    rng.shuffle(households)
    families = rng.choice([110, 220, 330], size=len(households))
    family_ids = np.unique(families)
    data = weight_data(
        households=households,
        families=families,
        household_weights=dict(zip([10, 20, 30, 40], rng.integers(1, 100, 4))),
        family_weights=dict(zip(family_ids, rng.integers(1, 100, len(family_ids)))),
        person_weights=rng.integers(1, 100, len(households)),
    )
    chosen = rng.choice([10, 20, 30, 40], size=5, replace=True)
    assert_weight_totals_and_reference(data, chosen, quantize_weights)


@pytest.mark.parametrize("quantize_weights", [False, True])
def test_zero_total_input_weight_column_remains_zero(quantize_weights):
    data = weight_data(
        households=[10, 10, 20],
        families=[110, 110, 220],
        household_weights={10: 100, 20: 200},
        family_weights={110: 0, 220: 0},
        person_weights=[0, 0, 0],
    )
    simulation = weight_simulation(data)
    simulation.subsample(n=2, seed="zero-column", quantize_weights=quantize_weights)
    for entity in ("person", "family"):
        for year in WEIGHT_YEARS:
            np.testing.assert_array_equal(
                simulation.calculate(f"{entity}_weight", year), 0
            )


@pytest.mark.parametrize("quantize_weights", [False, True])
def test_positive_total_with_only_zero_retained_weights(quantize_weights):
    data = weight_data(
        households=[10, 20],
        families=[110, 220],
        household_weights={10: 1, 20: 1},
        family_weights={110: 1, 220: 1},
        person_weights=[0, 5],
    )
    simulation = weight_simulation(data)
    original_dataset = simulation.dataset
    with patch("numpy.random.choice", return_value=np.array([10])):
        if quantize_weights:
            simulation.subsample(n=1, seed="zero-retained", quantize_weights=True)
            assert simulation.calculate("person_weight", YEAR_INPUT).sum() == 5
        else:
            with pytest.raises(ValueError, match="person_weight__2022.*zero weight"):
                simulation.subsample(n=1, seed="zero-retained", quantize_weights=False)
            assert simulation.dataset is original_dataset
            np.testing.assert_array_equal(
                simulation.calculate("person_weight", YEAR_INPUT), [0, 5]
            )
