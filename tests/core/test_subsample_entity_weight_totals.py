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
from policyengine_core.periods import ETERNITY, YEAR
from policyengine_core.simulations import Simulation
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.variables import Variable

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
