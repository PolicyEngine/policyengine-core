"""Regression test: ``Simulation.subsample`` must invalidate ``_fast_cache``.

Before this fix, ``subsample`` called ``to_input_dataframe`` (which runs
calculations against the pre-subsample populations and writes results into
``_fast_cache``) and then rebuilt the populations via ``build_from_dataset``.
The rebuilt populations had smaller entity counts, but ``_fast_cache`` still
held the pre-subsample arrays. The short-circuit at the top of
``Simulation.calculate`` returned those stale arrays whenever a call passed
``decode_enums=False`` (the path used by ``Population.__call__`` — e.g. when
a formula does ``person.household("household_weight", period)``), producing
"size X != Y = count" projection errors that never surfaced through the
normal ``sim.calc(...)`` entry point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from policyengine_core.country_template import Microsimulation
from policyengine_core.country_template import Simulation as CountryTemplateSimulation
from policyengine_core.country_template.entities import Person
from policyengine_core.data import Dataset
from policyengine_core.model_api import Variable
from policyengine_core.periods import YEAR
from policyengine_core.periods import period as make_period


class person_id(Variable):
    value_type = int
    entity = Person
    definition_period = YEAR
    label = "Formula-backed person ID for subsample regression tests."

    def formula(person, period):
        return np.arange(person.count)


def _build_mini_dataset() -> Dataset:
    """Build a 5-household / 10-person in-memory dataset for subsample tests."""
    df = pd.DataFrame(
        {
            "person_id__2022": list(range(1, 11)),
            "household_id__2022": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "person_household_id__2022": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "household_weight__2022": [100.0] * 2
            + [200.0] * 2
            + [300.0] * 2
            + [400.0] * 2
            + [500.0] * 2,
            "salary__2022-01": [1000.0] * 10,
        }
    )
    return Dataset.from_dataframe(df, "2022")


def _build_formula_backed_id_simulation(
    isolated_tax_benefit_system,
) -> CountryTemplateSimulation:
    isolated_tax_benefit_system.replace_variable(person_id)
    return CountryTemplateSimulation(
        tax_benefit_system=isolated_tax_benefit_system,
        dataset=_build_mini_dataset(),
    )


def test_subsample_preserves_formula_backed_structural_ids(
    isolated_tax_benefit_system,
) -> None:
    """Subsampling needs IDs that safe public exports intentionally omit."""
    sim = _build_formula_backed_id_simulation(isolated_tax_benefit_system)

    safe_columns = sim.to_input_dataframe().columns
    full_columns = sim.to_input_dataframe(include_computed_variables=True).columns

    assert "person_id__2022" not in safe_columns
    assert "person_id__2022" in full_columns

    sim.subsample(n=1, seed="formula-backed-person-id")

    assert sim.populations["household"].count == 1
    assert sim.persons.count == 2


def test_subsample_clears_stale_fast_cache_entries() -> None:
    """A pre-subsample entry in ``_fast_cache`` must not survive subsample.

    We pre-populate ``_fast_cache`` with an array sized for the pre-subsample
    population (5). After ``subsample(n=2)`` rebuilds populations at size 2,
    a surviving 5-sized entry would be returned by the fast-path short
    circuit in ``calculate`` whenever a caller passes ``decode_enums=False``
    — the exact path that triggers the "size X != Y = count" projection
    error in downstream code.
    """
    sim = Microsimulation(dataset=_build_mini_dataset())
    assert sim.populations["household"].count == 5

    # Stuff the fast cache with a pre-subsample-sized array at a period the
    # dataset doesn't know about. A period inside the dataset's known
    # periods would get overwritten (or popped via the ``invalidated_caches``
    # set) when ``subsample`` runs ``to_input_dataframe``; a period outside
    # survives that path, matching the real-world scenario where
    # policyengine-us loads a 2024 dataset and subsample's
    # ``to_input_dataframe`` caches ``household_weight @ 2025`` (the
    # ``default_calculation_period``) at the pre-subsample size.
    stale_period = make_period("2023-01")
    sim._fast_cache[("salary", stale_period)] = np.arange(10, dtype=float)

    sim.subsample(n=2)

    assert sim.populations["household"].count == 2
    assert ("salary", stale_period) not in sim._fast_cache, (
        "subsample() left a pre-subsample-sized entry in _fast_cache; the "
        "fast-path short-circuit in Simulation.calculate will return this "
        "stale array whenever a caller passes decode_enums=False."
    )
