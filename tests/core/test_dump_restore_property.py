"""Property: a dumped simulation restores with the same structure and values.

For any number of persons (0 to 6) and of households (0 to 4, at least one
when there are persons), any assignment of persons to households (so a
household may have no members, first, last or in between, and there may be
no one at all), any roles, ids given as integers or strings or never set (as
``scripts/simulation_generator.py`` leaves them for groups), any inputs of
each value type the country template has (float, int, date, Enum and str,
monthly and ETERNITY), and any calculations before the dump:

1. **Structure.** Every population's count and ids, and every group's
   memberships, positions and roles, are restored, including zero counts.
2. **Values.** Every holder stores the same values, byte for byte, with the
   same derived marks.
3. **Dumps without counts.** With the count files removed, as in dumps
   written before they were, restoring gives the same structure and values
   whenever the ids and memberships give the counts: when the ids are set,
   one per instance, as ``SimulationBuilder`` sets them, or when the last
   household has members. (Otherwise only the count file gives the count.)
4. **Same answers.** Any calculation on the restored simulation gives what
   it gives on the dumped one, including any error.

Examples are in ``test_dump_restore.py``.
"""

from __future__ import annotations

import datetime
import pathlib
import tempfile

import numpy as np
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.country_template import CountryTaxBenefitSystem  # noqa: E402
from policyengine_core.enums import EnumArray  # noqa: E402
from policyengine_core.simulations import SimulationBuilder  # noqa: E402
from policyengine_core.tools.simulation_dumper import (  # noqa: E402
    COUNT_FILE,
    dump_simulation,
    restore_simulation,
)

SYSTEM = CountryTaxBenefitSystem()
ROLES = SYSTEM.entities_by_singular()["household"].flattened_roles
MONTHS = ["2018-01", "2018-02"]

_floats = st.floats(-1e5, 1e5, allow_nan=False, width=32)
_dates = st.dates(datetime.date(1900, 1, 1), datetime.date(2030, 12, 31)).map(
    np.datetime64
)
_statuses = st.sampled_from(["owner", "tenant", "free_lodger", "homeless"])
# Variable, entity key, periods, element strategy and array dtype.
INPUTS = [
    ("salary", "person", MONTHS, _floats, float),
    ("age", "person", MONTHS, st.integers(0, 120), int),
    ("birth", "person", ["ETERNITY"], _dates, "datetime64[D]"),
    ("rent", "household", MONTHS, _floats, float),
    ("housing_occupancy_status", "household", MONTHS, _statuses, str),
    ("postal_code", "household", MONTHS, st.text("ab12", max_size=5), str),
]
CALCULATIONS = [
    "disposable_income",
    "basic_income",
    "income_tax",
    "household_income",
    "housing_allowance",
    "parenting_allowance",
    "housing_tax",
]


@st.composite
def _simulations(draw):
    person_count = draw(st.integers(0, 6))
    household_count = draw(st.integers(1 if person_count else 0, 4))
    memberships = draw(
        st.lists(
            st.integers(0, max(household_count - 1, 0)),
            min_size=person_count,
            max_size=person_count,
        )
    )
    roles = draw(
        st.lists(st.sampled_from(ROLES), min_size=person_count, max_size=person_count)
    )
    id_kind = draw(st.sampled_from(["int", "str", "unset"]))
    counts = {"person": person_count, "household": household_count}
    inputs = []
    for _ in range(draw(st.integers(0, 6))):
        name, entity, input_periods, elements, dtype = draw(st.sampled_from(INPUTS))
        values = draw(
            st.lists(elements, min_size=counts[entity], max_size=counts[entity])
        )
        inputs.append(
            (name, draw(st.sampled_from(input_periods)), np.array(values, dtype=dtype))
        )
    calculations = draw(
        st.lists(st.tuples(st.sampled_from(CALCULATIONS), st.sampled_from(MONTHS)))
    )
    return (
        person_count,
        household_count,
        memberships,
        roles,
        id_kind,
        inputs,
        calculations,
    )


def _build(person_count, household_count, memberships, roles, id_kind):
    """A simulation with this structure, set as ``SimulationBuilder`` sets one."""
    simulation = SimulationBuilder().build_default_simulation(
        SYSTEM, count=person_count
    )

    def ids(prefix, count):
        if id_kind == "str":
            return np.array([f"{prefix}{index}" for index in range(count)], dtype=str)
        if id_kind == "unset":
            return []
        return np.arange(count, dtype=np.int64)

    simulation.persons.ids = ids("p", person_count)
    households = simulation.household
    households.count = household_count
    households.ids = ids("h", household_count)
    households.members_entity_id = np.array(memberships, dtype=np.int64)
    households.members_role = np.array(roles, dtype=object)
    return simulation


def _result(function):
    try:
        value = function()
    except Exception as error:  # Compare failures as well as values.
        return ("error", type(error).__name__, str(error))
    if isinstance(value, EnumArray):
        value = value.view(np.ndarray)
    value = np.asarray(value)
    return ("array", value.dtype.str, value.shape, value.tobytes())


def _structure(simulation):
    structure = {}
    for key, population in simulation.populations.items():
        ids = np.asarray(population.ids)
        entry = {"count": population.count, "ids": (ids.dtype.str, ids.tolist())}
        if not population.entity.is_person:
            for name in ("members_entity_id", "members_position"):
                array = np.asarray(getattr(population, name))
                entry[name] = (array.dtype.kind in "iu", array.tolist())
            entry["members_role"] = [role.key for role in population.members_role]
        structure[key] = entry
    return structure


def _stored(simulation):
    """Every value and its derived mark, as bytes, keyed by variable and period.

    An Enum value also keeps its enum class.
    """
    stored = {}
    for population in simulation.populations.values():
        for name, holder in population._holders.items():
            for period in holder.get_known_periods():
                value = holder.get_array(period)
                array_type = type(value).__name__
                possible_values = None
                if isinstance(value, EnumArray):
                    possible_values = value.possible_values
                    value = value.view(np.ndarray)
                stored[(name, str(period))] = (
                    array_type,
                    possible_values,
                    value.dtype.str,
                    value.shape,
                    value.tobytes(),
                    holder.is_derived(period),
                )
    return stored


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(case=_simulations(), without_counts=st.booleans())
def test_dump_then_restore_keeps_structure_and_values(case, without_counts):
    *structure, inputs, calculations = case
    person_count, household_count, memberships, _, id_kind = structure
    last_household_empty = (
        household_count > 0 and household_count - 1 not in memberships
    )
    assume(not (without_counts and id_kind == "unset" and last_household_empty))
    dumped = _build(*structure)
    for name, period, values in inputs:
        dumped.set_input(name, period, values)
    for request in calculations:
        _result(lambda: dumped.calculate(*request))

    with tempfile.TemporaryDirectory(prefix="core-dump-") as directory:
        dump_directory = pathlib.Path(directory) / "dump"
        dump_simulation(dumped, str(dump_directory))
        count_files = list(dump_directory.glob(f"__entities__/*/{COUNT_FILE}"))
        assert len(count_files) == len(dumped.populations)
        for path in count_files:
            assert np.load(path).dtype == np.int64
        if without_counts:
            for path in count_files:
                path.unlink()
        restored = restore_simulation(str(dump_directory), SYSTEM)

    # 1. Structure, and 3. without counts.
    assert _structure(restored) == _structure(dumped)
    # 2. Values.
    assert _stored(restored) == _stored(dumped)
    # 4. Same answers.
    for name in CALCULATIONS:
        for period in MONTHS:
            assert _result(lambda: restored.calculate(name, period)) == _result(
                lambda: dumped.calculate(name, period)
            ), (name, period)
