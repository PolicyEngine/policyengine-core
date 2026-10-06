"""Known periods a branch cannot read.

Holder storage keys embed the branch name (``"no_salt:2018"`` in memory,
``"no_salt_2018"`` on disk), and a holder can hold keys for branches other
than the simulation's own. ``Holder.get_known_periods()`` lists the periods of
every key, while ``Holder.get_array(period, branch_name)`` reads only that
branch, its ``parent_branch`` ancestors and ``default``.
``Holder.get_known_periods(branch_name)`` lists exactly the periods
``get_array`` reads for that branch, each once. Three defects came from the
gap between the two:

* ``Simulation._calculate`` checked the unscoped list for stored periods. When
  it took the latest earlier period from it, a period stored only under a
  branch the simulation cannot read gave ``None``: uprating raised
  ``TypeError`` (``None * factor``) and auto-carry-over cached ``NaN``. Since
  both read only inputs the branch reads (``Holder.get_input_periods``), such
  a period still made the simulation leave a default uncached that it would
  otherwise cache.
* ``OnDiskStorage`` parsed ``f"{branch}_{period}"`` keys with
  ``split("_")[1]``. A branch name containing ``_`` raised ``ValueError``
  (``no_salt`` -> period ``"salt"``) or listed the wrong period
  (``y_2019`` -> ``2019``). ``delete(None, "pre_tcja")`` also wiped
  ``pre_tcja_ctc``, whose key shares the prefix.
* ``dump_simulation`` read every listed period under ``default``, saving
  ``None`` for a period stored only on the dumped branch, which
  ``restore_simulation`` could not load, and the derived marks of the
  ``default`` branch's values rather than those of the values dumped.

``test_known_periods_branch_visibility_property.py`` checks the listing
against ``get_array`` and ``get_input_periods`` for random stored values.
"""

import itertools
import math
import os
import tempfile
import warnings

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.experimental import MemoryConfig
from policyengine_core.model_api import YEAR, Variable
from policyengine_core.parameters import ParameterNode
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.tools import simulation_dumper

INDEX_START = 2015
# The index grows 10% a year, so ratios between years are powers of 1.1.
INDEX = {
    f"{year}-01-01": 100 * 1.1 ** (year - INDEX_START) for year in range(2015, 2023)
}


def growth(from_year: int, to_year: int) -> float:
    return 1.1 ** (to_year - from_year)


def build_system(auto_carry_over: bool = False) -> CountryTaxBenefitSystem:
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = auto_carry_over
    system.parameters.add_child(
        "test_uprating",
        ParameterNode("test_uprating", data={"index": {"values": INDEX}}),
    )

    class uprated_income(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Uprated yearly income"
        uprating = "test_uprating.index"

    class carried_income(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Yearly income carried over without uprating"

    system.add_variable(uprated_income)
    system.add_variable(carried_income)
    return system


@pytest.fixture(scope="module")
def system():
    return build_system()


@pytest.fixture(scope="module")
def carry_over_system():
    return build_system(auto_carry_over=True)


def new_simulation(system):
    return SimulationBuilder().build_default_simulation(system, count=1)


def store(simulation, variable, year, value, branch_name):
    """Store ``value`` for ``year`` under ``branch_name`` in the holder of
    ``simulation``, the way ``Holder.set_input`` does when a caller names a
    branch other than the simulation's own."""
    simulation.get_holder(variable).set_input(
        periods.period(year), np.array([value]), branch_name
    )


def only(array) -> float:
    (value,) = array
    return float(value)


# ----- Simulation._calculate --------------------------------------------------


def test_uprating_skips_period_stored_only_on_unrelated_branch(system):
    simulation = new_simulation(system)
    store(simulation, "uprated_income", 2016, 1_000.0, "default")
    store(simulation, "uprated_income", 2018, 5_000.0, "other")

    result = simulation.calculate("uprated_income", 2020)

    # Uprated from 2016, the latest year the default branch can read.
    assert only(result) == pytest.approx(1_000 * growth(2016, 2020))


def test_uprating_with_no_readable_earlier_period_uses_default(system):
    simulation = new_simulation(system)
    store(simulation, "uprated_income", 2018, 5_000.0, "other")

    assert only(simulation.calculate("uprated_income", 2020)) == 0


def test_uprating_in_branch_skips_sibling_branch_period(system):
    simulation = new_simulation(system)
    simulation.set_input("uprated_income", 2016, [1_000.0])
    branch = simulation.get_branch("reform")
    store(branch, "uprated_income", 2018, 5_000.0, "baseline")

    result = branch.calculate("uprated_income", 2020)

    assert only(result) == pytest.approx(1_000 * growth(2016, 2020))


def test_uprating_in_nested_branch_reads_ancestor_periods(system):
    simulation = new_simulation(system)
    simulation.set_input("uprated_income", 2016, [1_000.0])
    itemizing = simulation.get_branch("itemizing")
    itemizing.set_input("uprated_income", 2018, [5_000.0])
    no_salt = itemizing.get_branch("no_salt")

    result = no_salt.calculate("uprated_income", 2020)

    assert only(result) == pytest.approx(5_000 * growth(2018, 2020))


def test_carry_over_skips_period_stored_only_on_unrelated_branch(
    carry_over_system,
):
    simulation = new_simulation(carry_over_system)
    store(simulation, "carried_income", 2016, 1_000.0, "default")
    store(simulation, "carried_income", 2018, 5_000.0, "other")

    result = simulation.calculate("carried_income", 2020)

    assert not math.isnan(only(result))
    assert only(result) == 1_000


def test_carry_over_with_no_readable_period_uses_default(carry_over_system):
    simulation = new_simulation(carry_over_system)
    store(simulation, "carried_income", 2018, 5_000.0, "other")

    result = simulation.calculate("carried_income", 2020)

    assert not math.isnan(only(result))
    assert only(result) == 0


def test_carry_over_ignores_later_period_on_unrelated_branch(carry_over_system):
    simulation = new_simulation(carry_over_system)
    store(simulation, "carried_income", 2016, 1_000.0, "default")
    store(simulation, "carried_income", 2022, 9_000.0, "other")

    assert only(simulation.calculate("carried_income", 2020)) == 1_000


# Invariant: values stored only under branches a simulation cannot read do not
# change what it calculates, nor what it stores where it reads. Checked
# exhaustively against a simulation that never had those values, for every
# combination of readable years, unreadable years and requested year below.
READABLE_YEAR_SETS = [(), (2016,), (2018,), (2016, 2018)]
UNREADABLE_YEAR_SETS = [
    years
    for size in range(1, 4)
    for years in itertools.combinations((2015, 2017, 2019, 2021), size)
]
REQUESTED_YEARS = (2017, 2018, 2019, 2020)


def _calculate_with(system, variable, readable, unreadable, requested):
    """The branch's result, and the (period, derived) pairs it then reads."""
    simulation = new_simulation(system)
    branch = simulation.get_branch("reform")
    for year in readable:
        store(branch, variable, year, 1_000.0 * (year - 2000), "reform")
    for year in unreadable:
        store(branch, variable, year, 7_777.0, "baseline")
    result = only(branch.calculate(variable, requested))
    holder = branch.get_holder(variable)
    read = sorted(
        (str(period), holder.is_derived(period, "reform"))
        for period in holder.get_known_periods("reform")
    )
    return result, read


@pytest.mark.parametrize("requested", REQUESTED_YEARS)
@pytest.mark.parametrize("unreadable", UNREADABLE_YEAR_SETS)
@pytest.mark.parametrize("readable", READABLE_YEAR_SETS)
@pytest.mark.parametrize(
    "variable, auto_carry_over",
    [("uprated_income", False), ("carried_income", True)],
)
def test_unreadable_branch_periods_do_not_change_results(
    system,
    carry_over_system,
    variable,
    auto_carry_over,
    readable,
    unreadable,
    requested,
):
    tax_benefit_system = carry_over_system if auto_carry_over else system

    with_unreadable = _calculate_with(
        tax_benefit_system, variable, readable, unreadable, requested
    )
    without_unreadable = _calculate_with(
        tax_benefit_system, variable, readable, (), requested
    )

    assert math.isfinite(with_unreadable[0])
    assert with_unreadable == without_unreadable


# ----- Holder.get_known_periods(branch_name) agrees with get_array ------------

# default -> a -> a_b is one lineage; c is a sibling of a, and a
# branch named "default" nested under a reads only "default" in get_array.
STORED_BRANCHES = ("default", "a", "a_b", "c")


def _lineage(system):
    simulation = new_simulation(system)
    a = simulation.get_branch("a")
    return {
        "default": simulation,
        "a": a,
        "a_b": a.get_branch("a_b"),
        "c": simulation.get_branch("c"),
        "default under a": a.get_branch("default"),
    }


@pytest.mark.parametrize("on_disk", [False, True])
@pytest.mark.parametrize(
    "stored",
    [
        branches
        for size in range(1, len(STORED_BRANCHES) + 1)
        for branches in itertools.combinations(STORED_BRANCHES, size)
    ],
)
def test_known_periods_for_branch_are_exactly_the_readable_ones(
    system, tmp_path, stored, on_disk
):
    """For every reader, ``get_known_periods(branch_name)`` lists a period if
    and only if ``get_array(period, branch_name)`` returns a value."""
    for reader_name, reader in _lineage(system).items():
        holder = reader.get_holder("uprated_income")
        if on_disk:
            directory = tmp_path / reader_name.replace(" ", "-")
            directory.mkdir()
            holder._disk_storage = holder.create_disk_storage(
                str(directory), preserve=True
            )
        storage = holder._disk_storage if on_disk else holder._memory_storage
        # One distinct year per stored branch, so each listed period traces
        # back to exactly one branch.
        for offset, branch_name in enumerate(stored):
            storage.put(np.array([1.0]), periods.period(2016 + offset), branch_name)

        stored_periods = {period for _, period in holder.get_known_branch_periods()}
        readable = {
            period
            for period in stored_periods
            if holder.get_array(period, reader.branch_name) is not None
        }

        listed = holder.get_known_periods(reader.branch_name)

        assert set(listed) == readable, reader_name
        assert len(listed) == len(set(listed)), reader_name


@pytest.mark.parametrize("on_disk", [False, True])
def test_known_periods_for_branch_list_each_period_once(system, tmp_path, on_disk):
    """A period stored under several branches a reader reads, or both in
    memory and on disk, is listed once."""
    period = periods.period(2016)
    for reader_name, reader in _lineage(system).items():
        holder = reader.get_holder("uprated_income")
        directory = tmp_path / reader_name.replace(" ", "-")
        directory.mkdir()
        holder._disk_storage = holder.create_disk_storage(str(directory), preserve=True)
        storages = [holder._memory_storage, holder._disk_storage]
        for branch_name in STORED_BRANCHES:
            for storage in storages if on_disk else storages[:1]:
                storage.put(np.array([1.0]), period, branch_name)

        assert holder.get_known_periods(reader.branch_name) == [period], reader_name


def test_get_known_periods_without_branch_lists_every_branch(system):
    simulation = new_simulation(system)
    store(simulation, "uprated_income", 2016, 1.0, "default")
    store(simulation, "uprated_income", 2018, 1.0, "other")

    holder = simulation.get_holder("uprated_income")

    assert sorted(holder.get_known_periods()) == [
        periods.period(2016),
        periods.period(2018),
    ]
    assert holder.get_known_periods("default") == [periods.period(2016)]


# ----- OnDiskStorage key parsing ----------------------------------------------

BRANCH_NAMES = (
    "default",
    "no_salt",
    "mtr_for_adult_1",
    "pre_tcja_ctc",
    "y_2019",
    "trailing_",
)
# A multi-unit or rolling-year period's string form contains ``:``, which a
# Windows file name cannot, so disk storage cannot keep those periods there.
skip_on_windows = pytest.mark.skipif(
    os.name == "nt", reason="Windows file names cannot contain ':'."
)
PERIODS = (
    periods.period(2025),
    periods.period("2025-03"),
    periods.period("2025-03-05"),
    *(
        pytest.param(periods.period(period), id=period, marks=skip_on_windows)
        for period in ("month:2025-01:3", "year:2025-03", "year:2024:2")
    ),
)


@pytest.fixture
def disk_storage(tmp_path):
    return OnDiskStorage(str(tmp_path), preserve_storage_dir=True)


@pytest.mark.parametrize("period", PERIODS, ids=str)
@pytest.mark.parametrize("branch_name", BRANCH_NAMES)
def test_on_disk_keys_round_trip(disk_storage, branch_name, period):
    disk_storage.put(np.array([3.0]), period, branch_name)

    assert disk_storage.get_known_branch_periods() == [(branch_name, period)]
    assert disk_storage.get_known_periods() == [period]
    np.testing.assert_array_equal(disk_storage.get(period, branch_name), [3.0])

    # The same holds after rebuilding the index from the files on disk.
    disk_storage.restore()
    assert disk_storage.get_known_branch_periods() == [(branch_name, period)]


@pytest.mark.parametrize("branch_name", BRANCH_NAMES)
def test_on_disk_eternal_keys_round_trip(tmp_path, branch_name):
    storage = OnDiskStorage(str(tmp_path), is_eternal=True, preserve_storage_dir=True)
    storage.put(np.array([3.0]), periods.period(2025), branch_name)

    assert storage.get_known_branch_periods() == [
        (branch_name, periods.period(periods.ETERNITY))
    ]


def test_on_disk_delete_branch_leaves_branches_sharing_its_prefix(disk_storage):
    period = periods.period(2025)
    for branch_name in ("pre_tcja", "pre_tcja_ctc", "pre"):
        disk_storage.put(np.array([1.0]), period, branch_name)

    disk_storage.delete(None, "pre_tcja")

    assert sorted(disk_storage.get_known_branch_periods()) == [
        ("pre", period),
        ("pre_tcja_ctc", period),
    ]


@pytest.mark.parametrize("branch_name", ["no_salt", "y_2019"])
def test_disk_backed_branch_with_underscore_uprates(branch_name):
    """Only public API: a variable added after ``memory_config`` is set gets
    disk storage, and the branch's input is uprated from its disk key."""
    tax_benefit_system = build_system()
    simulation = new_simulation(tax_benefit_system)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation.memory_config = MemoryConfig(max_memory_occupation=0)

    class disk_income(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Uprated yearly income stored on disk"
        uprating = "test_uprating.index"

    tax_benefit_system.add_variable(disk_income)
    branch = simulation.get_branch(branch_name)
    branch.set_input("disk_income", 2018, [5_000.0])
    disk_keys = list(branch.get_holder("disk_income")._disk_storage._files)
    assert disk_keys == [f"{branch_name}_2018"]

    result = branch.calculate("disk_income", 2020)

    assert only(result) == pytest.approx(5_000 * growth(2018, 2020))


# ----- dump_simulation --------------------------------------------------------


def test_dump_branch_saves_the_values_the_branch_reads(system):
    simulation = new_simulation(system)
    simulation.set_input("uprated_income", 2016, [1_000.0])
    simulation.set_input("uprated_income", 2017, [2_000.0])
    branch = simulation.get_branch("reform")
    branch.set_input("uprated_income", 2017, [3_000.0])
    branch.set_input("uprated_income", 2018, [5_000.0])

    directory = os.path.join(tempfile.mkdtemp(), "dump")
    simulation_dumper.dump_simulation(branch, directory)
    restored = simulation_dumper.restore_simulation(directory, system)

    holder = restored.get_holder("uprated_income")
    assert sorted(holder.get_known_periods()) == [
        periods.period(2016),
        periods.period(2017),
        periods.period(2018),
    ]
    assert only(holder.get_array(2016)) == 1_000
    assert only(holder.get_array(2017)) == 3_000
    assert only(holder.get_array(2018)) == 5_000


def test_restored_branch_dump_calculates_what_the_branch_calculates(system):
    """The dump keeps the derived mark of each value the branch reads, so the
    restored simulation uprates from the same inputs as the branch."""
    simulation = new_simulation(system)
    simulation.set_input("uprated_income", 2016, [1_000.0])
    # Calculated on ``default``, then set as an input on the branch.
    simulation.calculate("uprated_income", 2019)
    branch = simulation.get_branch("reform")
    branch.set_input("uprated_income", 2019, [3_000.0])
    # Calculated on the branch only.
    branch.calculate("uprated_income", 2017)

    directory = os.path.join(tempfile.mkdtemp(), "dump")
    simulation_dumper.dump_simulation(branch, directory)
    restored = simulation_dumper.restore_simulation(directory, system)

    holder = restored.get_holder("uprated_income")
    assert sorted(
        (str(period), holder.is_derived(period))
        for period in holder.get_known_periods()
    ) == [("2016", False), ("2017", True), ("2019", False)]
    for year in range(2016, 2023):
        assert only(restored.calculate("uprated_income", year)) == pytest.approx(
            only(branch.calculate("uprated_income", year))
        ), year
    assert only(restored.calculate("uprated_income", 2021)) == pytest.approx(
        3_000 * growth(2019, 2021)
    )


def test_restore_simulation_leaves_out_files_that_are_not_values(system, tmp_path):
    simulation = new_simulation(system)
    simulation.set_input("uprated_income", 2016, [1_000.0])
    directory = str(tmp_path / "dump")
    simulation_dumper.dump_simulation(simulation, directory)
    np.save(os.path.join(directory, "uprated_income", "stray.npy"), np.array([9.0]))

    with pytest.warns(UserWarning, match="stray.npy"):
        restored = simulation_dumper.restore_simulation(directory, system)

    holder = restored.get_holder("uprated_income")
    assert holder.get_known_periods() == [periods.period(2016)]
    assert only(holder.get_array(2016)) == 1_000
