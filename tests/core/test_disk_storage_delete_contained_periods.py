"""``OnDiskStorage.delete(period, branch_name)`` deletes the periods within
``period`` (policyengine-core#564).

``Holder.delete_arrays`` documents that deleting a period removes "all values
for any period included in period", and ``InMemoryStorage.delete`` does that
(deleting ``2025`` removes ``2025-01`` ... ``2025-12``). ``OnDiskStorage.delete``
used to remove only the file keyed exactly ``f"{branch_name}_{period}"``, so a
disk-backed simulation kept values the caller had deleted.

Each example runs on both backends. ``test_disk_and_memory_delete_the_same_keys``
checks every pair of a stored and a deleted period on a small grid, and
``test_disk_storage_delete_property.py`` checks random sequences of puts and
deletes.
"""

import os
import warnings

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.data_storage import InMemoryStorage, OnDiskStorage
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import SimulationBuilder

BACKENDS = ("memory", "disk")

# On Windows a ``:`` in a file name is invalid or names an alternate data
# stream, so disk storage cannot keep periods whose string form has one
# (``month:2025-01:3``, ``year:2025-03``). That predates this fix; tests of
# such periods run elsewhere.
skip_colon_periods_on_windows = pytest.mark.skipif(
    os.name == "nt",
    reason="On-disk keys for multi-unit periods contain ':', which Windows "
    "file names cannot.",
)


@pytest.fixture
def make_storage(tmp_path):
    def make(backend, is_eternal=False):
        if backend == "memory":
            return InMemoryStorage(is_eternal=is_eternal)
        directory = tmp_path / f"storage_{len(os.listdir(tmp_path))}"
        directory.mkdir()
        return OnDiskStorage(
            str(directory), is_eternal=is_eternal, preserve_storage_dir=True
        )

    return make


def put_all(storage, keys):
    for index, (branch_name, period) in enumerate(keys):
        storage.put(np.array([float(index)]), periods.period(period), branch_name)


def known(storage):
    """The stored (branch name, period string) pairs, sorted."""
    return sorted(
        (branch_name, str(period))
        for branch_name, period in storage.get_known_branch_periods()
    )


# ----- Examples on both backends ----------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_exact_period(make_storage, backend):
    storage = make_storage(backend)
    put_all(storage, [("default", "2025-01"), ("default", "2025-02")])

    storage.delete(periods.period("2025-01"), "default")

    assert known(storage) == [("default", "2025-02")]


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_year_deletes_its_months_and_days(make_storage, backend):
    storage = make_storage(backend)
    put_all(
        storage,
        [
            ("default", "2024-12"),
            ("default", "2025"),
            ("default", "2025-01"),
            ("default", "2025-12"),
            ("default", "2025-03-15"),
            ("default", "2026-01"),
        ],
    )

    storage.delete(periods.period(2025), "default")

    assert known(storage) == [("default", "2024-12"), ("default", "2026-01")]


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_month_keeps_the_year_that_contains_it(make_storage, backend):
    storage = make_storage(backend)
    put_all(
        storage,
        [
            ("default", "2025"),
            ("default", "2025-01"),
            ("default", "2025-01-31"),
            ("default", "2025-02"),
            ("default", "2025-02-01"),
        ],
    )

    storage.delete(periods.period("2025-01"), "default")

    assert known(storage) == [
        ("default", "2025"),
        ("default", "2025-02"),
        ("default", "2025-02-01"),
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_keeps_other_branches(make_storage, backend):
    storage = make_storage(backend)
    put_all(
        storage,
        [
            ("default", "2025-01"),
            ("reform", "2025-01"),
            ("reform", "2025-02"),
        ],
    )

    storage.delete(periods.period(2025), "reform")

    assert known(storage) == [("default", "2025-01")]
    np.testing.assert_array_equal(storage.get("2025-01", "default"), [0.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_matches_whole_branch_names_containing_underscores(
    make_storage, backend
):
    # ``pre`` and ``pre_tcja`` are prefixes of ``pre_tcja_ctc``; the ``y``
    # branch's prefix ``y_`` starts the ``y_2019`` branch's keys.
    branch_names = ("pre", "pre_tcja", "pre_tcja_ctc", "y", "y_2019", "trailing_")
    storage = make_storage(backend)
    put_all(
        storage,
        [
            (branch_name, month)
            for branch_name in branch_names
            for month in ("2025-01", "2025-02")
        ],
    )

    storage.delete(periods.period(2025), "pre_tcja")
    storage.delete(periods.period(2025), "y")
    storage.delete(periods.period("2025-01"), "trailing_")

    assert known(storage) == [
        ("pre", "2025-01"),
        ("pre", "2025-02"),
        ("pre_tcja_ctc", "2025-01"),
        ("pre_tcja_ctc", "2025-02"),
        ("trailing_", "2025-02"),
        ("y_2019", "2025-01"),
        ("y_2019", "2025-02"),
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_eternal_storage_deletes_only_the_branch_eternity_key(make_storage, backend):
    storage = make_storage(backend, is_eternal=True)
    put_all(storage, [("default", "2025"), ("no_salt", "2025"), ("no", "2025")])

    # Any period deletes the branch's single ETERNITY value.
    storage.delete(periods.period("2025-01"), "no_salt")

    assert known(storage) == [("default", "ETERNITY"), ("no", "ETERNITY")]


@pytest.mark.parametrize("backend", BACKENDS)
def test_deleting_eternity_deletes_every_period_of_the_branch(make_storage, backend):
    storage = make_storage(backend)
    put_all(
        storage,
        [("default", "2024"), ("default", "2025-01"), ("reform", "2025-01")],
    )

    storage.delete(periods.period(periods.ETERNITY), "default")

    assert known(storage) == [("reform", "2025-01")]


@skip_colon_periods_on_windows
@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_multi_unit_periods(make_storage, backend):
    storage = make_storage(backend)
    put_all(
        storage,
        [
            ("default", "month:2025-01:3"),
            ("default", "year:2025-03"),
            ("default", "2025-02"),
            ("default", "2025-04"),
        ],
    )

    # A quarter contains its months, but a month does not contain the
    # quarter, and 2025 does not contain the year starting in March 2025.
    storage.delete(periods.period("month:2025-01:3"), "default")
    assert known(storage) == [("default", "2025-04"), ("default", "year:2025-03")]

    storage.delete(periods.period("2025-04"), "default")
    assert known(storage) == [("default", "year:2025-03")]

    storage.delete(periods.period(2025), "default")
    assert known(storage) == [("default", "year:2025-03")]

    storage.delete(periods.period("year:2025:2"), "default")
    assert known(storage) == []


# ----- Disk and memory agree on every pair of periods -------------------------

PERIODS = [
    "2024",
    "2025",
    "2025-01",
    "2025-02",
    "2025-12",
    "2025-01-01",
    "2025-01-31",
    "2025-02-01",
    "2026-01",
    "ETERNITY",
]
if os.name != "nt":
    PERIODS += ["month:2025-01:2", "year:2025-03", "year:2024:2"]
BRANCH_NAMES = ["default", "pre_tcja", "pre_tcja_ctc"]


@pytest.mark.parametrize("deleted_branch_name", BRANCH_NAMES)
@pytest.mark.parametrize("deleted_period", PERIODS)
def test_disk_and_memory_delete_the_same_keys(
    make_storage, deleted_period, deleted_branch_name
):
    keys = [(branch_name, period) for branch_name in BRANCH_NAMES for period in PERIODS]
    memory, disk = make_storage("memory"), make_storage("disk")
    put_all(memory, keys)
    put_all(disk, keys)
    deleted_period = periods.period(deleted_period)

    memory.delete(deleted_period, deleted_branch_name)
    disk.delete(deleted_period, deleted_branch_name)

    assert known(disk) == known(memory)
    # And both keep exactly the keys the period does not contain.
    assert known(memory) == sorted(
        (branch_name, str(periods.period(period)))
        for branch_name, period in keys
        if branch_name != deleted_branch_name
        or not deleted_period.contains(periods.period(period))
    )
    for branch_name, period in known(memory):
        np.testing.assert_array_equal(
            disk.get(period, branch_name), memory.get(period, branch_name)
        )


# ----- Through the simulation (the #564 reproduction) -------------------------


def simulation_with_rent(on_disk):
    simulation = SimulationBuilder().build_from_entities(
        CountryTaxBenefitSystem(),
        {"persons": {"a": {}}, "households": {"h": {"parents": ["a"]}}},
    )
    if on_disk:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulation.memory_config = MemoryConfig(max_memory_occupation=0)
        holder = simulation.get_holder("rent")
        holder._disk_storage = holder.create_disk_storage()
        holder._on_disk_storable = True
    simulation.set_input("rent", "2025-01", [500.0])
    simulation.set_input("rent", "2025-02", [600.0])
    simulation.set_input("rent", "2026-01", [700.0])
    return simulation


@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
def test_simulation_delete_arrays_deletes_the_months_of_a_year(on_disk):
    simulation = simulation_with_rent(on_disk)
    holder = simulation.get_holder("rent")
    if on_disk:
        assert holder._memory_storage.get_known_periods() == []
        assert len(holder._disk_storage.get_known_periods()) == 3

    simulation.delete_arrays("rent", "2025")

    assert holder.get_known_periods() == [periods.period("2026-01")]
    assert holder.get_array("2025-01") is None
    assert holder.get_array("2025-02") is None
    np.testing.assert_array_equal(holder.get_array("2026-01"), [700.0])


@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
def test_holder_delete_arrays_deletes_the_months_of_a_year(on_disk):
    simulation = simulation_with_rent(on_disk)
    holder = simulation.get_holder("rent")

    holder.delete_arrays("2025", "default")

    assert holder.get_known_periods() == [periods.period("2026-01")]
