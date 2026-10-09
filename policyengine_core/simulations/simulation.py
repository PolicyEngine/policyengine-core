import hashlib
import os
from contextlib import nullcontext
import types
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import logging
from pathlib import Path

from policyengine_core import commons, periods
from policyengine_core.data.dataset import Dataset
from policyengine_core.data_storage.store_history import (
    StoreHistory,
    next_sequence_number,
)
from policyengine_core.data_storage.storage_directory import (
    TemporaryStorageDirectory,
    directory_containing,
)
from policyengine_core.entities.entity import Entity
from policyengine_core.enums import Enum, EnumArray
from policyengine_core.errors import CycleError, SpiralError
from policyengine_core.holders.holder import Holder
from policyengine_core.periods import Period
from policyengine_core.periods.config import ETERNITY, MONTH, YEAR
from policyengine_core.periods.helpers import period
from policyengine_core.tracers import (
    FullTracer,
    SimpleTracer,
    TracingParameterNode,
)
import random
from policyengine_core.tools.hugging_face import *
from policyengine_core.tools.google_cloud import (
    parse_gs_url,
    download_gcs_file,
)

import json


class _BranchClone:
    """The simulation ``get_branch`` is cloning into a new branch.

    ``get_branch`` announces the clone it is about to make, and ``clone``
    shares the cached arrays with the copy (instead of copying them) only for
    that simulation, and only once: the first ``clone`` of it while
    ``get_branch`` runs. Any other ``clone`` call made meanwhile, or after
    ``get_branch`` returns, copies as usual. A subclass's ``clone`` normally
    reaches this class's ``clone`` once, through ``super().clone``; if it
    first clones the same simulation directly, that clone is the one that
    shares.
    """

    def __init__(self, simulation: "Simulation"):
        self.simulation = simulation
        self.pending = True


# Set by ``get_branch`` around its ``clone`` call. This is a context variable
# rather than a ``clone`` argument because country packages override ``clone``
# with its existing signature and call ``super().clone`` (policyengine-us's
# SPM ``Simulation`` does), so an extra argument could not reach this class's
# ``clone`` through them.
_branch_clone: ContextVar[Optional[_BranchClone]] = ContextVar(
    "_branch_clone", default=None
)


def _stable_hash_to_seed(value: str) -> int:
    """Deterministically hash a string to an int suitable for numpy.random.seed.

    Python's built-in ``hash()`` is randomized per process (PYTHONHASHSEED) for
    strings, which makes seeds derived from it non-reproducible across runs.
    Use a stable cryptographic hash truncated to the seed range instead.
    """
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 1000000


def _uprating_index_value(parameter, instant) -> Optional[float]:
    """Value of a variable's uprating index at ``instant``, held flat where
    the index has no value.

    Before the index's first value this returns that first value, as if the
    earliest value had been extended backward (country packages backdate
    parameters this way; a parameter itself returns ``None`` there). After an
    explicit null it returns the last value before the null. A value known
    for a period the index does not reach is therefore carried over
    unchanged until the index starts, then uprated with it. Returns ``None``
    only if the index has no non-null value at all.
    """
    value = parameter(instant)
    if value is not None:
        return value
    defined = [
        value_at_instant
        for value_at_instant in getattr(parameter, "values_list", [])
        if value_at_instant.value is not None
    ]
    if not defined:
        return None
    # ``values_list`` runs from the latest instant to the earliest.
    instant_str = str(instant)
    for value_at_instant in defined:
        if value_at_instant.instant_str <= instant_str:
            return value_at_instant.value
    return defined[-1].value


_DAYS_BEFORE_MONTH = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def _end_order(period: Period) -> float:
    """Sorts periods by when they end: the number of the day after the
    period's last day, counting days as ``date.toordinal`` does.

    ``period.stop`` gives the same order wherever it has a value, but it
    raises for a period that ends after 9999-12-31 (``day:9999-12-30:3``),
    the last date ``datetime`` can represent. This is integer arithmetic on
    the same (proleptic Gregorian) calendar, so every period has a value and
    one that ends later always sorts later.
    """
    unit, (year, month, day), size = period
    if unit == ETERNITY:
        return float("inf")
    if unit == periods.DAY:
        day += size
    elif unit == MONTH:
        year, month = divmod(year * 12 + month - 1 + size, 12)
        month += 1
    else:
        year += size
    # The first day of that month, then ``day - 1`` days on: a day past the
    # month's end runs into the next month, as it does in ``period.stop``.
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_before_year = (
        (year - 1) * 365 + (year - 1) // 4 - (year - 1) // 100 + (year - 1) // 400
    )
    return days_before_year + _DAYS_BEFORE_MONTH[month - 1] + (leap and month > 2) + day


def _latest_input_key(period: Period, definition_period: str) -> tuple:
    """Sort key for the stored input that auto-carry-over or uprating reads.

    The input at the variable's own definition-period unit beats one in
    another unit; then the one that starts last wins; on a tie, the one that
    ends last (``_end_order``), then the larger unit, then the period's
    string form. Distinct periods never tie, so the input chosen depends only
    on which periods are stored, never on the order they were stored in (or
    on whether a value is in memory or on disk). Compare ``period.start``
    (temporal order): sorting Period tuples lexicographically puts "year"
    before "month" alphabetically, so a known "2023" annual value would win
    over a later "2024-06" monthly value (bug H1).
    """
    return (
        period.unit == definition_period,
        period.start,
        _end_order(period),
        periods.unit_weight(period.unit),
        str(period),
    )


if TYPE_CHECKING:
    from policyengine_core.taxbenefitsystems import TaxBenefitSystem

from policyengine_core.experimental import MemoryConfig
from policyengine_core.populations import Population, GroupPopulation
from policyengine_core.tracers import SimpleTracer
from policyengine_core.variables import Variable, QuantityType
from policyengine_core.reforms.reform import Reform
from policyengine_core.parameters import get_parameter
from policyengine_core.simulations.simulation_macro_cache import (
    SimulationMacroCache,
)


# The simulation whose formula is running. When that formula calculates a
# value in another simulation (a branch it created, say), what the formula
# returns, and so what its simulation stores, may be calculated from the other
# simulation's values: ``calculate`` then merges the other's store history into
# this one's (see ``StoreHistory``).
_formula_simulation: ContextVar[Optional["Simulation"]] = ContextVar(
    "_formula_simulation", default=None
)

# How many times ``calculate`` runs a calculation again when an input set on
# the simulation while it ran (by a formula, say) dropped values. A formula
# that changes an input on every run is kept from looping; its last result is
# returned but not kept.
_RERUNS_AFTER_INPUT_CHANGE = 10

# The calculations running in this context, innermost last (``_Frame``).
_calculation_frames: ContextVar[tuple] = ContextVar("_calculation_frames", default=())

# Only these frames try to keep work that the stale shared frame refused.
# Context-free activity anywhere prevents that optional cache write, without
# changing ordinary frames' scoped read tracking or convergence decisions.
_restricted_frames = set()
_restricted_frames_lock = Lock()


def _note_unobserved_activity() -> None:
    if _restricted_frames:
        with _restricted_frames_lock:
            for frame in _restricted_frames:
                frame.unobserved_activity = True


class _Frame:
    """A calculation running in ``simulation``.

    ``start`` is the simulation's ``_input_epoch`` when the calculation (last)
    began, and ``reads`` holds, for each other simulation it got a value
    from, directly or through the calculations it called, that simulation's
    ``_input_epoch`` when the calculation that produced the value began. If
    any of those has changed since (a drop there), the value may come from a
    replaced one, and so may what the frame calculates from it, which is then
    not kept (:meth:`is_stale`).

    Used as a context manager: entering pushes the frame on
    ``_calculation_frames`` and lists it in the simulation's
    ``_open_frames``, so that a value read in a thread with no frame reaches
    the frames waiting for it (``_hand_to_waiting_ancestors``). Leaving,
    whatever way the calculation ends (a value, an early default, an error),
    tells the caller what it read (``_hand_to_caller``). A class with slots,
    not a generator or a dict: one runs for every calculation.
    """

    __slots__ = (
        "simulation",
        "start",
        "reads",
        "outer",
        "token",
        "registry",
        "changes",
        "drop_epochs",
        "untracked_changes",
        "inputs_set",
        "branch_identities",
        "created_branches",
        "carried_branches",
        "stale_parent",
        "unobserved_activity",
    )

    def __init__(self, simulation: "Simulation", stale_parent: bool = False):
        self.simulation = simulation
        self.created_branches = {}
        self.carried_branches = {}
        self.stale_parent = stale_parent

    def restart(self) -> None:
        """Begin again, as when the calculation runs again."""
        self.start = self.simulation._input_epoch
        self.inputs_set = self.simulation._inputs_set
        self.reads = {}
        self.changes = []
        self.drop_epochs = {}
        self.untracked_changes = False
        self.unobserved_activity = False
        self.branch_identities = {}
        # Retain only creations whose original registration still names
        # the same object. Saved or replaced snapshots keep their identities.
        self.carried_branches = _registered_creations(
            self.carried_branches | self.created_branches
        )
        self.created_branches = {}

    def note_branch(self, simulation: "Simulation") -> None:
        """A name path must refer to just one simulation within an attempt.

        A direct clone retains the branch's name and ancestors. Distinct
        clones at that path can take turns making identical stores without
        having reached the same state. Recreating one branch between attempts
        is still comparable, since this map resets on each restart.
        """
        path = _branch_path(simulation)
        previous = self.branch_identities.setdefault(path, simulation)
        if previous is not simulation:
            self.untracked_changes = True

    def transition_signature(self, result: ArrayLike) -> Optional[tuple]:
        """A complete, comparable branch transition, or no convergence proof.

        Two identical transitions returning identical values reach a fixed
        point for deterministic formulas. Only foreign branches qualify:
        own-simulation changes still require the existing input-first rerun.
        Every stale read's intervening drops must have been observed in this
        context; a thread's unobserved writes cannot prove convergence.
        """
        # Include read-only clones too: the same logged mutation can target
        # a different clone on each attempt after inspecting several clones.
        for other in tuple(self.reads):
            self.note_branch(other)
        if (
            self.untracked_changes
            or not self.changes
            or self.start != self.simulation._input_epoch
            or self.inputs_set != self.simulation._inputs_set
        ):
            return None
        for other, epoch in tuple(self.reads.items()):
            changed = other._input_epoch - epoch
            observed = self.drop_epochs.get(other, ())
            if changed < 0 or sum(number > epoch for number in observed) != changed:
                return None
        value = _value_signature(result)
        if value is None:
            return None
        # A saved branch snapshot can share a name with a later recreation.
        # Creations observed in this frame may keep their name path across
        # retries while still registered. Recheck here: a formula can replace
        # a registration after reading it, before the attempt ends.
        comparable_branches = (
            self.created_branches.keys()
            | _registered_creations(self.carried_branches).keys()
        )
        changes = tuple(
            (operation, _branch_path(other, comparable_branches), *details)
            for operation, other, *details in self.changes
        )
        reads = frozenset(
            _branch_path(other, comparable_branches) for other in self.reads
        )
        return changes, reads, value

    def settle_reads(self) -> None:
        """Keep dependencies, at their settled epochs, after a proven fixed point."""
        self.reads = {other: other._input_epoch for other in self.reads}

    def is_stale(self) -> bool:
        """Whether what the frame calculates may come from a value since replaced."""
        if self.simulation._input_epoch != self.start:
            return True
        reads = self.reads
        # A snapshot: a thread handing a read to this frame may add one meanwhile.
        return bool(reads) and any(
            other._input_epoch != epoch for other, epoch in tuple(reads.items())
        )

    def may_keep(self) -> bool:
        """Cache independent work after a stale caller without hiding foreign reads.

        A fresh nested frame can keep work using only its own simulation.
        Reads, mutations and branch creations must still reach the stale
        caller on its next attempt: caching them here could hide a later
        input change or skip a mutation when that caller retries.
        """
        return (
            not (
                self.stale_parent
                and (
                    self.reads
                    or self.changes
                    or self.untracked_changes
                    or self.created_branches
                    or self.unobserved_activity
                )
            )
            and not self.is_stale()
        )

    def __enter__(self) -> "_Frame":
        simulation = self.simulation
        self.restart()
        self.outer = outer = _calculation_frames.get()
        self.token = _calculation_frames.set(outer + (self,))
        registry = simulation._open_frames
        if registry is None:
            # ``setdefault``: two threads starting here at once share one set.
            registry = simulation.__dict__.setdefault("_open_frames", set())
        registry.add(self)
        self.registry = registry
        if self.stale_parent:
            with _restricted_frames_lock:
                _restricted_frames.add(self)
        return self

    def __exit__(self, *exc_info) -> bool:
        if self.stale_parent:
            with _restricted_frames_lock:
                _restricted_frames.discard(self)
        self.registry.discard(self)
        _calculation_frames.reset(self.token)
        _hand_to_caller(self.outer, self)
        return False


def _frame_for(simulation: "Simulation", outermost: bool) -> Optional[_Frame]:
    """A new frame for a calculation in ``simulation``, or ``None`` to share
    the frame it runs in.

    A calculation nested in another in the same simulation shares that one's
    frame while it can cache: the shared frame began no later and notes at
    least the same reads, each at the first epoch noted, so it is stale
    whenever the nested calculation's own would be, and the nested one keeps
    a result only when it would have anyway. The outermost calculation has its
    own, to run again from (``Simulation.calculate``). Once a shared frame
    cannot cache, independent nested work gets a fresh frame so it can still
    cache. Reads and mutations remain guarded (``_Frame.may_keep``).
    """
    if outermost:
        return _Frame(simulation)
    frames = _calculation_frames.get()
    if not frames or frames[-1].simulation is not simulation:
        return _Frame(simulation)
    if not frames[-1].may_keep():
        return _Frame(simulation, stale_parent=True)
    return None


def _hand_to_caller(outer: tuple, frame: _Frame) -> None:
    """Pass what a calculation read on to the frame that called it.

    The calculation's own simulation counts at its ``start``: if an input
    there changed while it ran, its caller's result is stale too. With no
    calling frame in this context (``outer`` empty), the reads go to the
    frames open in the simulation's ancestors (``_hand_to_waiting_ancestors``).
    """
    reads = frame.reads
    reads = tuple(reads.items()) if reads else ()
    simulation = frame.simulation
    if not outer:
        _hand_to_waiting_ancestors(simulation, frame.start, reads)
        return
    caller = outer[-1]
    caller.untracked_changes |= frame.untracked_changes
    caller_simulation = caller.simulation
    caller_reads = caller.reads
    for other, epoch in reads:
        if other is not caller_simulation and other not in caller_reads:
            caller_reads[other] = epoch
    if simulation is not caller_simulation and simulation not in caller_reads:
        caller_reads[simulation] = frame.start


def _hand_to_waiting_ancestors(
    simulation: "Simulation", start: int, reads: tuple
) -> None:
    """Note a value read in ``simulation`` with no calling frame in this context.

    That is a call from code outside any formula, or from a thread a formula
    started without copying its context: then a formula in one of the
    simulation's ancestors is waiting for the value, so every frame open in
    those ancestors notes the read, as it would had the formula called
    directly (as for the store history; see
    ``Simulation._share_store_history_with_caller``). A frame keeps the first
    epoch it notes for a simulation.
    """
    _note_unobserved_activity()
    ancestor = getattr(simulation, "parent_branch", None)
    while ancestor is not None:
        for open_frame in tuple(getattr(ancestor, "_open_frames", None) or ()):
            open_frame.untracked_changes = True
            open_reads = open_frame.reads
            for other, epoch in reads:
                if other is not ancestor:
                    open_reads.setdefault(other, epoch)
            open_reads.setdefault(simulation, start)
        ancestor = getattr(ancestor, "parent_branch", None)


def _value_signature(value: ArrayLike) -> Optional[tuple]:
    """An immutable exact value comparison, including float signs and enum type.

    Object and string arrays have no supported convergence comparison. Bytes
    avoid both hash collisions and later mutation of a retained array view.
    """
    if np.ma.isMaskedArray(value):
        return None
    enum = value.possible_values if isinstance(value, EnumArray) else None
    array = np.asarray(value)
    if array.dtype.kind not in "biufcmM":
        return None
    return array.dtype.str, array.shape, enum, array.tobytes()


def _registered_creations(branches: dict) -> dict:
    """Observed creations still registered under their original parent and name."""
    return {
        branch: (parent, name)
        for branch, (parent, name) in branches.items()
        if branch.parent_branch is parent
        and branch.branch_name == name
        and parent.branches.get(name) is branch
    }


def _branch_path(
    simulation: "Simulation", created_branches: Optional[set] = None
) -> tuple:
    """Identify recreated ``get_branch`` branches, keeping direct clones distinct.

    A direct clone retains its source's name and parent but has independent
    inputs. It anchors a new path by identity, even if later registered under
    the source's name; only branches made by ``get_branch`` compare by name.
    """
    names = []
    while (
        getattr(simulation, "parent_branch", None) is not None
        and getattr(simulation, "_fixed_point_branch", False)
        and (created_branches is None or simulation in created_branches)
    ):
        names.append(simulation.branch_name)
        simulation = simulation.parent_branch
    return simulation, tuple(reversed(names))


def _note_input_change(
    simulation: "Simulation",
    operation: str,
    variable: str,
    period: Optional[Period],
    branch_name: str,
    value: Optional[ArrayLike] = None,
) -> None:
    """Record actual stores and deletes in each frame observing this context."""
    frames = _calculation_frames.get()
    if not frames:
        _taint_waiting_frames(simulation)
        return
    signature = _value_signature(value) if operation == "set" else None
    if operation == "set" and signature is None:
        for frame in frames:
            frame.untracked_changes = True
        return
    change = (
        operation,
        simulation,
        branch_name,
        variable,
        period,
        signature,
    )
    for frame in frames:
        frame.note_branch(simulation)
        if (
            simulation is frame.simulation
            or getattr(simulation, "parent_branch", None) is None
        ):
            frame.untracked_changes = True
        frame.changes.append(change)


def _taint_waiting_frames(simulation: "Simulation") -> None:
    """An unobserved mutation cannot certify a waiting calculation's fixed point.

    A thread without a formula context can write before the first read, so
    counting only drops since that read would miss its mutation entirely.
    The simulation and its ancestors may have frames waiting on that thread.
    """
    _note_unobserved_activity()
    while simulation is not None:
        for frame in tuple(getattr(simulation, "_open_frames", None) or ()):
            frame.untracked_changes = True
        simulation = getattr(simulation, "parent_branch", None)


@dataclass(frozen=True)
class PreservedUserInput:
    variable_name: str
    branch_name: str
    period: Period
    value: object
    storage: str
    disk_key: Optional[str] = None
    disk_file: Optional[str] = None
    disk_enum: object = None


class Simulation:
    """
    Represents a simulation, and handles the calculation logic
    """

    default_tax_benefit_system: Type["TaxBenefitSystem"] = None
    """The default tax-benefit system class to use if none is provided."""

    # Counters read on every calculation (see ``set_input``). Class defaults,
    # so a subclass that skips ``__init__`` reads them too; incrementing one
    # gives the simulation its own.
    _input_epoch: int = 0  # drops on this simulation
    _inputs_set: int = 0  # inputs stored through ``Holder.set_input``
    _calculations_in_flight: int = 0  # ``calculate``/``calculate_add`` running
    _calculations_started: int = 0  # ``_calculate`` calls begun
    _requested_variables: Optional[set] = None
    _store_history: Optional[StoreHistory] = None  # see ``_get_store_history``
    _open_frames: Optional[set] = None  # the ``_Frame``s open here, any thread

    default_tax_benefit_system_instance: "TaxBenefitSystem" = None
    """The default tax-benefit system instance to use if none is provided. This requires that the tax-benefit system is initialised when importing a country package. This will slow down the import, but may speed up individual simulations."""

    default_dataset: Dataset = None
    """The default dataset class to use if none is provided."""

    default_role: str = "member"
    """The default role to assign people to groups if none is provided."""

    default_input_period: str = None
    """The default period to use when inputting variables."""

    default_calculation_period: str = None
    """The default period to calculate for if none is provided."""

    datasets: List[Dataset] = []
    """The list of datasets available for this simulation."""

    baseline: "Simulation" = None
    """The baseline simulation, if this simulation is a reform."""

    is_over_dataset: bool = False
    """Whether this simulation is built over a dataset."""

    macro_cache_read: bool = False
    """Whether to read from the macro cache."""

    macro_cache_write: bool = False
    """Whether to write to the macro cache."""

    start_instant: str = None
    """The earliest data input instant of the simulation."""

    # The temporary folder this simulation made for the values it stores on
    # disk, if it made one (see ``data_storage_dir``).
    _storage_directory: TemporaryStorageDirectory = None
    # The folder that one is made in; ``None`` for the default temporary
    # folder (see ``clone``).
    _storage_dir_parent: str = None
    # The process ``_data_storage_dir`` is for: the one that made the
    # simulation or the folder, or first stored a value on disk in it.
    _storage_dir_pid: int = None
    # The live temporary folder a simulation made that is or contains this
    # simulation's folder, or, until this one makes its folder, the folder
    # it will make it in: kept alive while this simulation is, since
    # removing it would remove this simulation's folder (see
    # ``_data_storage_dir``).
    _storage_dir_keeper: TemporaryStorageDirectory = None

    @property
    def _data_storage_dir(self) -> Optional[str]:
        """The folder this simulation stores values on disk in, once it has
        one. Set it to choose the folder (see ``data_storage_dir``)."""
        # Kept in the instance's ``__dict__`` under the same name, which
        # this property takes precedence over.
        return self.__dict__.get("_data_storage_dir")

    @_data_storage_dir.setter
    def _data_storage_dir(self, path: Optional[str]) -> None:
        self.__dict__["_data_storage_dir"] = path
        # A folder in a temporary folder a simulation made (another
        # simulation's, given to this one, say) goes when that one is
        # removed, so this simulation keeps that one alive.
        self._storage_dir_keeper = (
            None if path is None else directory_containing(os.fspath(path))
        )

    def __init__(
        self,
        tax_benefit_system: "TaxBenefitSystem" = None,
        populations: Dict[str, Population] = None,
        situation: dict = None,
        dataset: Union[str, Type[Dataset]] = None,
        reform: Reform = None,
        trace: bool = False,
        default_input_period: str = None,
        default_calculation_period: str = None,
    ):
        self.default_input_period = default_input_period or self.default_input_period
        self.default_calculation_period = (
            default_calculation_period or self.default_calculation_period
        )
        if tax_benefit_system is None:
            if self.default_tax_benefit_system_instance is not None and reform is None:
                tax_benefit_system = self.default_tax_benefit_system_instance
            else:
                tax_benefit_system = self.default_tax_benefit_system(reform=reform)
            self.tax_benefit_system = tax_benefit_system

        self.reform = reform
        self.tax_benefit_system = tax_benefit_system
        self.branch_name = "default"
        self.dataset = dataset

        if dataset is None:
            if self.default_dataset is not None:
                dataset = self.default_dataset
        self.is_over_dataset = dataset is not None

        self.invalidated_caches = set()
        self._fast_cache: dict = {}
        # ``set_input`` records each (variable_name, branch_name, period) it
        # populates so ``_invalidate_all_caches`` can tell user-provided
        # source data apart from formula-computed caches. Without this the
        # post-``apply_reform`` cache wipe would also wipe the dataset the
        # simulation was loaded from. The record follows this simulation's
        # storage: each entry names one stored value (the period is the one
        # storage keys it under), ``delete_arrays`` drops the entries for the
        # values it deletes, and ``clone`` (so also ``get_branch``) gives the
        # copy its own record.
        self._user_input_keys: set[tuple[str, str, Period]] = set()
        # What this simulation's values may have been calculated from; each
        # branch starts with a copy (see ``set_input``).
        self._store_history = StoreHistory()
        self.debug: bool = False
        self.trace: bool = trace
        self.tracer: SimpleTracer = SimpleTracer() if not trace else FullTracer()
        self.opt_out_cache: bool = False
        # controls the spirals detection; check for performance impact if > 1
        self.max_spiral_loops: int = 10
        self.memory_config: MemoryConfig = None
        self._data_storage_dir: str = None
        self._storage_dir_pid = os.getpid()

        self.branches: Dict[str, Simulation] = {}
        self.has_axes = False

        np.random.seed(0)

        if situation is not None:
            if dataset is not None:
                raise ValueError(
                    "You provided both a situation and a dataset. Only one input method is allowed."
                )
            self.build_from_populations(self.tax_benefit_system.instantiate_entities())
            from policyengine_core.simulations.simulation_builder import (
                SimulationBuilder,
            )  # Import here to avoid circular dependency

            builder = SimulationBuilder()
            builder.default_period = self.default_input_period
            builder.build_from_dict(self.tax_benefit_system, situation, self)
            self.has_axes = builder.has_axes

        if populations is not None:
            self.build_from_populations(populations)

        if dataset is not None:
            if isinstance(dataset, str):
                if "hf://" in dataset:
                    owner, repo, filename, version = parse_hf_url(dataset)
                    dataset = download_huggingface_dataset(
                        repo=f"{owner}/{repo}",
                        repo_filename=filename,
                        version=version,
                    )
                elif "gs://" in dataset:
                    bucket, file_path, version = parse_gs_url(dataset)
                    dataset = download_gcs_file(
                        bucket=bucket,
                        file_path=file_path,
                        version=version,
                    )
                datasets_by_name = {dataset.name: dataset for dataset in self.datasets}
                if dataset in datasets_by_name:
                    dataset = datasets_by_name.get(dataset)
                elif Path(dataset).exists():
                    dataset = Dataset.from_file(dataset, self.default_input_period)
            if isinstance(dataset, type):
                self.dataset: Dataset = dataset(require=True)
            elif isinstance(dataset, pd.DataFrame):
                self.dataset = Dataset.from_dataframe(
                    dataset, self.default_input_period
                )
            else:
                self.dataset = dataset
            self.build_from_dataset()

        self.tax_benefit_system.simulation = self

        if self.reform is not None:
            self.tax_benefit_system.apply_reform_set(self.reform)

        # Backwards compatibility methods
        self.calc = self.calculate
        self.df = self.calculate_dataframe

        self.input_variables = [
            variable.name
            for variable in self.tax_benefit_system.variables.values()
            if len(self.get_holder(variable.name).get_known_periods()) > 0
        ]

        self.situation_input = situation
        if self.situation_input is not None:
            original_input = json.loads(json.dumps(self.situation_input))
            if original_input.get("axes") is not None:
                original_input["axes"] = {}
            # Hash the situation input to a random number, so situations with axes behave the
            # same ways as the same situations without axes. ``sort_keys=True``
            # keeps the hash stable across equivalent inputs built from differently
            # ordered dicts, and ``_stable_hash_to_seed`` keeps it stable across
            # Python processes (built-in ``hash`` is randomized per process).
            hashed_input = _stable_hash_to_seed(
                json.dumps(original_input, sort_keys=True)
            )
            np.random.seed(hashed_input)

        if reform is not None:
            self.baseline = self.get_branch("baseline")
            self.baseline.trace = self.trace
            self.baseline.tracer = self.tracer
            self.baseline.tax_benefit_system = self.default_tax_benefit_system_instance
            # The branch was built under the reform's system: its populations
            # looked variables up there, and its holders kept the reform's
            # variables, so a variable the reform neutralized read as the
            # default in the baseline too.
            self.baseline._bind_to_tax_benefit_system()
        else:
            self.baseline = None

        self.parent_branch = None

    def apply_reform(self, reform: Union[tuple, Reform]):
        if isinstance(reform, tuple):
            for subreform in reform:
                self.apply_reform(subreform)
        else:
            if isinstance(reform, dict):
                reform = Reform.from_dict(reform)
            reform.apply(self.tax_benefit_system)
        # Invalidate every cached value so the simulation recomputes under
        # the new tax-benefit system. Previously ``apply_reform`` left
        # ``_fast_cache``, in-memory holder storage, and on-disk holder
        # storage populated with pre-reform values, so the next
        # ``calculate`` call returned stale values (bug H3).
        self._invalidate_all_caches()

    def _invalidate_all_caches(self) -> None:
        """Purge cached formula output, preserving user-provided inputs.

        Called after ``apply_reform`` and any other operation that changes
        the tax-benefit system underneath an already-calculated simulation.

        Still-present supplied inputs are preserved from the tiers that
        hold them, so a structural reform applied after dataset load
        doesn't silently discard the dataset. A cache write replacing an
        input does not make that cached value a supplied input. Everything
        else (formula outputs, cached short-path results, on-disk caches) is
        wiped so the next ``calculate`` recomputes under the new
        tax-benefit system.
        """
        self.invalidated_caches = set()
        # Snapshot user-provided inputs before wiping so they can be
        # replayed into the fresh storage. Use the storage API instead of
        # hand-building keys, since ETERNITY variables canonicalize every
        # period to the single ETERNITY storage key.
        preserved: list[PreservedUserInput] = []
        user_input_keys = getattr(self, "_user_input_keys", None) or set()
        for variable_name, branch_name, period in user_input_keys:
            holder = self.get_holder(variable_name)
            holder._seed_input_storage(period, branch_name)
            locations = holder._user_input_storage[(branch_name, str(period))][1]
            # Carry-over marks also include external cache writes. Replay
            # only tiers that still hold the recorded, supplied input.
            stored_value = (
                holder._memory_storage.get(period, branch_name)
                if "memory" in locations
                and not holder._memory_storage.is_derived(period, branch_name)
                else None
            )
            if stored_value is not None:
                preserved.append(
                    PreservedUserInput(
                        variable_name=variable_name,
                        branch_name=branch_name,
                        period=period,
                        value=stored_value,
                        storage="memory",
                    )
                )
                continue
            if (
                "disk" in locations
                and holder._disk_storage is not None
                and not holder._disk_storage.is_derived(period, branch_name)
            ):
                disk_period = (
                    periods.period(periods.ETERNITY)
                    if holder._disk_storage.is_eternal
                    else periods.period(period)
                )
                disk_key = f"{branch_name}_{disk_period}"
                disk_file = holder._disk_storage._files.get(disk_key)
                if disk_file is not None:
                    preserved.append(
                        PreservedUserInput(
                            variable_name=variable_name,
                            branch_name=branch_name,
                            period=period,
                            value=None,
                            storage="disk",
                            disk_key=disk_key,
                            disk_file=disk_file,
                            disk_enum=holder._disk_storage._enums.get(disk_file),
                        )
                    )
        self._user_input_keys = {
            (user_input.variable_name, user_input.branch_name, user_input.period)
            for user_input in preserved
        }
        # Iterate only over holders that already exist on each population —
        # lazy-creating a holder for every variable in the tax-benefit
        # system (thousands in policyengine-us) inflated the cost of
        # ``apply_reform`` from milliseconds to seconds and broke the
        # YAML full-suite on downstream repos. Untouched variables have
        # no holder and therefore nothing to wipe.
        for population in self.populations.values():
            for holder in population._holders.values():
                holder._memory_storage._arrays = {}
                holder._memory_storage._unmark_dropped_keys()
                holder._memory_storage._stop_sharing_dropped_keys()
                holder._memory_storage._forget_dropped_numbers()
                if holder._disk_storage is not None:
                    holder._disk_storage._files = {}
                    holder._disk_storage._derived = set()
                    holder._disk_storage._forget_dropped_keys()
                if hasattr(holder, "_user_input_storage"):
                    holder._user_input_storage = {}
        for frame in _calculation_frames.get():
            frame.untracked_changes = True
        self._drop_computed()
        # Replay preserved user inputs so ``calculate`` still sees them.
        for user_input in preserved:
            holder = self.get_holder(user_input.variable_name)
            if user_input.storage == "disk" and holder._disk_storage is not None:
                holder._disk_storage._files[user_input.disk_key] = user_input.disk_file
                holder._disk_storage._sequence_numbers[user_input.disk_key] = (
                    next_sequence_number()
                )
                if user_input.disk_enum is not None:
                    holder._disk_storage._enums[user_input.disk_file] = (
                        user_input.disk_enum
                    )
            else:
                holder._memory_storage.put(
                    user_input.value,
                    user_input.period,
                    user_input.branch_name,
                )
            holder._record_input_storage(
                user_input.period, user_input.branch_name, user_input.storage
            )
        for population in self.populations.values():
            for holder in population._holders.values():
                holder._record_inputs(None)
        for branch in self.branches.values():
            branch._invalidate_all_caches()

    def build_from_populations(self, populations: Dict[str, Population]) -> None:
        """This method of initialisation requires the populations to be pre-initialised.

        Args:
            populations (Dict[str, Population]): A dictionary of populations, indexed by entity key.
        """
        self.populations = populations
        self.link_to_entities_instances()
        self.create_shortcuts()

        self.populations = populations
        self.persons: Population = self.populations[
            self.tax_benefit_system.person_entity.key
        ]
        self.link_to_entities_instances()
        self.create_shortcuts()

    def build_from_dataset(self) -> None:
        """Build a simulation from a dataset."""
        self.build_from_populations(self.tax_benefit_system.instantiate_entities())
        from policyengine_core.simulations.simulation_builder import (
            SimulationBuilder,
        )  # Import here to avoid circular dependency

        builder = SimulationBuilder()
        builder.populations = self.populations

        try:
            data = self.dataset.load()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"The dataset file {self.dataset.name} could not be found. "
                + "Make sure you have downloaded or built it using the `policyengine-core data` command."
            ) from e

        if self.dataset.data_format == Dataset.FLAT_FILE:
            data_copy = {col: data[col].values for col in data.copy().columns}
            data = {col: data[col].values for col in data.columns}

        person_entity = self.tax_benefit_system.person_entity
        entity_id_field = f"{person_entity.key}_id"

        def get_eternity_array(name):
            if self.dataset.data_format == Dataset.FLAT_FILE:
                # Look for any column with variablename__timeperiod
                for col in data:
                    if col.split("__")[0] == name:
                        return data[col]
            elif self.dataset.data_format == Dataset.TIME_PERIOD_ARRAYS:
                return data[name][list(data[name].keys())[0]]
            return data[name]

        if self.dataset.data_format != Dataset.FLAT_FILE:
            assert entity_id_field in data, (
                f"Missing {entity_id_field} column in the dataset. Each person entity must have an ID array defined for ETERNITY."
            )
        elif entity_id_field not in data:
            data[entity_id_field] = np.arange(len(get_eternity_array("person_id")))

        entity_ids = get_eternity_array(entity_id_field)
        builder.declare_person_entity(person_entity.key, entity_ids)

        for group_entity in self.tax_benefit_system.group_entities:
            entity_id_field = f"{group_entity.key}_id"
            if self.dataset.data_format != Dataset.FLAT_FILE:
                assert entity_id_field in data, (
                    f"Missing {entity_id_field} column in the dataset. Each group entity must have an ID array defined for ETERNITY."
                )
                entity_ids = get_eternity_array(entity_id_field)
            elif entity_id_field not in data:
                entity_id_field_values = get_eternity_array(
                    f"person_{group_entity.key}_id"
                )
                if entity_id_field_values is not None:
                    entity_ids = np.arange(len(np.unique(entity_id_field_values)))
                else:
                    entity_ids = np.arange(len(data[list(data.keys())[0]]))

            builder.declare_entity(group_entity.key, entity_ids)

            person_membership_id_field = f"{person_entity.key}_{group_entity.key}_id"
            if self.dataset.data_format != Dataset.FLAT_FILE:
                assert person_membership_id_field in data, (
                    f"Missing {person_membership_id_field} column in the dataset. Each group entity must have a person membership array defined for ETERNITY."
                )
            elif person_membership_id_field not in data:
                data[person_membership_id_field] = np.arange(len(data))
            person_membership_ids = get_eternity_array(person_membership_id_field)

            person_role_field = f"{person_entity.key}_{group_entity.key}_role"
            if person_role_field in data:
                person_roles = get_eternity_array(person_role_field)
            elif "role" in data:
                person_roles = get_eternity_array("role")
            elif self.default_role is not None:
                person_roles = np.full(len(entity_ids), self.default_role)
            else:
                raise ValueError(
                    f"Missing {person_role_field} column in the dataset. Each group entity must have a person role array defined for ETERNITY."
                )
            builder.join_with_persons(
                self.populations[group_entity.key],
                person_membership_ids,
                person_roles,
            )

        self.build_from_populations(builder.populations)

        if self.dataset.data_format == Dataset.FLAT_FILE:
            # Ensure we're back to all person-level data.
            data = data_copy

        unknown_columns = []
        if self.dataset.data_format != Dataset.FLAT_FILE:
            for variable in data:
                if variable in self.tax_benefit_system.variables:
                    if self.dataset.data_format == Dataset.TIME_PERIOD_ARRAYS:
                        for time_period in data[variable]:
                            self.set_input(
                                variable,
                                time_period,
                                data[variable][time_period],
                            )
                    else:
                        self.set_input(
                            variable, self.dataset.time_period, data[variable]
                        )
                else:
                    unknown_columns.append(variable)
        else:
            for variable in data:
                if "__" in variable:
                    variable_name, time_period = variable.split("__")
                else:
                    variable_name = variable
                    time_period = self.dataset.time_period or self.default_input_period

                if variable_name not in self.tax_benefit_system.variables:
                    unknown_columns.append(variable)
                    continue

                variable_meta = self.tax_benefit_system.get_variable(variable_name)
                entity = variable_meta.entity
                population = self.get_population(entity.plural)

                # All data should be person level
                if len(data[variable]) != len(population.ids):
                    population: GroupPopulation
                    entity_level_data = population.value_from_first_person(
                        data[variable]
                    )
                else:
                    entity_level_data = data[variable]

                self.set_input(variable_name, time_period, entity_level_data)

        if unknown_columns:
            # A skipped column usually means the dataset was built for a
            # different model version (e.g. an input variable was renamed or
            # removed), and its data is silently lost — say so instead of
            # loading as if nothing happened.
            shown = ", ".join(sorted(unknown_columns)[:10])
            if len(unknown_columns) > 10:
                shown += f", … ({len(unknown_columns) - 10} more)"
            logging.warning(
                f"The dataset contains {len(unknown_columns)} column(s) that "
                f"do not match any variable in the tax-benefit system and "
                f"were ignored: {shown}"
            )

        self.default_calculation_period = (
            self.dataset.time_period or self.default_calculation_period
        )

        self.tax_benefit_system.data_modified = False

    @property
    def trace(self) -> bool:
        return self._trace

    @trace.setter
    def trace(self, trace: SimpleTracer) -> None:
        self._trace = trace
        if trace:
            self.tracer = FullTracer()
        else:
            self.tracer = SimpleTracer()

    def link_to_entities_instances(self) -> None:
        for _key, entity_instance in self.populations.items():
            entity_instance.simulation = self

    def _bind_to_tax_benefit_system(self) -> None:
        """Make this simulation's populations and holders use its own system.

        For a simulation given another system after its populations were
        built: a reform simulation's baseline branch is a branch under the
        reform's system that is then given the baseline's. Each population
        takes that system's entity, so it looks variables up there, and each
        holder takes that system's variable, so a variable the reform
        neutralized or redefined is the baseline's again. Holders, recorded
        inputs and ``input_variables`` entries are dropped for variables the
        system does not have (ones the reform added), and for variables whose
        holder was built for the reform's definition in a way a new variable
        cannot change: in another entity's population, or with storage for
        (or not for) an ``ETERNITY`` variable.
        """
        system = self.tax_benefit_system
        if system is None:
            # A reform simulation of a class with no default system instance
            # gets a baseline without a system; there is nothing to bind to.
            return
        entities = {
            entity.key: entity
            for entity in [system.person_entity, *system.group_entities]
        }
        variables = system.variables
        dropped = set()
        for population in self.populations.values():
            population.entity = entities[population.entity.key]
            for name, holder in list(population._holders.items()):
                variable = variables.get(name)
                if (
                    variable is None
                    or variable.entity.key != population.entity.key
                    or (variable.definition_period == ETERNITY)
                    != holder._memory_storage.is_eternal
                ):
                    del population._holders[name]
                    dropped.add(name)
                else:
                    holder.variable = variable

        def kept(name):
            return name in variables and name not in dropped

        if getattr(self, "_user_input_keys", None) is not None:
            self._user_input_keys = {
                key for key in self._user_input_keys if kept(key[0])
            }
        if getattr(self, "input_variables", None) is not None:
            self.input_variables = [name for name in self.input_variables if kept(name)]

    def create_shortcuts(self) -> None:
        for _key, population in self.populations.items():
            # create shortcut simulation.person and simulation.household (for instance)
            setattr(self, population.entity.key, population)

    @property
    def data_storage_dir(self) -> str:
        """
        Folder in which this simulation stores values on disk when memory is
        short (see ``MemoryConfig``).

        Set ``_data_storage_dir`` to choose the folder: it is created on first
        use if it does not exist. Nothing removes it,
        unless it is in a temporary folder a simulation made, which this
        simulation then keeps alive, and which removes it with everything else
        in it once nothing keeps it. Otherwise this is a new temporary folder,
        removed once this simulation, every disk storage in the folder (those
        its clones and branches copied included), every simulation given a
        folder in it and every temporary folder made in it are
        garbage-collected in this process, or at interpreter exit, except the
        subfolders of disk storages that preserve theirs. In a process forked
        from the one the folder is for, this is a new folder for that process,
        made inside that one. See the ``storage_directory`` module for what
        this guarantees.
        """
        pid = os.getpid()
        if self._data_storage_dir is not None and self._storage_dir_pid not in (
            None,
            pid,
        ):
            # Forked from the process the folder is for. Disk storages each
            # process made for a variable in one folder would write the same
            # files, and each would remove the other's when collected, so
            # this process stores new values in a folder of its own. It goes
            # inside the folder it inherited, so if the other process made
            # that one, it removes this one with it, however this process
            # ends. The disk storages this process has copies of still read
            # the other's files (see ``OnDiskStorage._path_to_write``).
            inherited = self._data_storage_dir
            self._data_storage_dir = None
            self._storage_directory = None
            if os.path.isdir(inherited):
                self._storage_dir_parent = inherited
        if self._data_storage_dir is None:
            if self._storage_dir_parent is not None:
                os.makedirs(self._storage_dir_parent, exist_ok=True)
            self._storage_directory = TemporaryStorageDirectory(
                self._storage_dir_parent
            )
            self._data_storage_dir = self._storage_directory.path
        elif not self._made_data_storage_dir():
            os.makedirs(self._data_storage_dir, exist_ok=True)
        self._storage_dir_pid = pid
        return self._data_storage_dir

    def _made_data_storage_dir(self) -> bool:
        """Whether this simulation made the folder it stores values on disk in."""
        return (
            self._storage_directory is not None
            and self._storage_directory.path == self._data_storage_dir
        )

    # ----- Calculation methods ----- #

    def calculate(
        self,
        variable_name: str,
        period: Period = None,
        map_to: str = None,
        decode_enums: bool = False,
    ) -> ArrayLike:
        """Calculate ``variable_name`` for ``period``.

        Args:
            variable_name (str): The name of the variable to calculate.
            period (Period): The period to calculate the variable for.
            map_to (str): The name of the variable to map the result to. If None, the result is returned as is.
            decode_enums (bool): If True, the result is decoded from an array of integers to an array of strings.

        Returns:
            ArrayLike: The calculated variable.
        """

        if period is not None and not isinstance(period, Period):
            period = periods.period(period)
        elif period is None and self.default_calculation_period is not None:
            period = periods.period(self.default_calculation_period)

        # Fast path: skip tracer, random seed and all _calculate() machinery for
        # already-computed values. map_to and decode_enums are NOT cached here —
        # they are post-processing steps that vary per call site.
        if map_to is None and not decode_enums and not getattr(self, "trace", False):
            _fast_key = (variable_name, period)
            _fast_cache = getattr(self, "_fast_cache", None)
            if _fast_cache is not None:
                _cached = _fast_cache.get(_fast_key)
                if _cached is not None:
                    self._share_store_history_with_caller()
                    frames = _calculation_frames.get()
                    if not frames:
                        _hand_to_waiting_ancestors(self, self._input_epoch, ())
                    elif frames[-1].simulation is not self:
                        frames[-1].reads.setdefault(self, self._input_epoch)
                    return _cached

        self.tracer.record_calculation_start(variable_name, period, self.branch_name)

        # No per-variable RNG seeding: formulas may not use randomness at all
        # (enforced statically at variable registration by
        # check_formula_determinism), so there is nothing to make reproducible
        # here.

        # Formulas running in this simulation may hold values they read in
        # local variables; a drop meanwhile must not forget what they came
        # from (see ``_drop_computed``).
        # Only the outermost calculation in this simulation runs again after
        # an input change here: running it again runs the inner ones too.
        outermost = not self._calculations_in_flight
        self._calculations_in_flight += 1
        frame = _frame_for(self, outermost)
        if frame is not None:
            frame.__enter__()
        try:
            result = self._calculate(variable_name, period)
            previous_transition = None
            for _ in range(_RERUNS_AFTER_INPUT_CHANGE if outermost else 0):
                if not frame.is_stale():
                    break
                transition = frame.transition_signature(result)
                if transition is not None and transition == previous_transition:
                    frame.settle_reads()
                    # This run repeated the complete input transition and its
                    # result. Preserve holder input/blacklist rules and the
                    # existing fast-cache guard when keeping the fixed point.
                    keep_state = self._calculation_start()
                    result = self._cache_result(
                        self.get_holder(variable_name),
                        result,
                        period,
                        keep_state,
                    )
                    if self._may_keep(keep_state):
                        if hasattr(self, "_fast_cache"):
                            self._fast_cache[(variable_name, period)] = result
                        if self.check_macro_cache(variable_name, str(period)):
                            macro = SimulationMacroCache(self.tax_benefit_system)
                            macro.set_cache_path(
                                self.dataset.file_path.parent,
                                self.dataset.name,
                                variable_name,
                                str(period),
                                self.branch_name,
                            )
                            macro.set_cache_value(macro.get_cache_path(), result)
                    break
                previous_transition = transition
                # An input set while it ran (by a formula, say), here or in a
                # simulation it read from, dropped values, so the result was
                # not kept (``_cache_result``). Calculate it again from the new
                # inputs, until a run changes none, and keep that result, as a
                # simulation given those inputs first would.
                frame.restart()
                result = self._calculate(variable_name, period)
            # Satisfies ``requires_computation_after`` from now on, even if a
            # branch input later drops the values.
            requested = self._requested_variables
            if requested is None:
                requested = self._get_requested_variables()
            requested.add(variable_name)
            if isinstance(result, EnumArray) and decode_enums:
                result = result.decode_to_str()
            self.tracer.record_calculation_result(result)
            if map_to is not None:
                source_entity = self.tax_benefit_system.get_variable(
                    variable_name
                ).entity.key
                result = self.map_result(result, source_entity, map_to)
            return result
        finally:
            try:
                # Also when the calculation fails: whether it fails can depend
                # on what it read, and a calling formula may catch the error.
                self._share_store_history_with_caller()
            finally:
                self._calculations_in_flight -= 1
                if frame is not None:
                    frame.__exit__(None, None, None)
                self.tracer.record_calculation_end()
                self.purge_cache_of_invalid_values()

    def map_result(
        self,
        values: ArrayLike,
        source_entity: str,
        target_entity: str,
        how: str = None,
    ):
        """Maps values from one entity to another.

        Args:
            arr (np.array): The values in their original position.
            source_entity (str): The source entity key.
            target_entity (str): The target entity key.
            how (str, optional): A function to use when mapping. Defaults to None.

        Raises:
            ValueError: If an invalid (dis)aggregation function is passed.

        Returns:
            np.array: The mapped values.
        """
        entity_pop = self.populations[source_entity]
        target_pop = self.populations[target_entity]
        if (
            source_entity == "person"
            and target_entity in self.tax_benefit_system.group_entity_keys
        ):
            if how and how not in (
                "sum",
                "any",
                "min",
                "max",
                "all",
                "value_from_first_person",
            ):
                raise ValueError("Not a valid function.")
            return target_pop.__getattribute__(how or "sum")(values)
        elif (
            source_entity in self.tax_benefit_system.group_entity_keys
            and target_entity == "person"
        ):
            if not how:
                return entity_pop.project(values)
            if how == "mean":
                return entity_pop.project(values / entity_pop.nb_persons())
        elif source_entity == target_entity:
            return values
        else:
            return self.map_result(
                self.map_result(
                    values,
                    source_entity,
                    self.tax_benefit_system.person_entity.key,
                    how="mean",
                ),
                "person",
                target_entity,
                how="sum",
            )

    def calculate_dataframe(
        self,
        variable_names: List[str],
        period: Period = None,
        map_to: str = None,
    ) -> pd.DataFrame:
        """Calculate ``variable_names`` for ``period``.

        Args:
            variable_names (List[str]): A list of variable names to calculate.
            period (Period): The period to calculate for.

        Returns:
            pd.DataFrame: A dataframe containing the calculated variables.
        """
        if period is not None and not isinstance(period, Period):
            period = periods.period(period)
        elif period is None and self.default_calculation_period is not None:
            period = periods.period(self.default_calculation_period)

        # Check each variable exists
        for variable_name in variable_names:
            if variable_name not in self.tax_benefit_system.variables:
                raise ValueError(f"Variable {variable_name} does not exist.")
        df = pd.DataFrame()
        entities = [
            self.tax_benefit_system.get_variable(variable_name).entity.key
            for variable_name in variable_names
        ]
        # Check that all variables are from the same entity. If not, map values to the entity of the first variable.
        entity = map_to or entities[0]
        if not all(entity == e for e in entities):
            map_to = entity
        for variable_name in variable_names:
            df[variable_name] = self.calculate(variable_name, period, map_to)
        return df

    def _calculate(self, variable_name: str, period: Period = None) -> ArrayLike:
        """
        Calculate the variable ``variable_name`` for the period ``period``, using the variable formula if it exists.

        Args:
            variable_name (str): The name of the variable to calculate.
            period (Period): The period to calculate the variable for.

        Returns:
            ArrayLike: The calculated variable.
        """
        if variable_name not in self.tax_benefit_system.variables:
            raise ValueError(f"Variable {variable_name} does not exist.")
        input_state = self._calculation_start()
        # Lets a custom ``set_input`` handler tell whether it calculated.
        self._calculations_started = self._calculations_started + 1
        population = self.get_variable_population(variable_name)
        holder = population.get_holder(variable_name)
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )

        # Check if we've neutralized via parameters.
        try:
            if (
                variable.is_neutralized
                or self.tax_benefit_system.parameters(period).gov.abolitions[
                    variable.name
                ]
            ):
                return holder.default_array()
        except Exception as e:
            pass

        # First look for a value already cached
        cached_array = holder.get_array(period, self.branch_name)
        if cached_array is not None:
            return cached_array

        # Check if cache can be used, if available, check if path exists
        is_cache_available = self.check_macro_cache(variable_name, str(period))
        if is_cache_available:
            smc = SimulationMacroCache(self.tax_benefit_system)
            smc.set_cache_path(
                self.dataset.file_path.parent,
                self.dataset.name,
                variable_name,
                str(period),
                self.branch_name,
            )
            cache_path = smc.get_cache_path()
            if cache_path.exists():
                if not self.macro_cache_read or self.tax_benefit_system.data_modified:
                    value = None
                else:
                    value = smc.get_cache_value(cache_path)

                if value is not None:
                    # Served without being stored: record it, so values
                    # calculated from it count as later (see ``set_input``).
                    # What it was calculated from was never calculated here,
                    # so values from here on may depend on any input.
                    sequence_number = next_sequence_number()
                    holder._record_store(period, sequence_number)
                    self._get_store_history().record_unknown_sources(sequence_number)
                    return value

        if variable.requires_computation_after is not None:
            variables_in_stack = [node.get("name") for node in self.tracer.stack]
            variable_in_stack = (
                variable.requires_computation_after in variables_in_stack
            )
            required_is_known_periods = self.get_holder(
                variable.requires_computation_after
            ).get_known_periods()
            # A branch input may have dropped the prerequisite's values; it
            # was still requested.
            required_was_requested = (
                variable.requires_computation_after in self._get_requested_variables()
            )
            if (
                (not variable_in_stack)
                and (not len(required_is_known_periods) > 0)
                and not required_was_requested
            ):
                raise ValueError(
                    f"Variable {variable_name} requires {variable.requires_computation_after} to be requested first. That variable is known in: {required_is_known_periods}. The full stack is: {variables_in_stack}. {variable_in_stack, len(required_is_known_periods) > 0}"
                )
        alternate_period_handling = False
        if variable.definition_period == MONTH and period.unit == YEAR:
            if variable.quantity_type == QuantityType.STOCK:
                contained_months = period.get_subperiods(MONTH)
                values = self._calculate(variable_name, contained_months[-1])
            else:
                values = self.calculate_add(variable_name, period)
            alternate_period_handling = True
        elif variable.definition_period == YEAR and period.unit == MONTH:
            alternate_period_handling = True
            if variable.quantity_type == QuantityType.STOCK:
                values = self._calculate(variable_name, period.this_year)
            else:
                values = self.calculate_divide(variable_name, period)

        if alternate_period_handling:
            if is_cache_available and self._may_keep(input_state):
                smc.set_cache_value(cache_path, values)
            return values

        self._check_period_consistency(period, variable)

        if variable.defined_for is not None:
            # Registration rejects a non-numeric defined_for variable (see
            # ``TaxBenefitSystem._check_defined_for``). This catches one set,
            # or a variable replaced, afterwards. It reads the variable's
            # type, not the values: mapped to a group entity, an Enum's
            # indices are summed into numbers, and str or date values fail
            # inside the mapping.
            defined_for_variable = self.tax_benefit_system.get_variable(
                variable.defined_for
            )
            if defined_for_variable is not None:
                variable.check_defined_for_variable(defined_for_variable)
            mask = (
                self.calculate(variable.defined_for, period, map_to=variable.entity.key)
                > 0
            )
            if np.all(~mask):
                array = holder.default_array()
                array = self._cast_formula_result(array, variable)
                return self._cache_result(holder, array, period, input_state)

        array = None

        # First, try to run a formula
        try:
            self._check_for_cycle(variable.name, period)
            token = _formula_simulation.set(self)
            try:
                array = self._run_formula(variable, population, period)
            finally:
                _formula_simulation.reset(token)

            # If no result, use the default value and cache it
            if array is None:
                if variable.uprating is not None or (
                    self.tax_benefit_system.auto_carry_over_input_variables
                    and variable.calculate_output is None
                ):
                    # The value is uprated or carried over from another
                    # period, or defaults for lack of one: an input set later
                    # for another period of the variable can change it.
                    self._get_store_history().record_derived(
                        variable_name, next_sequence_number()
                    )
                # Check if the variable has a previously defined value
                known_periods = holder.get_known_periods()
                earlier_known_periods = [
                    known_period
                    for known_period in known_periods
                    if known_period.unit == variable.definition_period
                    and known_period.start < period.start
                ]
                # Uprate only from an input this branch reads (see
                # ``Holder.get_input_periods``). Every value this simulation
                # calculated is marked derived when cached, and uprating from
                # one would make the result depend on which periods were
                # calculated first: an integer truncated, or a float32
                # rounded, at an intermediate period would compound, and a
                # value masked by ``defined_for``, a default, or a value
                # carried from another unit, cached there, would replace the
                # input. A period stored only under a branch this one cannot
                # read would read back as ``None``.
                earlier_input_periods = []
                if variable.uprating is not None:
                    input_periods = set(holder.get_input_periods(self.branch_name))
                    earlier_input_periods = [
                        known_period
                        for known_period in earlier_known_periods
                        if known_period in input_periods
                    ]
                if earlier_input_periods:
                    # Registration rejects these; an ``uprating`` assigned
                    # past the setter gets the same message here.
                    variable.check_uprating_value_type()
                    # Take the latest period from the filtered list itself.
                    # Indexing ``known_periods`` with a position in the
                    # filtered list picked the wrong period whenever a later
                    # one was stored first. Two inputs can start on the same
                    # day: a yearly variable stores a ``year:2012:2`` input as
                    # given, beside one for ``2012``. Break that tie as
                    # auto-carry-over does (the one that ends last), so the
                    # source never depends on which was stored first. The
                    # factor below runs from the source's start either way.
                    latest_known_period = max(
                        earlier_input_periods,
                        key=lambda p: _latest_input_key(p, variable.definition_period),
                    )
                    try:
                        uprating_parameter = get_parameter(
                            self.tax_benefit_system.parameters,
                            variable.uprating,
                        )
                    except:
                        raise ValueError(
                            f"Could not find uprating parameter {variable.uprating} when trying to uprate {variable_name}."
                        )
                    value_in_last_period = _uprating_index_value(
                        uprating_parameter, latest_known_period.start
                    )
                    value_in_this_period = _uprating_index_value(
                        uprating_parameter, period.start
                    )
                    if (
                        value_in_last_period is None
                        or value_in_this_period is None
                        or value_in_last_period == 0
                    ):
                        uprating_factor = 1
                    else:
                        uprating_factor = value_in_this_period / value_in_last_period

                    array = (
                        holder.get_array(latest_known_period, self.branch_name)
                        * uprating_factor
                    )
                elif (
                    self.tax_benefit_system.auto_carry_over_input_variables
                    and variable.calculate_output is None
                    and len(known_periods) > 0
                ):
                    # Carry over the latest input: of the stored periods that
                    # start no later than ``period``, the one that starts last
                    # (on a tie, the one that ends last, then the larger unit;
                    # see ``_latest_input_key``), preferring periods at the
                    # variable's own definition-period unit and using another
                    # unit only when there is none, as for an input to a
                    # variable with no ``set_input`` helper.
                    #
                    # Only inputs carry. Every value this simulation
                    # calculated is marked derived when cached (formula
                    # results, carried, uprated and default values, a twelfth
                    # cached by ``calculate_divide``, a sum cached by
                    # ``calculate_add``), and carrying one would make the
                    # result depend on what was calculated first: a later
                    # period's carried value would hide an earlier input, and
                    # a value already masked by ``defined_for``, or given by a
                    # formula that has since ended, would carry forward.
                    # A later input does not carry backwards.
                    last_known_period = max(
                        (
                            input_period
                            for input_period in holder.get_input_periods(
                                self.branch_name
                            )
                            if input_period.start <= period.start
                        ),
                        key=lambda p: _latest_input_key(p, variable.definition_period),
                        default=None,
                    )
                    if last_known_period is not None:
                        # Pass branch_name through so auto-carry-over respects
                        # the active branch instead of reaching for the
                        # "default" branch's cache (bug H2).
                        array = holder.get_array(last_known_period, self.branch_name)
                    elif any(
                        known_period.start > period.start
                        for known_period in known_periods
                    ):
                        # No input to carry, but a later period is stored: as
                        # before, return the default without caching it. A
                        # cached default would change what a formula testing
                        # whether a value is stored sees. (The uprating path
                        # above skips derived periods, so it would not uprate
                        # from one.)
                        return holder.default_array()
                    else:
                        array = holder.default_array()
                else:
                    array = holder.default_array()

            if variable.defined_for is not None:
                array = np.where(mask, array, variable.default_value)
                if variable.value_type == Enum:
                    array = np.array(
                        [
                            item.index if isinstance(item, Enum) else item
                            for item in array
                        ]
                    )
                    array = EnumArray(array, variable.possible_values)

            array = self._cast_formula_result(array, variable)
            array = self._cache_result(holder, array, period, input_state)

        except SpiralError:
            array = holder.default_array()
            # Not stored, but what reads it is (see ``set_input``).
            holder._record_store(period, next_sequence_number())
        except RecursionError as e:
            if isinstance(self.tracer, FullTracer):
                self.tracer.print_computation_log()
            stack = self.tracer.stack
            stack_formatted = "\n".join(
                [
                    f"  - {node.get('name')} {node.get('period')}, {node.get('branch_name')}"
                    for node in stack
                ]
            )
            raise Exception(
                f"RecursionError while calculating {variable_name} for period {period}. The full computation stack is:\n{stack_formatted}"
            )

        # Neither cache keeps a result an input change may have made obsolete
        # (see ``_cache_result``).
        unchanged = self._may_keep(input_state)
        if is_cache_available and unchanged:
            smc.set_cache_value(cache_path, array)

        if hasattr(self, "_fast_cache") and unchanged:
            self._fast_cache[(variable_name, period)] = array

        return array

    def _may_keep(self, input_state: Tuple[int, int, int]) -> bool:
        """Whether a result calculated since ``input_state`` may be kept (see ``_cache_result``)."""
        if input_state[0] != self._input_epoch or input_state[1] != self._inputs_set:
            return False
        frames = _calculation_frames.get()
        return not frames or frames[-1].may_keep()

    def _calculation_start(self) -> Tuple[int, int, int]:
        """When a calculation begins: how many drops ran here (``_input_epoch``)
        and inputs were set here (``_inputs_set``), and a sequence number."""
        return self._input_epoch, self._inputs_set, next_sequence_number()

    def _input_set_meanwhile(
        self, holder: Holder, period: Period, started_at: int
    ) -> Optional[ArrayLike]:
        """The input this simulation reads for ``period``, if it was stored after ``started_at``.

        That is an input set while the calculation that began at
        ``started_at`` ran (by its own formula, say), under any branch name
        the simulation reads.
        """
        stored_on = holder._branch_storing(period, self.branch_name)
        if (
            stored_on is not None
            and holder._is_input(period, stored_on)
            and (holder._stored_sequence_number(period, stored_on) or 0) > started_at
        ):
            return holder._get_array_from_storage(period, stored_on)
        return None

    def _cache_result(
        self,
        holder: Holder,
        array: ArrayLike,
        period: Period,
        input_state: Tuple[int, int, int],
    ) -> ArrayLike:
        """Cache a calculated value, and return the value to use for it.

        ``input_state`` is :meth:`_calculation_start` when the calculation began.
        If an input for the same period was set meanwhile (by the formula
        itself, say), that input is the value, as it would be had it been set
        first. A calculation that was running when an input set on this
        simulation dropped values may have read the replaced value, so its
        result is returned but not kept (see ``_drop_computed``).
        """
        epoch, inputs_set, started_at = input_state
        if inputs_set != self._inputs_set:
            stored_input = self._input_set_meanwhile(holder, period, started_at)
            if stored_input is not None:
                return stored_input
        frames = _calculation_frames.get()
        may_keep = not frames or frames[-1].may_keep()
        if epoch == self._input_epoch and may_keep:
            holder.put_in_cache(array, period, self.branch_name, derived=True)
        else:
            # Not kept, but whatever reads it is stored after it all the same.
            holder._record_store(period, next_sequence_number())
        return array

    def purge_cache_of_invalid_values(self) -> None:
        # We wait for the end of calculate(), signalled by an empty stack, before purging the cache
        if self.tracer.stack:
            return
        _fast_cache = getattr(self, "_fast_cache", None)
        invalidated_caches = getattr(self, "invalidated_caches", None)
        if invalidated_caches is None:
            return
        for _name, _period in invalidated_caches:
            holder = self.get_holder(_name)
            holder.delete_arrays(_period)
            if _fast_cache is not None:
                _fast_cache.pop((_name, _period), None)
        self.invalidated_caches = set()

    def calculate_add(
        self,
        variable_name: str,
        period: Period = None,
        decode_enums: bool = False,
    ) -> ArrayLike:
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )

        if period is not None and not isinstance(period, Period):
            period = periods.period(period)

        # Check that the requested period matches definition_period
        if periods.unit_weight(variable.definition_period) > periods.unit_weight(
            period.unit
        ):
            raise ValueError(
                "Unable to compute variable '{0}' for period {1}: '{0}' can only be computed for {2}-long periods. You can use the DIVIDE option to get an estimate of {0} by dividing the yearly value by 12, or change the requested period to 'period.this_year'.".format(
                    variable.name, period, variable.definition_period
                )
            )

        if variable.definition_period not in [
            periods.DAY,
            periods.MONTH,
            periods.YEAR,
        ]:
            raise ValueError(
                "Unable to sum constant variable '{}' over period {}: only variables defined daily, monthly, or yearly can be summed over time.".format(
                    variable.name, period
                )
            )

        sub_periods = list(period.get_subperiods(variable.definition_period))

        def total():
            return sum(
                self.calculate(variable_name, sub_period) for sub_period in sub_periods
            )

        # As in ``calculate``: only an outermost sum runs again, and while it
        # sums it is in flight, so its terms do not run again on their own
        # (and a drop meanwhile keeps the records of what they read).
        outermost = not self._calculations_in_flight
        self._calculations_in_flight += 1
        frame = _frame_for(self, outermost)
        try:
            with frame if frame is not None else nullcontext():
                input_state = self._calculation_start()
                result = total()
                previous_transition = None
                for _ in range(_RERUNS_AFTER_INPUT_CHANGE if outermost else 0):
                    if not frame.is_stale():
                        break
                    transition = frame.transition_signature(result)
                    if transition is not None and transition == previous_transition:
                        frame.settle_reads()
                        break
                    previous_transition = transition
                    # An input changed while summing: earlier terms may be
                    # obsolete (see ``calculate``). Sum again.
                    frame.restart()
                    input_state = self._calculation_start()
                    result = total()
                return self._cache_option_result(variable, period, result, input_state)
        finally:
            self._calculations_in_flight -= 1

    def calculate_divide(
        self,
        variable_name: str,
        period: Period = None,
        decode_enums: bool = False,
    ) -> ArrayLike:
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )

        if period is not None and not isinstance(period, Period):
            period = periods.period(period)

        # Check that the requested period matches definition_period
        if variable.definition_period != periods.YEAR:
            raise ValueError(
                "Unable to divide the value of '{}' over time on period {}: only variables defined yearly can be divided over time.".format(
                    variable_name, period
                )
            )

        if period.size != 1:
            raise ValueError(
                "DIVIDE option can only be used for a one-year or a one-month requested period"
            )

        if period.unit == periods.MONTH:
            frame = _frame_for(self, outermost=False)
            with frame if frame is not None else nullcontext():
                input_state = self._calculation_start()
                computation_period = period.this_year
                result = self.calculate(variable_name, period=computation_period) / 12.0
                return self._cache_option_result(variable, period, result, input_state)
        elif period.unit == periods.YEAR:
            return self.calculate(variable_name, period)

        raise ValueError(
            "Unable to divide the value of '{}' to match period {}.".format(
                variable_name, period
            )
        )

    def _cache_option_result(
        self,
        variable: Variable,
        period: Period,
        result: ArrayLike,
        input_state: Tuple[int, int, int],
    ) -> ArrayLike:
        """Cache an ADD or DIVIDE result at ``period`` if a plain read would return it.

        A value cached at ``period`` is what every later ``calculate`` of the
        variable at ``period`` returns. ``_calculate`` computes a FLOW
        variable over a period of another unit with these same options (a
        monthly variable over a year with ``calculate_add``, a yearly one over
        a month with ``calculate_divide``), so their result is the plain value
        there and is cached. Anywhere else it is not:

        - A STOCK variable's plain value over a year is its last month's, and
          over a month the year's, not the sum or the twelfth.
        - Over several periods of the variable's own unit, a plain read
          raises instead.
        - A day variable's plain read over a month or a year does not sum.
        - Over a single period of its own unit, the sum is the value
          ``calculate`` has already stored.

        Caching there would make a later plain read depend on whether the
        option ran first. Nor is an option result cached when storing it would
        change it (the twelfth of an integer or a count of true months is
        stored as the variable's own type, so a later read would return the
        truncated value where the first returned the exact one).

        The result is cached as derived, so auto-carry-over and uprating
        never take it for an input, and ``put_in_cache`` keeps an input a
        plain read finds at ``period`` (in this branch, an ancestor or
        ``default``) instead of storing it. It does replace a value
        calculated there before, which may predate a change to the inputs.
        Whether the value read is an input comes from the storage's mark for
        the first visible value, which every write sets. The simulation's
        supplied-input record follows inputs across storage tiers for replay
        and export; it can name a disk input hidden by a derived memory value,
        and raw holder inputs need not be registered there. Accepted cache
        writes never register the aggregate as a supplied input.
        """
        if variable.quantity_type == QuantityType.STOCK:
            return result
        routed = (variable.definition_period == MONTH and period.unit == YEAR) or (
            variable.definition_period == YEAR and period.unit == MONTH
        )
        if not routed or np.asarray(result).dtype != variable.dtype:
            return result
        return self._cache_result(
            self.get_holder(variable.name), result, period, input_state
        )

    def calculate_output(self, variable_name: str, period: Period = None) -> ArrayLike:
        """
        Calculate the value of a variable using the ``calculate_output`` attribute of the variable.
        """

        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )

        if variable.calculate_output is None:
            return self.calculate(variable_name, period)

        return variable.calculate_output(self, variable_name, period)

    def _run_formula(
        self, variable: str, population: Population, period: Period
    ) -> ArrayLike:
        """
        Find the ``variable`` formula for the given ``period`` if it exists, and apply it to ``population``.
        """

        formula = variable.get_formula(period)
        if formula is None:
            values = None
            if variable.adds is not None and len(variable.adds) > 0:
                if isinstance(variable.adds, str):
                    try:
                        adds_parameter = get_parameter(
                            self.tax_benefit_system.parameters,
                            variable.adds,
                        )
                    except:
                        raise ValueError(
                            f"In the variable '{variable.name}', the 'adds' attribute is a string '{variable.adds}' that does not match any parameter."
                        )
                    adds_list = adds_parameter(period.start)
                else:
                    adds_list = variable.adds
                values = 0
                for added_variable in adds_list:
                    if added_variable in self.tax_benefit_system.variables:
                        values = values + self.calculate(
                            added_variable, period, map_to=variable.entity.key
                        )
                    else:
                        try:
                            parameter = get_parameter(
                                self.tax_benefit_system.parameters,
                                added_variable,
                            )
                            values = values + parameter(period.start)
                        except:
                            raise ValueError(
                                f"In the variable '{variable.name}', the 'adds' attribute is a list that contains a string '{added_variable}' that does not match any variable or parameter."
                            )
            if variable.subtracts is not None and len(variable.subtracts) > 0:
                if isinstance(variable.subtracts, str):
                    try:
                        subtracts_parameter = get_parameter(
                            self.tax_benefit_system.parameters,
                            variable.subtracts,
                        )
                    except:
                        raise ValueError(
                            f"In the variable '{variable.name}', the 'subtracts' attribute is a string '{variable.subtracts}' that does not match any parameter."
                        )
                    subtracts_list = subtracts_parameter(period.start)
                else:
                    subtracts_list = variable.subtracts
                if values is None:
                    values = 0
                for subtracted_variable in subtracts_list:
                    if subtracted_variable in self.tax_benefit_system.variables:
                        values = values - self.calculate(
                            subtracted_variable,
                            period,
                            map_to=variable.entity.key,
                        )
                    else:
                        try:
                            parameter = get_parameter(
                                self.tax_benefit_system.parameters,
                                subtracted_variable,
                            )
                            values = values - parameter(period.start)
                        except:
                            raise ValueError(
                                f"In the variable '{variable.name}', the 'subtracts' attribute is a list that contains a string '{subtracted_variable}' that does not match any variable or parameter."
                            )
            return values

        parameters_at = self.tax_benefit_system.parameters
        if self.trace:
            # Trace through a view of the parameter tree that belongs to this
            # call. Switching tracing on in the tree itself would leave the
            # shared tax-benefit system traced for every simulation, branch
            # and clone that uses it afterwards, with this simulation's tracer
            # and branch name cached in the tree's nodes at each instant.
            parameters_at = TracingParameterNode(
                parameters_at, self.tracer, self.branch_name
            )

        # A rules-engine formula must be a pure, deterministic function of its
        # inputs. Randomness is forbidden statically at variable registration
        # (check_formula_determinism), so no runtime guard is needed here.
        if formula.__code__.co_argcount == 2:
            array = formula(population, period)
        else:
            array = formula(population, period, parameters_at)

        return array

    def _check_period_consistency(self, period: Period, variable: Variable) -> None:
        """
        Check that a period matches the variable definition_period
        """
        if variable.definition_period == periods.ETERNITY:
            return  # For variables which values are constant in time, all periods are accepted

        if variable.definition_period == periods.MONTH and period.unit != periods.MONTH:
            raise ValueError(
                "Unable to compute variable '{0}' for period {1}: '{0}' must be computed for a whole month. You can use the ADD option to sum '{0}' over the requested period, or change the requested period to 'period.first_month'.".format(
                    variable.name, period
                )
            )

        if variable.definition_period == periods.YEAR and period.unit != periods.YEAR:
            raise ValueError(
                "Unable to compute variable '{0}' for period {1}: '{0}' must be computed for a whole year. You can use the DIVIDE option to get an estimate of {0} by dividing the yearly value by 12, or change the requested period to 'period.this_year'.".format(
                    variable.name, period
                )
            )

        if period.size != 1:
            raise ValueError(
                "Unable to compute variable '{0}' for period {1}: '{0}' must be computed for a whole {2}. You can use the ADD option to sum '{0}' over the requested period.".format(
                    variable.name,
                    period,
                    (
                        "month"
                        if variable.definition_period == periods.MONTH
                        else "year"
                    ),
                )
            )

    def _cast_formula_result(self, value: Any, variable: str) -> ArrayLike:
        if variable.value_type == Enum and not isinstance(value, EnumArray):
            return variable.possible_values.encode(value)

        if not isinstance(value, np.ndarray):
            population = self.get_variable_population(variable.name)
            value = population.filled_array(value)

        if value.dtype != variable.dtype:
            return value.astype(variable.dtype)

        return value

    # ----- Handle circular dependencies in a calculation ----- #

    def _check_for_cycle(self, variable: str, period: Period) -> None:
        """
        Raise an exception in the case of a circular definition, where evaluating a variable for
        a given period loops around to evaluating the same variable/period pair. Also guards, as
        a heuristic, against "quasicircles", where the evaluation of a variable at a period involves
        the same variable at a different period.
        """
        # The last frame is the current calculation, so it should be ignored from cycle detection
        previous_periods = [
            frame["period"]
            for frame in self.tracer.stack[:-1]
            if frame["name"] == variable and frame["branch_name"] == self.branch_name
        ]
        if period in previous_periods:
            found_last_frame = False
            i = -2
            while not found_last_frame:
                frame = self.tracer.stack[i]
                if (
                    frame["name"] == variable
                    and frame["branch_name"] == self.branch_name
                ):
                    found_last_frame = True
                i -= 1
            raise CycleError(
                f"Circular definition detected on formula {variable}@{period}. The circle is:\n\nNormal computation tree:\n"
                + "\n".join(
                    f"  {frame['name']}@{frame['period']} (branch {frame['branch_name']})"
                    for frame in self.tracer.stack[:i]
                )
                + "\n\nCycle start:\n"
                + "\n".join(
                    f"  >> {frame['name']}@{frame['period']} (branch {frame['branch_name']})"
                    for frame in self.tracer.stack[i:]
                )
            )
        spiral = len(previous_periods) >= self.max_spiral_loops
        if spiral:
            self.invalidate_spiral_variables(variable)
            message = "Quasicircular definition detected on formula {}@{} involving {}".format(
                variable, period, self.tracer.stack
            )
            raise SpiralError(message, variable)

    def invalidate_cache_entry(self, variable: str, period: Period) -> None:
        invalidated_caches = getattr(self, "invalidated_caches", None)
        if invalidated_caches is None:
            self.invalidated_caches = {(variable, period)}
            return
        invalidated_caches.add((variable, period))

    def invalidate_spiral_variables(self, variable: str) -> None:
        # Visit the stack, from the bottom (most recent) up; we know that we'll find
        # the variable implicated in the spiral (max_spiral_loops+1) times; we keep the
        # intermediate values computed (to avoid impacting performance) but we mark them
        # for deletion from the cache once the calculation ends.
        count = 0
        for frame in reversed(self.tracer.stack):
            self.invalidate_cache_entry(frame["name"], frame["period"])
            if frame["name"] == variable:
                count += 1
                if count > self.max_spiral_loops:
                    break

    # ----- Methods to access stored values ----- #

    def get_array(self, variable_name: str, period: Period) -> ArrayLike:
        """
        Return the value of ``variable_name`` for ``period``, if this value is alreay in the cache (if it has been set as an input or previously calculated).

        Unlike :meth:`.calculate`, this method *does not* trigger calculations and *does not* use any formula.
        """
        if period is not None and not isinstance(period, Period):
            period = periods.period(period)
        return self.get_holder(variable_name).get_array(period, self.branch_name)

    def get_holder(self, variable_name: str) -> Holder:
        """
        Get the :obj:`.Holder` associated with the variable ``variable_name`` for the simulation
        """
        return self.get_variable_population(variable_name).get_holder(variable_name)

    def get_memory_usage(self, variables: List[str] = None) -> dict:
        """
        Get data about the virtual memory usage of the simulation
        """
        result = dict(total_nb_bytes=0, by_variable={})
        for entity in self.populations.values():
            entity_memory_usage = entity.get_memory_usage(variables=variables)
            result["total_nb_bytes"] += entity_memory_usage["total_nb_bytes"]
            result["by_variable"].update(entity_memory_usage["by_variable"])
        return result

    # ----- Misc ----- #

    def delete_arrays(self, variable: str, period: Period = None) -> None:
        """
        Delete a variable's values visible to this simulation branch.

        The calling branch, each ancestor branch, and the default branch are
        purged from this simulation's private holder storage. Other branch
        names and the parent simulation's holder storage remain unchanged.
        Deleted inputs stop counting as inputs: a value calculated later for
        the same period is a formula result, which ``apply_reform`` discards
        and ``to_input_dataframe`` does not export.

        :param variable: the variable whose cached values should be deleted
        :param period: the period to delete, or all periods when omitted

        Example:

        >>> from policyengine_core.country_template import CountryTaxBenefitSystem
        >>> simulation = Simulation(CountryTaxBenefitSystem())
        >>> simulation.set_input('age', '2018-04', [12, 14])
        >>> simulation.set_input('age', '2018-05', [13, 14])
        >>> simulation.get_array('age', '2018-05')
        array([13, 14], dtype=int32)
        >>> simulation.delete_arrays('age', '2018-05')
        >>> simulation.get_array('age', '2018-04')
        array([12, 14], dtype=int32)
        >>> simulation.get_array('age', '2018-05') is None
        True
        >>> simulation.set_input('age', '2018-05', [13, 14])
        >>> simulation.delete_arrays('age')
        >>> simulation.get_array('age', '2018-04') is None
        True
        >>> simulation.get_array('age', '2018-05') is None
        True
        """
        holder = self.get_holder(variable)
        for branch_name in self._get_visible_branch_names():
            holder.delete_arrays(period, branch_name)
        _fast_cache = getattr(self, "_fast_cache", None)
        if period is None:
            if _fast_cache is not None:
                self._fast_cache = {
                    k: v for k, v in _fast_cache.items() if k[0] != variable
                }
        else:
            if not isinstance(period, Period):
                period = periods.period(period)
            if _fast_cache is not None:
                _fast_cache.pop((variable, period), None)

    def drop_computed_arrays(self) -> int:
        """Delete every value this simulation holds except inputs.

        Inputs are the values stored through ``set_input``: the dataset or
        situation the simulation was built from, inputs set on it and, for a
        branch, inputs set on the simulations it was created from before it
        was created. Values a custom ``set_input`` handler calculates are not
        inputs. Every other value is calculated again when next requested,
        and the simulation stops reading macro-cache files.

        Use this on a branch whose tax-benefit system or parameters differ
        from its parent's, whose values the branch would otherwise inherit;
        ``set_input`` on a branch drops what depends on the input by itself.
        Branches already created from this simulation keep their values.

        Returns:
            int: The number of arrays deleted.
        """
        # Recalculate rather than read macro-cache files written before.
        self.macro_cache_read = False
        for frame in _calculation_frames.get():
            frame.untracked_changes = True
        return self._drop_computed()

    def get_known_periods(self, variable: str) -> List[Period]:
        """
        Get a list variable's known period, i.e. the periods where a value has been initialized and

        :param variable: the variable to be set

        Example:

        >>> from policyengine_core.country_template import CountryTaxBenefitSystem
        >>> simulation = Simulation(CountryTaxBenefitSystem())
        >>> simulation.set_input('age', '2018-04', [12, 14])
        >>> simulation.set_input('age', '2018-05', [13, 14])
        >>> simulation.get_known_periods('age')
        [Period((u'month', Instant((2018, 5, 1)), 1)), Period((u'month', Instant((2018, 4, 1)), 1))]
        """
        return self.get_holder(variable).get_known_periods()

    def set_input(self, variable_name: str, period: Period, value: ArrayLike) -> None:
        """
        Set a variable's value for a given period

        :param variable: the variable to be set
        :param value: the input value for the variable
        :param period: the period for which the value is setted

        Example:
        >>> from policyengine_core.country_template import CountryTaxBenefitSystem
        >>> simulation = Simulation(CountryTaxBenefitSystem())
        >>> simulation.set_input('age', '2018-04', [12, 14])
        >>> simulation.get_array('age', '2018-04')
        array([12, 14], dtype=int32)

        If a ``set_input`` property has been set for the variable, this method may accept inputs for periods not matching the ``definition_period`` of the variable. To read more about this, check the `documentation <https://openfisca.org/doc/coding-the-legislation/35_periods.html#automatically-process-variable-inputs-defined-for-periods-not-matching-the-definitionperiod>`_.

        On a branch (see :meth:`get_branch`), the input also drops what the
        branch holds that may have been calculated from the value it
        replaces, so what the branch calculates next uses the input, as a
        simulation given the input before calculating anything would. Every
        value is stored after everything it was calculated from, and each
        simulation records the first stores its values may have been
        calculated from (see :mod:`policyengine_core.data_storage.store_history`):
        its own, its parent's when it was created, and those of simulations
        its formulas calculated in. The branch drops each value it holds,
        other than an input, stored at or after the earliest recorded store of
        ``variable_name`` for a period that shares a day with ``period``, or
        the earliest recorded value of ``variable_name`` uprated or carried
        over from another period (or given the default for want of one), or
        the first value that may depend on anything (restored from a dump,
        or calculated from a macro-cache read). If there is none of these,
        nothing it holds depends on the value and nothing is dropped. That is the case when a formula creates the branch while
        still calculating the variable it overrides, unless another branch
        it created for the same comparison already returned a value
        calculated from the variable; then only what came back from there,
        and what was calculated after, is dropped.

        Once an input is set on it, the branch stops reading macro-cache
        files, which are keyed by branch name and period but not by inputs.
        Each drop counts as an input change of the branch. A calculation
        that may have read a value from before the change, running in the
        branch or in any simulation that got a value from it, is not kept;
        the outermost such calculation in each simulation runs again until it
        reads nothing changed since (at most ten times; after that its result
        is returned but not kept), and an input stored for the very period it
        calculates after it began is its result. Setting an input to the
        value the branch already reads for that period, as an input, drops
        nothing.

        What this does not track: a branch given a different tax-benefit
        system or parameters (call :meth:`drop_computed_arrays` on it);
        formulas that write into an array they read instead of returning a
        new one; formulas that test whether a value is stored
        (``get_known_periods``, ``get_array``) or read another simulation's
        storage directly, rather than calculating; a simulation other than
        the formula's own branches calculated from a thread the formula
        starts without copying its context; and branches a formula keeps
        between calls, which hold what their parent held when they were
        created. On any simulation, input helpers remove overlapping
        calculated values of the variable they write. Dependency invalidation
        runs only on branches; branches already created from the branch keep
        their values; and a value another simulation calculated from the
        branch and kept stays when the branch's input changes after that
        calculation ended.
        """
        period = periods.period(period)
        if self.start_instant is None or self.start_instant > period.start:
            self.start_instant = period.start
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )
        if (variable.end is not None) and (period.start.date > variable.end):
            return
        self.get_holder(variable_name).set_input(period, value, self.branch_name)
        _fast_cache = getattr(self, "_fast_cache", None)
        if _fast_cache is not None:
            _fast_cache.pop((variable_name, period), None)

    def _get_store_history(self) -> StoreHistory:
        history = self._store_history
        if history is None:
            history = self._store_history = StoreHistory()
        return history

    def _get_requested_variables(self) -> set:
        requested = self._requested_variables
        if requested is None:
            requested = self._requested_variables = set()
        return requested

    def _share_store_history_with_caller(self) -> None:
        """Merge this simulation's store history into that of a formula calling it."""
        caller = _formula_simulation.get()
        if caller is not None:
            if caller is not self:
                caller._get_store_history().merge(self._get_store_history())
            return
        # No formula is visible here: the call comes from code outside any
        # formula, or from a thread a formula started without copying its
        # context. In the second case a formula in one of this simulation's
        # ancestors is waiting for the result, so give it to every ancestor
        # with a calculation running (more records only make drops broader).
        ancestor = getattr(self, "parent_branch", None)
        while ancestor is not None:
            if ancestor._calculations_in_flight:
                ancestor._get_store_history().merge(self._get_store_history())
            ancestor = getattr(ancestor, "parent_branch", None)

    def _drop_values_that_may_depend_on(
        self, variable_name: str, period: Period
    ) -> int:
        """On a branch, drop what may depend on ``variable_name`` at ``period``.

        Called before an input for ``variable_name`` at ``period`` is stored;
        see :meth:`set_input`. Returns the number of arrays dropped.
        """
        if getattr(self, "parent_branch", None) is None:
            return 0
        # (``Holder.set_input`` has already stopped macro-cache reads here.)
        since = self._get_store_history().earliest_dependency(
            variable_name, periods.period(period)
        )
        if self.get_holder(variable_name)._has_unnumbered_values():
            # Written into storage directly, so neither numbered nor recorded:
            # anything may have been calculated from it.
            since = 0
        if since is None:
            return 0
        return self._drop_computed(since)

    def _drop_computed(self, since: Optional[int] = None) -> int:
        """Drop every non-input value numbered ``since`` or later (all, without it)."""
        dropped = 0
        for population in self.populations.values():
            for holder in population._holders.values():
                dropped += holder._drop_computed(since)
        # The fast cache can also hold values a holder does not keep.
        self._fast_cache = {}
        # A calculation that got a value from here before the drop, running
        # here or in another simulation, may hold it or what it calculated
        # from it: none of them keeps its result (``_Frame.is_stale``), and
        # the outermost one in each simulation runs again (``calculate``).
        self._input_epoch = self._input_epoch + 1
        frames = _calculation_frames.get()
        if not frames:
            _taint_waiting_frames(self)
        for frame in frames:
            frame.drop_epochs.setdefault(self, set()).add(self._input_epoch)
        if self._calculations_in_flight:
            # A formula running here may still hold values calculated from
            # what the records describe: keep the records.
            return dropped
        # Nothing the simulation still holds was calculated from what the
        # records numbered ``since`` or later describe, except the inputs it
        # keeps, which are recorded again.
        self._get_store_history().prune(since)
        for population in self.populations.values():
            for holder in population._holders.values():
                holder._record_inputs(since)
        return dropped

    def get_variable_population(self, variable_name: str) -> Population:
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )
        return self.populations[variable.entity.key]

    def get_population(self, plural: str = None) -> Population:
        return next(
            (
                population
                for population in self.populations.values()
                if population.entity.plural == plural
            ),
            None,
        )

    def get_entity(self, plural: str = None) -> Entity:
        population = self.get_population(plural)
        return population and population.entity

    def describe_entities(self) -> dict:
        return {
            population.entity.plural: population.ids
            for population in self.populations.values()
        }

    def clone(
        self,
        debug: bool = False,
        trace: bool = False,
        clone_tax_benefit_system: bool = True,
    ) -> "Simulation":
        """
        Copy the simulation just enough to be able to run the copy without modifying the original simulation.

        The copy records its own cache invalidations (``invalidated_caches``,
        starting from this simulation's pending ones). If ``baseline`` is a
        branch of this simulation, as a reform simulation's is, the copy gets
        a copy of that branch as its own: a branch of the copy, under the same
        name, with this baseline's tax-benefit system, traced in the copy if
        this baseline is traced in this simulation. Any other ``baseline`` is
        shared: a branch's is its parent's, and a country package that builds
        a separate baseline simulation copies it, if it needs to, in its own
        ``clone``.

        Every cached array is copied, except in the first ``clone`` of this
        simulation made while ``get_branch`` is creating a branch of it: that
        copy shares the arrays until it reads them (see :meth:`get_branch`).
        A subclass's ``clone`` that calls this one through ``super().clone``
        takes part in that the same way. A subclass ``clone`` that first
        clones the same simulation directly gets the sharing in that direct
        clone instead, and its branch is a full copy.

        The copy's method aliases (``calc``, ``df``, and any other bound
        method of this simulation kept on the instance) are bound to the
        copy. With ``clone_tax_benefit_system``, the copy of the system names
        the copy as its simulation and the copy's populations use its
        entities, so nothing in the copy refers back to this simulation
        through them. Without it, the copy shares this simulation's system,
        which still names this simulation.
        """
        request = _branch_clone.get()
        share_arrays = (
            request is not None and request.pending and request.simulation is self
        )
        if share_arrays:
            request.pending = False
        new = commons.empty_clone(self)
        new_dict = new.__dict__

        for key, value in self.__dict__.items():
            if key not in (
                "debug",
                "trace",
                "tracer",
                "branches",
                "_fast_cache",
            ):
                new_dict[key] = value
        # Aliases of this simulation's methods (``calc`` and ``df``, and any a
        # subclass adds) are bound methods of this simulation, which the copy
        # above carried over as they were: the clone's ``calc`` calculated on
        # this simulation, and kept it alive. Bind each to the clone.
        for key, value in new_dict.items():
            if isinstance(value, types.MethodType) and value.__self__ is self:
                new_dict[key] = types.MethodType(value.__func__, new)
        new._fast_cache = {}
        # The clone stores what it puts on disk in a folder of its own, made
        # when first needed, never in this simulation's. Disk storages the
        # two each made for a variable in one folder would write the same
        # files (the clone keeps the branch name), and each would remove the
        # other's files when collected. The disk storages the clone copies
        # from this simulation's holders do read this simulation's files:
        # they keep its folder until they are collected (see
        # ``OnDiskStorage.clone``). A clone of a simulation given a folder
        # makes its own inside that one, and so does one made in a process
        # forked from the one this simulation's folder is for (see
        # ``data_storage_dir``). Until it makes its own, it keeps the
        # temporary folder it will make it in, if a simulation made that one.
        if self._data_storage_dir is not None and (
            not self._made_data_storage_dir()
            or self._storage_dir_pid not in (None, os.getpid())
        ):
            new._storage_dir_parent = self._data_storage_dir
        new._data_storage_dir = None
        new._storage_directory = None
        if new._storage_dir_parent is not None:
            new._storage_dir_keeper = directory_containing(new._storage_dir_parent)
        # Each simulation records its own inputs. A shared record let an
        # input set on one replay, in the other's ``apply_reform``, whatever
        # the other had calculated for that period, as an input.
        if hasattr(self, "_user_input_keys"):
            new._user_input_keys = set(self._user_input_keys)
        # Each records its own invalidations, too. With one set, a spiral in
        # one (in a branch, say) made the other delete, at its next purge,
        # its own cached values for those variables and periods, inputs
        # among them. Invalidations this simulation has not purged yet carry
        # over: the clone's cached arrays start as copies of its arrays.
        if getattr(self, "invalidated_caches", None) is not None:
            new.invalidated_caches = set(self.invalidated_caches)

        # Only pass ``share_arrays`` when sharing, so a population or holder
        # ``clone`` override with the earlier signature still deep-copies.
        sharing = {"share_arrays": True} if share_arrays else {}
        new.persons = self.persons.clone(new, **sharing)
        setattr(new, new.persons.entity.key, new.persons)
        new.populations = {new.persons.entity.key: new.persons}
        new.branches = {}

        for entity in self.tax_benefit_system.group_entities:
            population = self.populations[entity.key].clone(new, new.persons, **sharing)
            new.populations[entity.key] = population
            setattr(
                new, entity.key, population
            )  # create shortcut simulation.household (for instance)
        if clone_tax_benefit_system:
            system = self.tax_benefit_system.clone()
            new.tax_benefit_system = system
            # The copy of the system is the clone's alone. It names the clone
            # as its simulation, as a new simulation's system does, and the
            # clone's populations use its entities, so they look variables up
            # in it. They had kept this simulation's system: a variable a
            # reform added to the clone's was not found, and the clone kept
            # this simulation alive through it.
            system.simulation = new
            entities = {
                entity.key: entity
                for entity in [system.person_entity, *system.group_entities]
            }
            for population in new.populations.values():
                population.entity = entities[population.entity.key]
        else:
            new.tax_benefit_system = self.tax_benefit_system
        new.debug = debug
        new.trace = trace
        # The copy holds what this simulation holds, so it starts from what
        # those values may have been calculated from, and diverges from there.
        new._store_history = self._get_store_history().copy()
        new._requested_variables = set(self._get_requested_variables())
        # Calculations running in this simulation are not running in the copy.
        new._calculations_in_flight = 0
        new._open_frames = set()
        # A direct clone can retain its source's branch name and parent while
        # taking different inputs. ``get_branch`` marks the returned branch
        # after assigning its ancestry, for comparable recreated branches.
        new._fixed_point_branch = False
        # A ``set_input`` running on this simulation is not running on the copy.
        new._user_input_contexts = []

        # A branch shares its parent's baseline: formulas that run in a
        # branch read the parent's baseline values through it. That holds for
        # every clone of this simulation made while ``get_branch`` is making
        # a branch of it, not only the one that shares its arrays (a subclass
        # ``clone`` may clone it directly first; see ``_BranchClone``).
        branching = request is not None and request.simulation is self
        baseline = getattr(self, "baseline", None)
        if (
            not branching
            and baseline is not None
            and getattr(baseline, "parent_branch", None) is self
        ):
            # This simulation's own baseline branch (``__init__`` makes one
            # for a reform). A shared one filled this simulation's caches
            # with the copy's baseline calculations and took inputs set
            # through either, ``get_branch("baseline")`` on the copy made a
            # branch under the copy's (reform) policy, ``subsample`` of the
            # copy left its baseline at the old size, and the copy kept this
            # simulation alive through the branch's ``parent_branch``. The
            # copy gets a copy of the branch, not a new branch of itself as in
            # ``__init__``: its cached arrays include values calculated under
            # this simulation's policy, which a new branch would read as its
            # own.
            new_baseline = baseline.clone(
                debug=debug, trace=trace, clone_tax_benefit_system=False
            )
            new_baseline.parent_branch = new
            # A baseline traced in its simulation (as ``__init__`` makes it)
            # is traced in the copy.
            if baseline.tracer is self.tracer:
                new_baseline.tracer = new.tracer
            if self.branches.get(baseline.branch_name) is baseline:
                new.branches[baseline.branch_name] = new_baseline
            new.baseline = new_baseline

        return new

    def get_branch(
        self, name: str = "branch", clone_system: bool = False
    ) -> "Simulation":
        """Create a clone of this simulation, whose calculations are traced in the original.

        The branch starts from the values this simulation has cached when the
        branch is created. It does not copy them up front: each of the
        branch's holders gets its own index of this simulation's arrays, and
        copies an array the first time the branch reads it. A numpy array the
        branch never reads is never copied (masked arrays, and values that
        are not numpy arrays, are copied when the branch is created).

        What the branch stores (``set_input``, calculations, deletions) goes
        into its own index only, and what it reads is its own copy, so
        nothing done through the branch changes this simulation's values.
        What this simulation stores after branching stays out of the branch.

        The one difference from copying every array up front: code that
        writes in place into one of this simulation's cached arrays
        (``array[mask] = 0`` or ``array += 1``, instead of storing a new
        array with ``set_input``) after branching also changes the value the
        branch reads, if the branch has not read that array yet.

        As with the rest of a simulation, a branch is not safe to read from
        several threads at once: two first reads of the same array can each
        make a copy.

        A new branch starts with the values this simulation holds; asked for
        a name it already has, this returns that branch as it is. An input
        set on the branch drops those that may have been calculated from the
        value it replaces (see :meth:`set_input`). A branch whose
        tax-benefit system or parameters are changed should call
        :meth:`drop_computed_arrays` before calculating.

        Args:
            name (str, optional): Name of the branch. Defaults to "branch".
            clone_system (bool, optional): Whether to clone the tax-benefit system. Use this if you're changing policy parameters. Defaults to False.

        Returns:
            Simulation: The cloned simulation.
        """
        if name == self.branch_name:
            return self
        if name in self.branches:
            return self.branches[name]
        request = _BranchClone(self)
        token = _branch_clone.set(request)
        try:
            branch = self.clone(clone_tax_benefit_system=clone_system)
        finally:
            request.pending = False
            _branch_clone.reset(token)
        self.branches[name] = branch
        branch.branch_name = name
        branch.parent_branch = self
        branch._fixed_point_branch = True
        frames = _calculation_frames.get()
        if not frames:
            _note_unobserved_activity()
        for frame in frames:
            frame.created_branches[branch] = (self, name)
        if self.trace:
            branch.trace = True
            branch.tracer = self.tracer
        return branch

    def derivative(
        self, variable: str, wrt: str, period: Period = None, delta: float = 1
    ) -> ArrayLike:
        """
        Compute the derivative of a variable w.r.t another variable.

        Args:
            variable (str): The variable to differentiate.
            wrt (str): The variable to differentiate with respect to.
            period (Period): The period for which to compute the derivative.
            delta (float): The infinitesimal to use for the derivative.

        Returns:
            ArrayLike: The derivative.
        """

        if period is not None and not isinstance(period, Period):
            period = periods.period(period)
        elif period is None and self.default_calculation_period is not None:
            period = periods.period(self.default_calculation_period)

        alt_sim = self.clone()
        alt_sim.drop_computed_arrays()
        alt_sim.set_input(wrt, period, self.calculate(wrt, period) + delta)
        original_value = self.calculate(variable, period)
        new_value = alt_sim.calculate(variable, period)
        difference = new_value - original_value
        return difference / delta

    def sample_person(self) -> dict:
        """
        Sample a person from the simulation. Returns a situation JSON with their inputs (including their containing entities).

        Returns:
            dict: A dictionary containing the person's values.
        """
        person_count = self.persons.count
        index = np.random.randint(person_count)
        return self.extract_person(index)

    def extract_person(
        self,
        index: int = 0,
        exclude_entities: tuple = ("state",),
    ) -> dict:
        """
        Extract a person from the simulation. Returns a situation JSON with their inputs (including their containing entities).

        Args:
            index (int): The index of the person to extract.

        Returns:
            dict: A dictionary containing the person's values.
        """
        situation = {}
        people_indices = []
        people_indices_by_entity = {}

        for population in self.populations.values():
            entity = population.entity
            if not population.entity.is_person and entity.key not in exclude_entities:
                situation[entity.plural] = {
                    entity.key: {
                        "members": [],
                    },
                }
                group_index = population.members_entity_id[index]
                other_people_indices = [
                    index
                    for index in range(len(population.members_entity_id))
                    if population.members_entity_id[index] == group_index
                ]

                people_indices.extend(other_people_indices)
                people_indices = list(set(people_indices))
                people_indices_by_entity[entity.key] = other_people_indices
                for variable in self.input_variables:
                    if (
                        self.tax_benefit_system.get_variable(variable).entity.key
                        == entity.key
                    ):
                        known_periods = self.get_holder(variable).get_known_periods()
                        if len(known_periods) > 0:
                            first_known_period = known_periods[0]
                            value = self.calculate(variable, first_known_period)[
                                group_index
                            ]
                            situation[entity.plural][entity.key][variable] = {
                                str(known_periods[0]): value
                            }

        person = self.populations["person"].entity
        situation[person.plural] = {}
        for person_index in people_indices:
            person_name = f"{person.key}_{person_index + 1}"
            for entity_key in people_indices_by_entity:
                entity = self.populations[entity_key].entity
                if person_index in people_indices_by_entity[entity.key]:
                    situation[entity.plural][entity.key]["members"].append(person_name)
            situation[person.plural][person_name] = {}
            for variable in self.input_variables:
                if (
                    self.tax_benefit_system.get_variable(variable).entity.key
                    == person.key
                ):
                    known_periods = self.get_holder(variable).get_known_periods()
                    if len(known_periods) > 0:
                        first_known_period = known_periods[0]
                        value = self.calculate(variable, first_known_period)[
                            person_index
                        ]
                        situation[person.plural][person_name][variable] = {
                            str(known_periods[0]): value
                        }

        return json.loads(json.dumps(situation, cls=NpEncoder))

    def check_macro_cache(self, variable_name: str, period: str) -> bool:
        """
        Check if the variable is able to have cached value
        """
        if not self.macro_cache_read:
            return False

        # Dataset should always exist, but just in case
        if not hasattr(self, "dataset"):
            return False

        # If no dataset, no need to cache
        if self.dataset is None:
            return False

        # If using a flat file dataset, we're unable to cache
        if self.dataset.data_format == Dataset.FLAT_FILE:
            return False

        if not self.is_over_dataset:
            return False

        variable = self.tax_benefit_system.get_variable(variable_name)
        parameter_deps = variable.exhaustive_parameter_dependencies

        if parameter_deps is None:
            return False

        for parameter in parameter_deps:
            param = get_parameter(self.tax_benefit_system.parameters, parameter)
            if param.modified:
                return False

        return True

    def get_input_variables(self, include_computed_variables: bool = True) -> List[str]:
        """Return variable names stored as inputs on this simulation.

        Args:
            include_computed_variables: When ``True``, return the legacy
                runtime list of variables with stored values. When ``False``,
                return only structurally input variables that were populated
                through ``set_input`` on the current branch.

        Returns:
            List[str]: Stored input variable names.
        """
        if include_computed_variables:
            return list(self.input_variables)

        return [
            variable_name
            for variable_name in self.tax_benefit_system.variables
            if len(
                self._get_exportable_input_periods(
                    variable_name,
                    include_computed_variables=False,
                )
            )
            > 0
        ]

    @property
    def true_input_variables(self) -> List[str]:
        """Stored variables that are safe to reload as source inputs."""
        return self.get_input_variables(include_computed_variables=False)

    def _is_exportable_input_variable(self, variable_name: str) -> bool:
        variable = self.tax_benefit_system.get_variable(variable_name)
        return variable is not None and variable.is_input_variable()

    def _get_visible_branch_names(self) -> List[str]:
        branch_names = [getattr(self, "branch_name", "default")]
        parent = getattr(self, "parent_branch", None)
        while parent is not None:
            branch_names.append(parent.branch_name)
            parent = getattr(parent, "parent_branch", None)
        branch_names.append("default")
        return list(dict.fromkeys(branch_names))

    def _get_exportable_input_periods(
        self,
        variable_name: str,
        include_computed_variables: bool,
    ) -> List[Period]:
        if include_computed_variables:
            return self.get_holder(variable_name).get_known_periods()

        if not self._is_exportable_input_variable(variable_name):
            return []

        user_input_periods = {
            period
            for input_variable_name, branch_name, period in getattr(
                self, "_user_input_keys", set()
            )
            if input_variable_name == variable_name
            and branch_name in self._get_visible_branch_names()
        }
        if not user_input_periods:
            return []
        variable = self.tax_benefit_system.get_variable(variable_name)
        holder = self.get_holder(variable_name)
        if variable.definition_period == ETERNITY:
            return holder.get_known_periods()
        known_periods = set(holder.get_known_periods())
        return sorted(user_input_periods & known_periods, key=str)

    def to_input_dataframe(
        self,
        include_computed_variables: bool = False,
    ) -> pd.DataFrame:
        """Exports a DataFrame that can be loaded back into a new Simulation.

        By default, only structurally input variables populated through
        ``set_input`` are exported. This avoids serializing pseudo-inputs and
        stale calculated values that would override formulas when reloaded.

        Args:
            include_computed_variables: If ``True``, export every variable with
                a known period, matching the historical unsafe behavior.

        Returns:
            pd.DataFrame: The DataFrame containing the input values.
        """

        df = pd.DataFrame()

        for variable in self.tax_benefit_system.variables:
            variable_meta = self.tax_benefit_system.variables[variable]
            for period in self._get_exportable_input_periods(
                variable, include_computed_variables
            ):
                # Test if period matches entity definition period
                if variable_meta.definition_period != period.unit:
                    continue
                values = self.calculate(variable, period, map_to="person")
                if values is not None:
                    df[f"{variable}__{period}"] = values

        return df

    def to_input_dict(self, include_computed_variables: bool = False) -> dict:
        """Exports a dictionary that can be loaded back into a new Simulation.

        By default, only structurally input variables populated through
        ``set_input`` are exported. This avoids serializing pseudo-inputs and
        stale calculated values that would override formulas when reloaded.

        Args:
            include_computed_variables: If ``True``, export every variable with
                a known period, matching the historical unsafe behavior.

        Returns:
            dict: The dictionary containing the input values.
        """
        data = {}

        for variable in self.tax_benefit_system.variables:
            data[variable] = {}
            for period in self._get_exportable_input_periods(
                variable, include_computed_variables
            ):
                values = self.calculate(variable, period, map_to="person")
                if values is not None:
                    data[variable][str(period)] = values.tolist()

            if len(data[variable]) == 0:
                del data[variable]

        return data

    def subsample(
        self,
        n=None,
        frac=None,
        seed=None,
        time_period=None,
        quantize_weights: bool = True,
    ) -> "Simulation":
        """Quantize the simulation to a smaller size by sampling households.

        Args:
            n (int, optional): The number of households to sample. Defaults to 10_000.
            frac (float, optional): The fraction of households to sample. Defaults to None.
            seed (int, optional): The key used to seed the random number generator. Defaults to the dataset name.
            time_period (str, optional): Sample households based on their weight in this time period. Defaults to the default calculation period.

        Returns:
            Simulation: The quantized simulation.
        """
        default_calculation_period = self.default_calculation_period
        # Set default key if not provided
        if seed is None:
            seed = self.dataset.name

        # Set default time period if not provided
        if time_period is None:
            time_period = self.default_calculation_period

        # Subsampling rebuilds the complete dataset, so preserve computed
        # structural variables such as formula-backed IDs.
        df = self.to_input_dataframe(include_computed_variables=True)

        # Extract time period from DataFrame columns
        df_time_period = (
            df.columns[df.columns.str.contains("household_id__")]
            .values[0]
            .split("__")[1]
        )
        df_household_id_column = f"household_id__{df_time_period}"
        df_person_id_column = f"person_id__{df_time_period}"

        # Determine the appropriate household weight column
        if f"household_weight__{time_period}" in df.columns:
            household_weight_column = f"household_weight__{time_period}"
        else:
            household_weight_column = f"household_weight__{df_time_period}"

        # Group by household ID and get the first entry for each group
        h_df = df.groupby(df_household_id_column).first()
        h_ids = pd.Series(h_df.index)
        if n is None and frac is None:
            raise ValueError("Either n or frac must be provided.")
        if n is None:
            n = int(len(h_ids) * frac)
        h_weights = pd.Series(h_df[household_weight_column].values)

        frac = n / len(h_ids)

        # Seed the random number generators for reproducibility
        random.seed(str(seed))
        state = random.randint(0, 2**32 - 1)
        np.random.seed(state)

        h_ids = h_ids[h_weights > 0]
        h_weights = h_weights[h_weights > 0]

        # Sample household IDs based on their weights
        chosen_household_ids = pd.Series(
            np.random.choice(
                h_ids,
                n,
                p=(
                    h_weights.values / h_weights.values.sum()
                    if quantize_weights
                    else None
                ),
                replace=True,
            )
        )

        household_id_to_count = {}
        for household_id in chosen_household_ids:
            if household_id not in household_id_to_count:
                household_id_to_count[household_id] = 0
            household_id_to_count[household_id] += 1

        subset_df = df[df[df_household_id_column].isin(chosen_household_ids)].copy()

        household_counts = subset_df[df_household_id_column].map(
            lambda x: household_id_to_count.get(x, 0)
        )

        # Adjust household weights to maintain the total weight

        for col in subset_df.columns:
            if "weight__" in col:
                target_total_weight = df[col].values.sum()
                if not quantize_weights:
                    subset_df[col] *= household_counts.values
                else:
                    subset_df[col] = household_counts.values
                subset_df[col] *= target_total_weight / subset_df[col].values.sum()

        df = subset_df

        # Rebuilding replaces storage, its input record, and store history.
        self.dataset = Dataset.from_dataframe(df, self.dataset.time_period)
        self._store_history = StoreHistory()
        self._user_input_keys = set()
        self.build_from_dataset()

        # Purge ``_fast_cache`` entries populated by ``to_input_dataframe``
        # above. Those arrays were computed against the pre-subsample
        # populations, so their shapes (e.g. 41314 households) no longer
        # match the rebuilt populations (e.g. 669 after subsample).
        # ``build_from_dataset`` only rebuilds populations and holders;
        # the simulation-level fast cache is independent and must be
        # cleared explicitly or the stale entries bypass ``_calculate``
        # (see the short-circuit at the top of ``calculate``) and surface
        # as "size X != Y = count" projection errors.
        self._invalidate_all_caches()

        # Ensure the baseline branch has the new data: rebuild it from the
        # subsampled simulation through ``get_branch`` (the same wiring
        # ``__init__`` uses), then restore the saved baseline tax-benefit
        # system. Previously the saved system was assigned to
        # ``self.branches["tax_benefit_system"]`` — a stray dict key — so the
        # rebuilt baseline branch kept the reform system and reform-vs-baseline
        # comparisons after ``subsample`` compared the reform against itself.
        if "baseline" in self.branches:
            baseline_tax_benefit_system = self.branches["baseline"].tax_benefit_system
            del self.branches["baseline"]
            baseline = self.get_branch("baseline")
            # As in ``__init__``, the branch is traced in this simulation,
            # uses the baseline system's entities and variables, and has no
            # baseline of its own: ``get_branch`` gave it this simulation's,
            # the branch it replaces, which kept the old population alive.
            baseline.trace = self.trace
            baseline.tracer = self.tracer
            baseline.tax_benefit_system = baseline_tax_benefit_system
            baseline._bind_to_tax_benefit_system()
            baseline.baseline = None
            if getattr(self, "baseline", None) is not None:
                self.baseline = baseline

        self.default_calculation_period = default_calculation_period
        return self


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
