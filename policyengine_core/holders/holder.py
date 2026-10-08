import os
import warnings
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy
import psutil
from numpy.typing import ArrayLike

from policyengine_core import commons, periods, tools
from policyengine_core.data_storage import InMemoryStorage, OnDiskStorage
from policyengine_core.enums import Enum
from policyengine_core.errors import PeriodMismatchError
from policyengine_core.periods import Period

if TYPE_CHECKING:
    from policyengine_core.populations import Population
    from policyengine_core.variables import Variable


class Holder:
    """
    A holder keeps tracks of a variable values after they have been calculated, or set as an input.
    """

    def __init__(self, variable: "Variable", population: "Population"):
        self.population = population
        self.variable = variable
        self.simulation = population.simulation
        self._memory_storage = InMemoryStorage(
            is_eternal=(self.variable.definition_period == periods.ETERNITY)
        )

        # By default, do not activate on-disk storage, or variable dropping
        self._disk_storage = None
        self._on_disk_storable = False
        self._do_not_store = False
        if self.simulation and self.simulation.memory_config:
            if (
                self.variable.name
                not in self.simulation.memory_config.priority_variables
            ):
                self._disk_storage = self.create_disk_storage()
                self._on_disk_storable = True
            if self.variable.name in self.simulation.memory_config.variables_to_drop:
                self._do_not_store = True

    def clone(self, population: "Population", share_arrays: bool = False) -> "Holder":
        """
        Copy the holder just enough to be able to run a new simulation without modifying the original simulation.

        With ``share_arrays``, the new holder's in-memory storage shares
        this holder's arrays and copies each one when it is first read,
        instead of copying them all now (see :meth:`InMemoryStorage.clone`).
        """
        new = commons.empty_clone(self)
        new_dict = new.__dict__

        for key, value in self.__dict__.items():
            if key not in (
                "population",
                "formula",
                "simulation",
                "_memory_storage",
                "_disk_storage",
            ):
                new_dict[key] = value

        new._memory_storage = (
            self._memory_storage.clone(share_arrays=True)
            if share_arrays
            else self._memory_storage.clone()
        )
        new._disk_storage = (
            self._disk_storage.clone() if self._disk_storage is not None else None
        )

        new_dict["population"] = population
        new_dict["simulation"] = population.simulation

        return new

    def create_disk_storage(
        self, directory: str = None, preserve: bool = False
    ) -> OnDiskStorage:
        if directory is None:
            directory = self.simulation.data_storage_dir
        storage_dir = os.path.join(directory, self.variable.name)
        if not os.path.isdir(storage_dir):
            os.mkdir(storage_dir)
        # In the temporary folder the simulation made, the storage (and every
        # clone or copy of it) keeps the folder until it is collected, so the
        # folder outlives the simulation while a clone still reads it (see
        # ``OnDiskStorage``).
        return OnDiskStorage(
            storage_dir,
            is_eternal=(self.variable.definition_period == periods.ETERNITY),
            preserve_storage_dir=preserve,
        )

    def delete_arrays(
        self, period: Period = None, branch_name: str = "default"
    ) -> None:
        """
        If ``period`` is ``None``, remove all known values of the variable.

        If ``period`` is not ``None``, only remove all values for any period included in period (e.g. if period is "2017", values for "2017-01", "2017-07", etc. would be removed)
        """

        self._memory_storage.delete(period, branch_name)
        if self._disk_storage:
            self._disk_storage.delete(period, branch_name)
        self._evict_fast_cache(period, branch_name, contained=True)

    def _get_array_from_storage(
        self, period: Period, branch_name: str = "default"
    ) -> ArrayLike:
        value = self._memory_storage.get(period, branch_name)
        if value is None and self._disk_storage:
            value = self._disk_storage.get(period, branch_name)
        return value

    def get_array(self, period: Period, branch_name: str = "default") -> ArrayLike:
        """
        Get the value of the variable for the given period.

        If the value is not known, return ``None``.
        """
        if self.variable.is_neutralized:
            return self.default_array()
        value = self._get_array_from_storage(period, branch_name)
        if value is not None:
            return value
        if value is None and branch_name != "default":
            # Walk up ``simulation.parent_branch`` so nested branches inherit
            # values from their parent (e.g. a ``no_salt`` branch cloned
            # from an ``itemizing`` branch still sees ``tax_unit_itemizes``
            # set on the ``itemizing`` branch). Fall back to ``default``
            # only if no ancestor branch has a value. Previously the
            # fallback returned the first branch in dict-insertion order
            # (bug C1) — silently swapping values between unrelated
            # sibling branches (reform vs baseline) and producing wrong
            # reform deltas. The post-C1 behavior only fell back to
            # ``default``, which broke country-package nested-branch
            # patterns that relied on the ancestor's input being visible.
            parent = (
                getattr(self.simulation, "parent_branch", None)
                if self.simulation
                else None
            )
            while parent is not None:
                ancestor_value = self._get_array_from_storage(
                    period,
                    parent.branch_name,
                )
                if ancestor_value is not None:
                    return ancestor_value
                parent = getattr(parent, "parent_branch", None)
            default_value = self._get_array_from_storage(period, "default")
            if default_value is not None:
                return default_value

    def get_memory_usage(self) -> dict:
        """
        Get data about the virtual memory usage of the holder.

        An array a branch still shares with the simulation it was created
        from (see :meth:`Simulation.get_branch`) is counted for both, though
        it is held in memory once.

        :returns: Memory usage data
        :rtype: dict

        Example:

        >>> holder.get_memory_usage()
        >>> {
        >>>    'nb_arrays': 12,  # The holder contains the variable values for 12 different periods
        >>>    'nb_cells_by_array': 100, # There are 100 entities (e.g. persons) in our simulation
        >>>    'cell_size': 8,  # Each value takes 8B of memory
        >>>    'dtype': dtype('float64')  # Each value is a float 64
        >>>    'total_nb_bytes': 10400  # The holder uses 10.4kB of virtual memory
        >>>    'nb_requests': 24  # The variable has been computed 24 times
        >>>    'nb_requests_by_array': 2  # Each array stored has been on average requested twice
        >>>    }
        """

        usage = dict(
            nb_cells_by_array=self.population.count,
            dtype=self.variable.dtype,
        )

        usage.update(self._memory_storage.get_memory_usage())

        if self.simulation.trace:
            nb_requests = self.simulation.tracer.get_nb_requests(self.variable.name)
            usage.update(
                dict(
                    nb_requests=nb_requests,
                    nb_requests_by_array=(
                        nb_requests / float(usage["nb_arrays"])
                        if usage["nb_arrays"] > 0
                        else numpy.nan
                    ),
                )
            )

        return usage

    def get_known_periods(self) -> List[Period]:
        """
        Get the list of periods the variable value is known for.
        """

        return list(self._memory_storage.get_known_periods()) + list(
            (self._disk_storage.get_known_periods() if self._disk_storage else [])
        )

    def get_known_branch_periods(self) -> List[Tuple[str, Period]]:
        """
        Get the list of periods the variable value is known for.
        """

        return list(self._memory_storage.get_known_branch_periods()) + list(
            (
                self._disk_storage.get_known_branch_periods()
                if self._disk_storage
                else []
            )
        )

    def set_input(
        self, period: Period, array: ArrayLike, branch_name: str = "default"
    ) -> None:
        """
        Set a variable's value (``array``) for a given period (``period``)

        :param array: the input value for the variable
        :param period: the period at which the value is setted

        Example :

        >>> holder.set_input([12, 14], '2018-04')
        >>> holder.get_array('2018-04')
        >>> [12, 14]


        If a ``set_input`` property has been set for the variable, this method may accept inputs for periods not matching the ``definition_period`` of the variable. To read more about this, check the `documentation <https://openfisca.org/doc/coding-the-legislation/35_periods.html#set-input-automatically-process-variable-inputs-defined-for-periods-not-matching-the-definition-period>`_.
        """

        period = periods.period(period)
        if (
            period.unit == periods.ETERNITY
            and self.variable.definition_period != periods.ETERNITY
        ):
            error_message = os.linesep.join(
                [
                    "Unable to set a value for variable {0} for periods.ETERNITY.",
                    "{0} is only defined for {1}s. Please adapt your input.",
                ]
            ).format(self.variable.name, self.variable.definition_period)
            raise PeriodMismatchError(
                self.variable.name,
                period,
                self.variable.definition_period,
                error_message,
            )
        if self.variable.is_neutralized:
            warning_message = "You cannot set a value for the variable {}, as it has been neutralized. The value you provided ({}) will be ignored.".format(
                self.variable.name, array
            )
            return warnings.warn(warning_message, Warning)
        if self.variable.value_type in (float, int) and isinstance(array, str):
            array = tools.eval_expression(array)
        self._raise_if_input_contains_nan(numpy.asarray(array))
        simulation = getattr(self, "simulation", None)
        if simulation is not None:
            if not hasattr(simulation, "_user_input_keys"):
                simulation._user_input_keys = set()
            if not hasattr(simulation, "_user_input_contexts"):
                simulation._user_input_contexts = []
            simulation._user_input_contexts.append(branch_name)
        try:
            if (
                self.variable.set_input
                and period.unit != self.variable.definition_period
            ):
                return self.variable.set_input(self, period, array)
            return self._set(period, array, branch_name, validate_nan=True)
        finally:
            if simulation is not None:
                simulation._user_input_contexts.pop()

    def _raise_if_input_contains_nan(self, value: ArrayLike) -> None:
        if self.variable.value_type not in (float, int):
            return
        value = numpy.asarray(value)
        try:
            if value.dtype.kind in ("O", "S", "U"):
                value = value.astype(float)
            contains_nan = numpy.isnan(value).any()
        except (TypeError, ValueError):
            return
        if contains_nan:
            raise ValueError(
                'Unable to set value for variable "{}", as the input contains NaN values.'.format(
                    self.variable.name,
                )
            )

    def _to_array(self, value: Any, validate_nan: bool = False) -> ArrayLike:
        if not isinstance(value, numpy.ndarray):
            value = numpy.asarray(value)
        if value.ndim == 0:
            # 0-dim arrays are casted to scalar when they interact with float. We don't want that.
            value = value.reshape(1)
        if len(value) != self.population.count:
            raise ValueError(
                'Unable to set value "{}" for variable "{}", as its length is {} while there are {} {} in the simulation.'.format(
                    value,
                    self.variable.name,
                    len(value),
                    self.population.count,
                    self.population.entity.plural,
                )
            )
        if validate_nan:
            self._raise_if_input_contains_nan(value)
        if self.variable.value_type == Enum:
            original_value = value
            value = self.variable.possible_values.encode(value)
            if value.shape != original_value.shape:
                value = self.variable.possible_values.encode(original_value.astype("O"))
        if value.dtype != self.variable.dtype:
            try:
                value = value.astype(self.variable.dtype)
            except ValueError:
                raise ValueError(
                    'Unable to set value "{}" for variable "{}", as the variable dtype "{}" does not match the value dtype "{}".'.format(
                        value,
                        self.variable.name,
                        self.variable.dtype,
                        value.dtype,
                    )
                )
        if validate_nan:
            self._raise_if_input_contains_nan(value)
        return value

    def _set(
        self,
        period: Period,
        value: ArrayLike,
        branch_name: str = "default",
        validate_nan: bool = False,
        derived: bool = False,
    ) -> None:
        simulation = getattr(self, "simulation", None)
        # A value calculated while an input is being set (say, by a
        # ``set_input`` helper that calculates) is not part of that input: it
        # belongs to the branch it was calculated on.
        user_input_contexts = (
            None if derived else getattr(simulation, "_user_input_contexts", None)
        )
        if user_input_contexts and branch_name == "default":
            branch_name = user_input_contexts[-1]
        value = self._to_array(value, validate_nan=validate_nan)
        if self.variable.definition_period != periods.ETERNITY:
            if period is None:
                raise ValueError(
                    "A period must be specified to set values, except for variables with periods.ETERNITY as as period_definition."
                )

        should_store_on_disk = (
            self._on_disk_storable
            and self._memory_storage.get(period, branch_name) is None
            and psutil.virtual_memory().percent  # If there is already a value in memory, replace it and don't put a new value in the disk storage
            >= self.simulation.memory_config.max_memory_occupation_pc
        )

        if should_store_on_disk:
            self._disk_storage.put(value, period, branch_name, derived=derived)
        else:
            self._memory_storage.put(value, period, branch_name, derived=derived)
        self._evict_fast_cache(period, branch_name)
        if user_input_contexts:
            if not hasattr(simulation, "_user_input_keys"):
                simulation._user_input_keys = set()
            simulation._user_input_keys.add((self.variable.name, branch_name, period))

    def put_in_cache(
        self,
        value: ArrayLike,
        period: Period,
        branch_name: str = "default",
        derived: bool = False,
    ) -> None:
        """Cache ``value`` for ``period``.

        ``derived`` marks a value the simulation calculated rather than took
        as input: a formula result, a carried, uprated or default value, a
        twelfth of a yearly flow cached at a month by ``calculate_divide``,
        or a sum over several sub-periods cached by ``calculate_add``.
        Auto-carry-over never carries such a value into another period (see
        ``is_derived``). The mark is stored with the value, for the (branch,
        period) key written, and any later write to that key replaces it.

        A derived value never replaces an input that ``get_array(period,
        branch_name)`` reads: the input is kept and nothing is stored.
        """
        if self._do_not_store:
            return

        if (
            self.simulation.opt_out_cache
            and self.simulation.tax_benefit_system.cache_blacklist
            and self.variable.name in self.simulation.tax_benefit_system.cache_blacklist
        ):
            return

        if (
            derived
            and self._branch_storing(period, branch_name) is not None
            and not self.is_derived(period, branch_name)
        ):
            return

        self._set(period, value, branch_name, derived=derived)

    def default_array(self) -> ArrayLike:
        """
        Return a new array of the appropriate length for the entity, filled with the variable default values.
        """

        return self.variable.default_array(self.population.count)

    def _stores(self, period: Period, branch_name: str) -> bool:
        """Whether a value is stored for ``period`` under ``branch_name``,
        without reading or copying it."""
        return self._memory_storage.has(period, branch_name) or (
            self._disk_storage is not None
            and self._disk_storage.has(period, branch_name)
        )

    def _readable_branches(self, branch_name: str = "default") -> List[str]:
        """``get_array``'s lookup order: the branch, its ``parent_branch``
        ancestors, then ``default``."""
        names = [branch_name]
        if branch_name != "default":
            parent = (
                getattr(self.simulation, "parent_branch", None)
                if self.simulation
                else None
            )
            while parent is not None:
                names.append(parent.branch_name)
                parent = getattr(parent, "parent_branch", None)
            names.append("default")
        return list(dict.fromkeys(names))

    def _branch_storing(self, period: Period, branch_name: str = "default") -> str:
        """The branch whose stored value ``get_array(period, branch_name)``
        reads, or ``None`` if none stores one."""
        for name in self._readable_branches(branch_name):
            if self._stores(period, name):
                return name
        return None

    def is_derived(self, period: Period, branch_name: str = "default") -> bool:
        """Whether the value ``get_array(period, branch_name)`` reads was
        calculated by the simulation rather than set as an input.

        The answer comes from the branch that stores the value read: the
        branch itself, else its ``parent_branch`` ancestors, else
        ``default``. ``False`` if no value is stored for ``period``.
        """
        storing = self._branch_storing(period, branch_name)
        if storing is None:
            return False
        if self._memory_storage.has(period, storing):
            return self._memory_storage.is_derived(period, storing)
        return self._disk_storage.is_derived(period, storing)

    def get_input_periods(self, branch_name: str = "default") -> List[Period]:
        """The periods for which the value ``get_array(period, branch_name)``
        reads is an input rather than a value the simulation calculated (see
        ``put_in_cache``). Periods stored only under branches this one cannot
        read are left out.

        One pass over the stored keys: for each period, the key ``get_array``
        reads first (the branch before its ancestors, memory before disk).
        """
        rank = {
            name: index
            for index, name in enumerate(self._readable_branches(branch_name))
        }
        storages = [self._memory_storage]
        if self._disk_storage is not None:
            storages.append(self._disk_storage)
        read = {}
        for order, storage in enumerate(storages):
            for stored_branch, period in storage.get_known_branch_periods():
                if stored_branch not in rank:
                    continue
                key = (rank[stored_branch], order)
                if period not in read or key < read[period][0]:
                    read[period] = (key, storage, stored_branch)
        return [
            period
            for period, (_, storage, stored_branch) in read.items()
            if not storage.is_derived(period, stored_branch)
        ]

    def _evict_fast_cache(
        self, period: Period, branch_name: str, contained: bool = False
    ) -> None:
        """Drop the simulation's ``_fast_cache`` entries a storage write or delete makes stale.

        ``Simulation.calculate`` answers a repeated request from
        ``_fast_cache``, keyed by ``(variable name, requested period)``,
        before it reads this holder. So every write into, or delete from,
        this holder's storage drops the entries for the periods it changes;
        otherwise ``calculate`` kept returning the value the storage no
        longer held (for example after ``holder.set_input``).

        The fast cache belongs to this holder's simulation: a branch has its
        own holders and its own fast cache, and keeps the values it started
        with, so nothing outside this simulation is touched. Nor is anything
        here when ``branch_name`` is a branch this simulation does not read.

        A write changes one storage key: ``period``, or, for an ETERNITY
        variable, the one value every period reads. A delete (``contained``)
        removes every period ``period`` contains, or every period when
        ``period`` is ``None``.
        """
        simulation = self.simulation
        fast_cache = getattr(simulation, "_fast_cache", None)
        if not fast_cache:
            return
        name = self.variable.name
        drop_all = period is None or self.variable.definition_period == periods.ETERNITY
        if not drop_all:
            period = periods.period(period)
            if not contained and (name, period) not in fast_cache:
                return
        visible_branch_names = getattr(simulation, "_get_visible_branch_names", None)
        if visible_branch_names is not None and branch_name not in (
            visible_branch_names()
        ):
            return
        if not drop_all and not contained:
            del fast_cache[(name, period)]
            return
        stale_keys = [
            key
            for key in fast_cache
            if key[0] == name
            and (drop_all or not isinstance(key[1], Period) or period.contains(key[1]))
        ]
        for key in stale_keys:
            del fast_cache[key]
