import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Type, Union

import numpy
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from policyengine_core import commons, periods
from policyengine_core.data.dataset import Dataset
from policyengine_core.entities.entity import Entity
from policyengine_core.enums import Enum, EnumArray
from policyengine_core.errors import CycleError, SpiralError
from policyengine_core.holders.holder import Holder
from policyengine_core.periods import Period
from policyengine_core.periods.config import ETERNITY
from policyengine_core.periods.helpers import period
from policyengine_core.tracers import (
    FullTracer,
    SimpleTracer,
    TracingParameterNodeAtInstant,
)

if TYPE_CHECKING:
    from policyengine_core.taxbenefitsystems import TaxBenefitSystem

from policyengine_core.experimental import MemoryConfig
from policyengine_core.populations import Population
from policyengine_core.tracers import SimpleTracer
from policyengine_core.variables import Variable
from policyengine_core.reforms.reform import Reform
from policyengine_core.parameters import get_parameter


class Simulation:
    """
    Represents a simulation, and handles the calculation logic
    """

    default_tax_benefit_system: Type["TaxBenefitSystem"] = None
    """The default tax-benefit system class to use if none is provided."""

    default_tax_benefit_system_instance: "TaxBenefitSystem" = None
    """The default tax-benefit system instance to use if none is provided. This requires that the tax-benefit system is initialised when importing a country package. This will slow down the import, but may speed up individual simulations."""

    default_dataset: Dataset = None
    """The default dataset class to use if none is provided."""

    default_dataset_year: int = None
    """The default dataset year to use if none is provided."""

    default_role: str = None
    """The default role to assign people to groups if none is provided."""

    default_input_period: str = None
    """The default period to use when inputting variables."""

    default_calculation_period: str = None
    """The default period to calculate for if none is provided."""

    def __init__(
        self,
        tax_benefit_system: "TaxBenefitSystem" = None,
        populations: Dict[str, Population] = None,
        situation: dict = None,
        dataset: Type[Dataset] = None,
        dataset_year: int = None,
        reform: Reform = None,
    ):
        if tax_benefit_system is None:
            if self.default_tax_benefit_system_instance is not None:
                tax_benefit_system = self.default_tax_benefit_system_instance
                if reform is not None:
                    tax_benefit_system = tax_benefit_system.clone()
            else:
                tax_benefit_system = self.default_tax_benefit_system()
            self.tax_benefit_system = tax_benefit_system

        self.reform = reform
        self.tax_benefit_system = tax_benefit_system

        if reform is not None:
            self.apply_reform(reform)

        self.branch_name = "default"

        if dataset is None:
            if self.default_dataset is not None:
                dataset = self.default_dataset

        if dataset_year is None:
            if self.default_dataset_year is not None:
                dataset_year = self.default_dataset_year

        self.invalidated_caches = set()
        self.debug: bool = False
        self.trace: bool = False
        self.tracer: SimpleTracer = SimpleTracer()
        self.opt_out_cache: bool = False
        # controls the spirals detection; check for performance impact if > 1
        self.max_spiral_loops: int = 1
        self.memory_config: MemoryConfig = None
        self._data_storage_dir: str = None

        self.branches: Dict[str, Simulation] = {}
        self.has_axes = False

        if situation is not None:
            if dataset is not None:
                raise ValueError(
                    "You provided both a situation and a dataset. Only one input method is allowed."
                )
            self.build_from_populations(
                self.tax_benefit_system.instantiate_entities()
            )
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
            self.dataset = dataset
            self.dataset_year = dataset_year
            self.build_from_dataset()

        # Backwards compatibility methods
        self.calc = self.calculate
        self.df = self.calculate_dataframe

        self.input_variables = [
            variable.name
            for variable in self.tax_benefit_system.variables.values()
            if len(self.get_holder(variable.name).get_known_periods()) > 0
        ]

    def apply_reform(self, reform: Union[tuple, Reform]):
        if isinstance(reform, tuple):
            for subreform in reform:
                self.apply_reform(subreform)
        else:
            reform.apply(self.tax_benefit_system)

    def build_from_populations(
        self, populations: Dict[str, Population]
    ) -> None:
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
        self.build_from_populations(
            self.tax_benefit_system.instantiate_entities()
        )
        from policyengine_core.simulations.simulation_builder import (
            SimulationBuilder,
        )  # Import here to avoid circular dependency

        builder = SimulationBuilder()
        builder.populations = self.populations
        try:
            data = self.dataset.load(self.dataset_year)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"The dataset file {self.dataset.name} (with year {self.dataset_year}) could not be found. "
                + "Make sure you have downloaded or built it using the `policyengine-core data` command."
            ) from e

        person_entity = self.tax_benefit_system.person_entity
        entity_id_field = f"{person_entity.key}_id"
        assert (
            entity_id_field in data
        ), f"Missing {entity_id_field} column in the dataset. Each person entity must have an ID array defined for ETERNITY."

        get_eternity_array = (
            lambda ds: ds[list(ds.keys())[0]]
            if self.dataset.data_format == Dataset.TIME_PERIOD_ARRAYS
            else ds
        )
        entity_ids = get_eternity_array(data[entity_id_field])
        builder.declare_person_entity(person_entity.key, entity_ids)

        for group_entity in self.tax_benefit_system.group_entities:
            entity_id_field = f"{group_entity.key}_id"
            assert (
                entity_id_field in data
            ), f"Missing {entity_id_field} column in the dataset. Each group entity must have an ID array defined for ETERNITY."

            entity_ids = get_eternity_array(data[entity_id_field])
            builder.declare_entity(group_entity.key, entity_ids)

            person_membership_id_field = (
                f"{person_entity.key}_{group_entity.key}_id"
            )
            assert (
                person_membership_id_field in data
            ), f"Missing {person_membership_id_field} column in the dataset. Each group entity must have a person membership array defined for ETERNITY."
            person_membership_ids = get_eternity_array(
                data[person_membership_id_field]
            )

            person_role_field = f"{person_entity.key}_{group_entity.key}_role"
            if person_role_field in data:
                person_roles = get_eternity_array(data[person_role_field])
            elif "role" in data:
                person_roles = get_eternity_array(data["role"])
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

        for variable in data:
            if variable in self.tax_benefit_system.variables:
                if self.dataset.data_format == Dataset.TIME_PERIOD_ARRAYS:
                    for time_period in data[variable]:
                        self.set_input(
                            variable, time_period, data[variable][time_period]
                        )
                else:
                    self.set_input(variable, self.dataset_year, data[variable])
            else:
                # Silently skip.
                pass

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

    def create_shortcuts(self) -> None:
        for _key, population in self.populations.items():
            # create shortcut simulation.person and simulation.household (for instance)
            setattr(self, population.entity.key, population)

    @property
    def data_storage_dir(self) -> str:
        """
        Temporary folder used to store intermediate calculation data in case the memory is saturated
        """
        if self._data_storage_dir is None:
            self._data_storage_dir = tempfile.mkdtemp(prefix="openfisca_")
            message = [
                (
                    "Intermediate results will be stored on disk in {} in case of memory overflow."
                ).format(self._data_storage_dir),
                "You should remove this directory once you're done with your simulation.",
            ]
        return self._data_storage_dir

    # ----- Calculation methods ----- #

    def calculate(
        self,
        variable_name: str,
        period: Period = None,
        map_to: str = None,
    ) -> ArrayLike:
        """Calculate ``variable_name`` for ``period``.

        Args:
            variable_name (str): The name of the variable to calculate.
            period (Period): The period to calculate the variable for.
            map_to (str): The name of the variable to map the result to. If None, the result is returned as is.

        Returns:
            ArrayLike: The calculated variable.
        """

        if period is not None and not isinstance(period, Period):
            period = periods.period(period)
        elif period is None and self.default_calculation_period is not None:
            period = periods.period(self.default_calculation_period)

        self.tracer.record_calculation_start(
            variable_name, period, self.branch_name
        )

        try:
            result = self._calculate(variable_name, period)
            self.tracer.record_calculation_result(result)
            if map_to is not None:
                source_entity = self.tax_benefit_system.get_variable(
                    variable_name
                ).entity.key
                result = self.map_result(result, source_entity, map_to)
            return result
        finally:
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

        df = pd.DataFrame()
        entities = [
            self.tax_benefit_system.get_variable(variable_name).entity.key
            for variable_name in variable_names
        ]
        # Check that all variables are from the same entity. If not, map values to the entity of the first variable.
        entity = entities[0]
        if not all(entity == e for e in entities):
            map_to = entity
        for variable_name in variable_names:
            df[variable_name] = self.calculate(variable_name, period, map_to)
        return df

    def _calculate(
        self, variable_name: str, period: Period = None
    ) -> ArrayLike:
        """
        Calculate the variable ``variable_name`` for the period ``period``, using the variable formula if it exists.

        Args:
            variable_name (str): The name of the variable to calculate.
            period (Period): The period to calculate the variable for.

        Returns:
            ArrayLike: The calculated variable.
        """
        population = self.get_variable_population(variable_name)
        holder = population.get_holder(variable_name)
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )

        self._check_period_consistency(period, variable)

        # First look for a value already cached
        cached_array = holder.get_array(period, self.branch_name)
        if cached_array is not None:
            return cached_array

        if variable.defined_for is not None:
            mask = (
                self.calculate(
                    variable.defined_for, period, map_to=variable.entity.key
                )
                > 0
            )
            if np.all(~mask):
                array = holder.default_array()
                array = self._cast_formula_result(array, variable)
                holder.put_in_cache(array, period, self.branch_name)
                return array

        array = None

        # First, try to run a formula
        try:
            self._check_for_cycle(variable.name, period)
            array = self._run_formula(variable, population, period)

            # If no result, use the default value and cache it
            if array is None:
                # Check if the variable has a previously defined value
                known_periods = holder.get_known_periods()
                if variable.uprating is not None and len(known_periods) > 0:
                    start_instants = [
                        known_period.start for known_period in known_periods
                    ]
                    latest_known_period = known_periods[
                        np.argmax(start_instants)
                    ]
                    if latest_known_period.start < period.start:
                        try:
                            uprating_parameter = get_parameter(
                                self.tax_benefit_system.parameters,
                                variable.uprating,
                            )
                        except:
                            raise ValueError(
                                f"Could not find uprating parameter {variable.uprating} when trying to uprate {variable_name}."
                            )
                        value_in_last_period = uprating_parameter(
                            latest_known_period.start
                        )
                        value_in_this_period = uprating_parameter(period.start)
                        uprating_factor = (
                            value_in_this_period / value_in_last_period
                        )
                        return (
                            holder.get_array(
                                latest_known_period, self.branch_name
                            )
                            * uprating_factor
                        )
                elif (
                    self.tax_benefit_system.auto_carry_over_input_variables
                    and variable.calculate_output is None
                    and len(known_periods) > 0
                ):
                    # Variables with a calculate-output property specify
                    last_known_period = sorted(known_periods)[-1]
                    array = holder.get_array(last_known_period)
                else:
                    array = holder.default_array()

            if variable.defined_for is not None:
                array = np.where(mask, array, np.zeros_like(array))

            array = self._cast_formula_result(array, variable)
            holder.put_in_cache(array, period, self.branch_name)

        except SpiralError:
            array = holder.default_array()
        except RecursionError:
            if self.trace:
                self.tracer.print_computation_log()
            else:
                print(
                    "Computation log is only available with the trace=True option."
                )
            raise Exception(
                f"RecursionError while calculating {variable_name} for period {period}. The full computation trace is printed above."
            )

        return array

    def purge_cache_of_invalid_values(self) -> None:
        # We wait for the end of calculate(), signalled by an empty stack, before purging the cache
        if self.tracer.stack:
            return
        for (_name, _period) in self.invalidated_caches:
            holder = self.get_holder(_name)
            holder.delete_arrays(_period)
        self.invalidated_caches = set()

    def calculate_add(
        self, variable_name: str, period: Period = None
    ) -> ArrayLike:
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )

        if period is not None and not isinstance(period, Period):
            period = periods.period(period)

        # Check that the requested period matches definition_period
        if periods.unit_weight(
            variable.definition_period
        ) > periods.unit_weight(period.unit):
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

        return sum(
            self.calculate(variable_name, sub_period)
            for sub_period in period.get_subperiods(variable.definition_period)
        )

    def calculate_divide(
        self, variable_name: str, period: Period = None
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
            computation_period = period.this_year
            return (
                self.calculate(variable_name, period=computation_period) / 12.0
            )
        elif period.unit == periods.YEAR:
            return self.calculate(variable_name, period)

        raise ValueError(
            "Unable to divide the value of '{}' to match period {}.".format(
                variable_name, period
            )
        )

    def calculate_output(
        self, variable_name: str, period: Period = None
    ) -> ArrayLike:
        """
        Calculate the value of a variable using the ``calculate_output`` attribute of the variable.
        """

        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )

        if variable.calculate_output is None:
            return self.calculate(variable_name, period)

        return variable.calculate_output(self, variable_name, period)

    def trace_parameters_at_instant(self, formula_period):
        return TracingParameterNodeAtInstant(
            self.tax_benefit_system.get_parameters_at_instant(formula_period),
            self.tracer,
            self.branch_name,
        )

    def _run_formula(
        self, variable: str, population: Population, period: Period
    ) -> ArrayLike:
        """
        Find the ``variable`` formula for the given ``period`` if it exists, and apply it to ``population``.
        """

        formula = variable.get_formula(period)
        if formula is None:
            values = None
            if variable.adds is not None:
                values = 0
                for added_variable in variable.adds:
                    values = values + self.calculate(
                        added_variable, period, map_to=variable.entity.key
                    )
            if variable.subtracts is not None:
                if values is None:
                    values = 0
                for subtracted_variable in variable.subtracts:
                    values = values - self.calculate(
                        subtracted_variable, period, map_to=variable.entity.key
                    )
            return values

        if self.trace:
            parameters_at = self.trace_parameters_at_instant
        else:
            parameters_at = self.tax_benefit_system.get_parameters_at_instant

        if formula.__code__.co_argcount == 2:
            array = formula(population, period)
        else:
            array = formula(population, period, parameters_at)

        return array

    def _check_period_consistency(
        self, period: Period, variable: Variable
    ) -> None:
        """
        Check that a period matches the variable definition_period
        """
        if variable.definition_period == periods.ETERNITY:
            return  # For variables which values are constant in time, all periods are accepted

        if (
            variable.definition_period == periods.MONTH
            and period.unit != periods.MONTH
        ):
            raise ValueError(
                "Unable to compute variable '{0}' for period {1}: '{0}' must be computed for a whole month. You can use the ADD option to sum '{0}' over the requested period, or change the requested period to 'period.first_month'.".format(
                    variable.name, period
                )
            )

        if (
            variable.definition_period == periods.YEAR
            and period.unit != periods.YEAR
        ):
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
                    "month"
                    if variable.definition_period == periods.MONTH
                    else "year",
                )
            )

    def _cast_formula_result(self, value: Any, variable: str) -> ArrayLike:
        if variable.value_type == Enum and not isinstance(value, EnumArray):
            return variable.possible_values.encode(value)

        if not isinstance(value, numpy.ndarray):
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
            if frame["name"] == variable
            and frame["branch_name"] == self.branch_name
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
        self.invalidated_caches.add((variable, period))

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
        return self.get_holder(variable_name).get_array(
            period, self.branch_name
        )

    def get_holder(self, variable_name: str) -> Holder:
        """
        Get the :obj:`.Holder` associated with the variable ``variable_name`` for the simulation
        """
        return self.get_variable_population(variable_name).get_holder(
            variable_name
        )

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
        Delete a variable's value for a given period

        :param variable: the variable to be set
        :param period: the period for which the value should be deleted

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
        self.get_holder(variable).delete_arrays(period)

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

    def set_input(
        self, variable_name: str, period: Period, value: ArrayLike
    ) -> None:
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
        """
        variable = self.tax_benefit_system.get_variable(
            variable_name, check_existence=True
        )
        period = periods.period(period)
        if (variable.end is not None) and (period.start.date > variable.end):
            return
        self.get_holder(variable_name).set_input(
            period, value, self.branch_name
        )

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

    def clone(self, debug: bool = False, trace: bool = False) -> "Simulation":
        """
        Copy the simulation just enough to be able to run the copy without modifying the original simulation
        """
        new = commons.empty_clone(self)
        new_dict = new.__dict__

        for key, value in self.__dict__.items():
            if key not in ("debug", "trace", "tracer", "branches"):
                new_dict[key] = value

        new.persons = self.persons.clone(new)
        setattr(new, new.persons.entity.key, new.persons)
        new.populations = {new.persons.entity.key: new.persons}
        new.branches = {}

        for entity in self.tax_benefit_system.group_entities:
            population = self.populations[entity.key].clone(new, new.persons)
            new.populations[entity.key] = population
            setattr(
                new, entity.key, population
            )  # create shortcut simulation.household (for instance)

        new.tax_benefit_system = self.tax_benefit_system.clone()
        new.debug = debug
        new.trace = trace

        return new

    def get_branch(self, name: str = "branch") -> "Simulation":
        """Create a clone of this simulation, whose calculations are traced in the original.

        Args:
            name (str, optional): Name of the branch. Defaults to "branch".
            store (bool, optional): Whether to store the branch in the original simulation. Defaults to True.

        Returns:
            Simulation: The cloned simulation.
        """
        if name == self.branch_name:
            return self
        branch = self.clone()
        self.branches[name] = branch
        branch.branch_name = name
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
        for computed_variable in alt_sim.tax_benefit_system.variables:
            if computed_variable not in self.input_variables:
                alt_sim.delete_arrays(computed_variable)
        alt_sim.set_input(wrt, period, self.calculate(wrt, period) + delta)
        original_value = self.calculate(variable, period)
        new_value = alt_sim.calculate(variable, period)
        difference = new_value - original_value
        return difference / delta
