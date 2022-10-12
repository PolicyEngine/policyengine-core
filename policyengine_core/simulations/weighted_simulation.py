from typing import Dict
from policyengine_core.periods import Period, period as get_period
from policyengine_core.periods.config import MONTH, YEAR
from policyengine_core.simulations.simulation import Simulation
from microdf import MicroSeries, MicroDataFrame
from policyengine_core.types import ArrayLike


class WeightedSimulation(Simulation):
    """A `Simulation` whose entities use weights to represent larger populations."""

    def get_weights(
        self, variable_name: str, period: Period, map_to: str = None
    ) -> ArrayLike:
        time_period = get_period(period)
        variable = self.tax_benefit_system.get_variable(variable_name)
        entity_key = map_to or variable.entity.key
        weight_variable_name = f"{entity_key}_weight"
        weight_variable = self.tax_benefit_system.get_variable(
            weight_variable_name
        )
        weights = None

        if time_period.unit == weight_variable.definition_period:
            weights = super().calculate(weight_variable_name, time_period)
        elif (time_period.unit == MONTH) and (
            weight_variable.definition_period == YEAR
        ):
            # Common use-case. To-do: implement others if needed.
            weights = super().calculate(
                weight_variable_name, time_period.this_year
            )

        return weights

    def calculate(
        self, variable_name: str, period: Period, map_to: str = None
    ) -> MicroSeries:
        values = super().calculate(variable_name, period)
        weights = self.get_weights(variable_name, period, map_to)
        return MicroSeries(values, weights=weights)

    def calculate_add(self, variable_name: str, period: Period) -> MicroSeries:
        values = super().calculate_add(variable_name, period)
        weights = self.get_weights(variable_name, period)
        return MicroSeries(values, weights=weights)

    def calculate_divide(
        self, variable_name: str, period: Period
    ) -> MicroSeries:
        values = super().calculate_divide(variable_name, period)
        weights = self.get_weights(variable_name, period)
        return MicroSeries(values, weights=weights)

    def calculate_dataframe(
        self, variable_names: list, period: Period
    ) -> MicroDataFrame:
        values = super().calculate_dataframe(variable_names, period)
        weights = self.get_weights(variable_names[0], period)
        return MicroDataFrame(values, weights=weights)
