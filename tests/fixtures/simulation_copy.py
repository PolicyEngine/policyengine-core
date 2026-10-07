"""Helpers for the simulation copy and pickle tests."""

from __future__ import annotations

import copy
import pickle

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.enums import Enum, EnumArray
from policyengine_core.parameters import ParameterNode
from policyengine_core.reforms import Reform
from policyengine_core.simulations import SimulationBuilder

JAN = "2025-01"
FEB = "2025-02"

COPIERS = {
    "deepcopy": copy.deepcopy,
    "pickle": lambda value: pickle.loads(pickle.dumps(value)),
}


class CopyEnum(Enum):
    first = "First"
    second = "Second"


class ChildEnumArray(EnumArray):
    """Importable subclass for copy and pickle type-preservation checks."""


def build_simulation(
    tax_benefit_system=None, salaries=(1000,), households=None, occupancy=None
):
    """One household per entry of ``households`` (lists of person indices).

    ``occupancy`` gives each household's ``housing_occupancy_status``.
    """
    tax_benefit_system = tax_benefit_system or CountryTaxBenefitSystem()
    names = [f"p{i}" for i in range(len(salaries))]
    households = households or [list(range(len(salaries)))]
    occupancy = occupancy or [None] * len(households)
    return SimulationBuilder().build_from_entities(
        tax_benefit_system,
        {
            "persons": {
                name: {"salary": {JAN: salary}} for name, salary in zip(names, salaries)
            },
            "households": {
                f"h{i}": {
                    "parents": [names[j] for j in members[:2]],
                    "children": [names[j] for j in members[2:]],
                    "accommodation_size": {JAN: 100},
                    **(
                        {"housing_occupancy_status": {JAN: occupancy[i]}}
                        if occupancy[i]
                        else {}
                    ),
                }
                for i, members in enumerate(households)
            },
        },
    )


def rate_node() -> ParameterNode:
    return ParameterNode(
        "rate",
        data={
            "single": {
                "owner": {"values": {"2015-01-01": 100}},
                "tenant": {"values": {"2015-01-01": 300}},
            },
            "couple": {
                "owner": {"values": {"2015-01-01": 500}},
                "tenant": {"values": {"2015-01-01": 700}},
            },
        },
    )


class DoubleIncomeTaxRate(Reform):
    """Module level, so pickle can find the reform class by name."""

    def apply(self):
        def double(parameters):
            rate = parameters.taxes.income_tax_rate
            rate.update(period="year:2015:20", value=rate("2015-01-01") * 2)
            return parameters

        self.modify_parameters(double)
