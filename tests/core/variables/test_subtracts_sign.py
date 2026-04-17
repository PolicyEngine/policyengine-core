"""Regression test for ``subtracts`` with a parameter element (bug C5).

The bug: when a ``subtracts`` list element refers to a parameter (not another
variable), the simulation previously did ``values = values + parameter(...)``
instead of subtracting it. That flipped the sign for any such entry, making
the formula value off by 2x the parameter value.
"""

from __future__ import annotations

from policyengine_core.entities import Entity
from policyengine_core.model_api import ETERNITY, Variable
from policyengine_core.parameters import Parameter, ParameterNode
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.taxbenefitsystems import TaxBenefitSystem


def _build_system_with_parameter_subtracts():
    Person = Entity("person", "people", "Person", "A person")
    system = TaxBenefitSystem([Person])

    class base_income(Variable):
        value_type = float
        entity = Person
        definition_period = ETERNITY
        label = "Base income before subtraction"
        default_value = 1000.0

    class net_income(Variable):
        value_type = float
        entity = Person
        definition_period = ETERNITY
        label = "Net income"
        adds = ["base_income"]
        # ``fee`` is a parameter, NOT a variable; the simulation must subtract it.
        subtracts = ["fee"]

    system.add_variables(base_income, net_income)

    # Attach a Parameter called ``fee`` worth 300 to the parameter tree.
    if system.parameters is None:
        system.parameters = ParameterNode("", data={})
    fee = Parameter("fee", data={"2000-01-01": 300.0})
    system.parameters.add_child("fee", fee)

    return system


def test_subtracts_with_parameter_actually_subtracts():
    system = _build_system_with_parameter_subtracts()
    simulation = SimulationBuilder().build_from_dict(
        system,
        {"people": {"person": {}}},
    )
    value = simulation.calculate("net_income", "2024")
    # base_income defaults to 1000; fee is 300. net_income must be 700.
    # Before the fix, it was 1000 + 300 = 1300 (off by 2 * 300).
    assert value[0] == 700.0
