import numpy as np

from policyengine_core.entities import Entity
from policyengine_core.model_api import *
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.taxbenefitsystems import TaxBenefitSystem


def test_bad_adds_raises_error():
    """A basic test of an ill-defined adds attribute in a variable."""
    Person = Entity("person", "people", "Person", "A person")
    system = TaxBenefitSystem([Person])

    class some_income(Variable):
        value_type = float
        entity = Person
        definition_period = ETERNITY
        label = "Income"
        adds = ["income_not_defined"]

    system.add_variables(some_income)

    simulation = SimulationBuilder().build_from_dict(
        system,
        {
            "people": {
                "person": {},
            },
        },
    )

    try:
        simulation.calculate("some_income")
        raise Exception("Should have raised an error.")
    except ValueError as e:
        pass
