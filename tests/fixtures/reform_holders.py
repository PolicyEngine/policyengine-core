"""Build small simulations for reforms applied after holders exist."""

from policyengine_core import periods
from policyengine_core.entities import Entity, GroupEntity
from policyengine_core.experimental import MemoryConfig
from policyengine_core.parameters import ParameterNode
from policyengine_core.reforms import Reform
from policyengine_core.simulations import Simulation
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.variables import Variable

YEAR = "2025"
Person = Entity("person", "persons", "Person", "")
Household = GroupEntity(
    "household",
    "households",
    "Household",
    "",
    roles=[{"key": "member", "plural": "members"}],
)


class eligible(Variable):
    value_type = bool
    entity = Person
    definition_period = periods.YEAR
    label = "Eligibility"


class award(Variable):
    value_type = float
    entity = Person
    definition_period = periods.YEAR
    default_value = 0
    defined_for = "eligible"
    label = "Award"

    def formula(person, period):
        return person.filled_array(100)


class income(Variable):
    value_type = float
    entity = Person
    definition_period = periods.YEAR
    label = "Income"


class annual_value(Variable):
    value_type = float
    entity = Person
    definition_period = periods.YEAR
    label = "Annual value"


class eternal_value(Variable):
    value_type = float
    entity = Person
    definition_period = periods.ETERNITY
    label = "Eternal value"


class group_value(Variable):
    value_type = float
    entity = Household
    definition_period = periods.YEAR
    label = "Group value"


def build(mask=(False, True)):
    """Put everyone in one household and record each person's eligibility."""
    system = TaxBenefitSystem([Person, Household])
    system.add_variables(
        eligible, award, income, annual_value, eternal_value, group_value
    )
    system.parameters = ParameterNode(
        "", data={"amount": {"values": {"2025-01-01": 1}}}
    )
    names = [f"p{index}" for index in range(len(mask))]
    return Simulation(
        tax_benefit_system=system,
        situation={
            "persons": {
                name: {"eligible": {YEAR: bool(value)}}
                for name, value in zip(names, mask)
            },
            "households": {"h": {"members": names}},
        },
    )


def change_default(value):
    """Return a reform that replaces the award's default with ``value``."""

    class ChangeDefault(Reform):
        def apply(self):
            self.update_variable(
                type(
                    "award",
                    (Variable,),
                    {"default_value": float(value), "defined_for": "eligible"},
                )
            )

    return ChangeDefault


def store_on_disk(simulation, *names):
    """Store future writes to the named holders on disk."""
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    for name in names:
        holder = simulation.get_holder(name)
        if holder._disk_storage is None:
            holder._disk_storage = holder.create_disk_storage()
        holder._on_disk_storable = True
