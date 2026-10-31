from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.simulations.individual_sim import IndividualSim


class TemplateIndividualSim(IndividualSim):
    tax_benefit_system = CountryTaxBenefitSystem
    required_entities = ["household"]
    default_roles = {"household": "parent"}


def test_placeholder_and_default_groups_are_not_supplied():
    # __init__ seeds an empty mapping for every group and build() fills the
    # required household with every person; neither was supplied by the
    # caller.
    simulation = TemplateIndividualSim(year=2024)
    simulation.add_person(name="Alicia")
    simulation.build()
    assert simulation.simulation.input_group_entities == frozenset()


def test_added_groups_are_supplied():
    simulation = TemplateIndividualSim(year=2024)
    simulation.add_person(name="Alicia")
    simulation.add_household(name="home", parents=["Alicia"])
    simulation.build()
    assert simulation.simulation.input_group_entities == frozenset({"household"})
