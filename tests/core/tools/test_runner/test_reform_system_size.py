"""A reform system holds one copy of the parameter tree, and the cache never
holds more reform systems than its size, even while it builds one."""

import gc
import weakref

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.reforms import Reform
from policyengine_core.tools.test_runner import _get_tax_benefit_system

NEW_TAX = "policyengine_core.country_template.reforms.add_new_tax.add_new_tax"


def test_a_named_reform_is_built_on_the_reform_free_system_and_leaves_it_alone():
    baseline = CountryTaxBenefitSystem()
    reform_free = _get_tax_benefit_system(baseline, [], [])
    variables = dict(reform_free.variables)
    parameters = reform_free.parameters
    entities = list(reform_free.entities)

    system = _get_tax_benefit_system(baseline, [NEW_TAX], [])

    assert system.baseline is reform_free
    assert system.parameters is not parameters
    assert "new_tax" in system.variables
    assert "new_tax" not in reform_free.variables
    assert reform_free.variables == variables
    assert reform_free.parameters is parameters
    assert reform_free.entities == entities
    assert all(
        entity._tax_benefit_system is reform_free for entity in reform_free.entities
    )


def test_a_reform_function_is_applied_to_a_clone():
    """Only reforms that copy what they change are built on the shared system."""
    baseline = CountryTaxBenefitSystem()
    reform_free = _get_tax_benefit_system(baseline, [], [])
    received = []

    def reform_function(system):
        received.append(system)
        return system

    _get_tax_benefit_system(baseline, [reform_function], [], reform_key="function")
    assert received and received[0] is not reform_free


def test_the_cache_releases_a_system_before_building_the_one_replacing_it():
    baseline = CountryTaxBenefitSystem()
    first = weakref.ref(_get_tax_benefit_system(baseline, [NEW_TAX], [], cache_size=1))
    alive_while_building = []

    class probe(Reform):
        def apply(self):
            gc.collect()
            alive_while_building.append(first() is not None)

    _get_tax_benefit_system(baseline, [probe], [], cache_size=1)
    assert alive_while_building == [False]
