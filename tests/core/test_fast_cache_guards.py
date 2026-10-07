"""Cache operations use state established by the supported Core constructor."""

from policyengine_core.periods import period
from policyengine_core.simulations import SimulationBuilder


def _simulation(tax_benefit_system):
    return SimulationBuilder().build_from_variables(
        tax_benefit_system, {"salary": {"2017-01": [4000]}}
    )


def test_set_input_with_constructor_owned_cache(tax_benefit_system):
    sim = _simulation(tax_benefit_system)
    sim.set_input("salary", "2017-01", [5000])
    assert sim.get_supplied_input("salary", "2017-01")[0] == 5000


def test_delete_arrays_with_constructor_owned_cache(tax_benefit_system):
    sim = _simulation(tax_benefit_system)
    sim.delete_arrays("salary", "2017-01")
    assert sim.supplied_input_periods("salary") == []


def test_purge_cache_of_invalid_values_with_owned_cache(tax_benefit_system):
    sim = _simulation(tax_benefit_system)
    sim.calculate("income_tax", "2017-01")
    sim.invalidate_cache_entry("income_tax", period("2017-01"))
    sim.purge_cache_of_invalid_values()
    assert not sim.result_cache.invalidated
    assert sim.get_holder("income_tax").get_array(period("2017-01")) is None


def test_purge_empty_owned_cache(tax_benefit_system):
    sim = _simulation(tax_benefit_system)
    sim.purge_cache_of_invalid_values()
    assert not sim.result_cache.invalidated


def test_invalidate_cache_entry_accumulates_in_owned_cache(tax_benefit_system):
    sim = _simulation(tax_benefit_system)
    sim.invalidate_cache_entry("salary", period("2017-01"))
    sim.invalidate_cache_entry("income_tax", period("2017-02"))
    assert sim.result_cache.invalidated == {
        ("salary", period("2017-01")),
        ("income_tax", period("2017-02")),
    }
