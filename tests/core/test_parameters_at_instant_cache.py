"""``TaxBenefitSystem.get_parameters_at_instant`` follows changes to the tree.

The system caches the parameters at each instant on top of the tree's own
cache. ``Parameter.update`` clears the tree's caches up to the root but not
the system's, which went on returning the node it cached before the change.
"""

from policyengine_core.country_template import CountryTaxBenefitSystem

INSTANT = "2015-01-01"


def test_repeated_requests_return_the_cached_node():
    system = CountryTaxBenefitSystem()
    first = system.get_parameters_at_instant(INSTANT)
    assert system.get_parameters_at_instant(INSTANT) is first


def test_a_request_after_an_update_reads_the_update():
    system = CountryTaxBenefitSystem()
    before = system.get_parameters_at_instant(INSTANT)
    assert before.taxes.income_tax_rate == 0.15

    taxes = system.parameters.taxes
    taxes.income_tax_rate.update(period="year:2015:1", value=0.99)
    taxes.housing_tax.rate.update(period="year:2015:1", value=77)

    after = system.get_parameters_at_instant(INSTANT)
    assert after is not before
    # Read before the update and not: both as the tree now holds them.
    assert (after.taxes.income_tax_rate, after.taxes.housing_tax.rate) == (0.99, 77)
    assert system.get_parameters_at_instant(INSTANT) is after


def test_an_update_refreshes_every_cached_instant():
    """``update`` clears the root's cache at every instant, so the system
    rebuilds each instant it had cached, not only the one the update names."""
    system = CountryTaxBenefitSystem()
    other_instant = system.get_parameters_at_instant("2016-01-01")
    system.get_parameters_at_instant(INSTANT)

    system.parameters.taxes.income_tax_rate.update(period="year:2015:1", value=0.99)

    assert system.get_parameters_at_instant("2016-01-01") is not other_instant
    assert system.get_parameters_at_instant("2016-01-01").taxes.income_tax_rate == (
        system.parameters.taxes.income_tax_rate("2016-01-01")
    )
