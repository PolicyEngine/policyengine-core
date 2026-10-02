import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem


@pytest.fixture(scope="module")
def tax_benefit_system(request):
    system = CountryTaxBenefitSystem()
    import sys

    if sys.platform == "win32":  # TEMPORARY DIAGNOSTIC (to be reverted)
        sys.__stderr__.write(
            f"\nDIAG fixture {request.module.__name__} system {id(system):x} "
            f"params {id(system.parameters):x} trace {system.parameters.trace}\n"
        )
    return system


@pytest.fixture
def isolated_tax_benefit_system():
    return CountryTaxBenefitSystem()
