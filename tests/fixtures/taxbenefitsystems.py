import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem


@pytest.fixture(scope="module")
def tax_benefit_system(request):
    system = CountryTaxBenefitSystem()
    import sys

    if sys.platform == "win32" or __import__("os").environ.get(
        "DIAG_LOCAL"
    ):  # TEMPORARY DIAGNOSTIC (to be reverted)
        from tests.conftest import _diag_write

        _diag_write(
            request.config,
            f"DIAG fixture {request.module.__name__} system {id(system):x} "
            f"params {id(system.parameters):x} trace {system.parameters.trace}",
        )
    return system


@pytest.fixture
def isolated_tax_benefit_system():
    return CountryTaxBenefitSystem()
