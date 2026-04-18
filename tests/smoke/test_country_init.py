"""Preflight: instantiating the country tax-benefit systems must not
raise under the current `policyengine-core` under test.

This catches a class of cross-repo regressions that full microsim
smoke tests don't: breakdown-validator errors at parameter load,
``_fast_cache``-attribute assumptions, and any other
system-initialisation changes core makes that happen to break
downstream country models.

Unlike ``tests/smoke/test_us.py`` (which is gated by
``RUN_SMOKE_TESTS=1`` and needs real microdata credentials), these
tests only exercise ``CountryTaxBenefitSystem()`` — they don't need
a dataset, don't need HF tokens, and complete in seconds. They run
on every PR against the policyengine-core master branch via
``SmokeTestForMultipleVersions``.

Each test is a soft-fail via ``skip`` if the corresponding country
package isn't installed — the workflow installs them explicitly, so
in CI this should always find the package. The skip just keeps
local test runs frictionless if you happen not to have both
policyengine-us and policyengine-uk on your machine.
"""

from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.smoke
def test_policyengine_us_tax_benefit_system_instantiates():
    if importlib.util.find_spec("policyengine_us") is None:
        pytest.skip("policyengine-us is not installed")
    # Importing this triggers the country package to load all
    # parameters and all Variables, which is where breakdown-validator
    # errors / _fast_cache-init bugs / etc. surface.
    from policyengine_us import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    assert len(system.variables) > 0
    assert len(list(system.parameters.get_descendants())) > 0


@pytest.mark.smoke
def test_policyengine_uk_tax_benefit_system_instantiates():
    if importlib.util.find_spec("policyengine_uk") is None:
        pytest.skip("policyengine-uk is not installed")
    from policyengine_uk import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    assert len(system.variables) > 0
    assert len(list(system.parameters.get_descendants())) > 0
