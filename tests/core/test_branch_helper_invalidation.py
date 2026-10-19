"""Branch dependency invalidation composes with input-helper overlap cleanup."""

from collections import Counter
import warnings

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.experimental import MemoryConfig
from policyengine_core.holders import (
    set_input_dispatch_by_period,
    set_input_divide_by_period,
)
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable


@pytest.mark.parametrize(
    "helper,year_input",
    [
        pytest.param(set_input_divide_by_period, [120.0, 240.0], id="divide"),
        pytest.param(set_input_dispatch_by_period, [10.0, 20.0], id="dispatch"),
    ],
)
@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
@pytest.mark.parametrize("year_string", ["2024", "0025"])
def test_branch_helper_drops_inherited_aggregates_and_dependents(
    helper, year_input, on_disk, year_string
):
    calls = Counter()

    class helper_flow(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        set_input = helper
        label = "Monthly input handled over a year"

    class helper_result(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "Annual result depending on the monthly input aggregate"

        def formula(person, period):
            calls[("result", person.simulation.branch_name)] += 1
            return person("helper_flow", period) * 2

    class helper_independent(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "Independent result calculated before the input was stored"

        def formula(person, period):
            calls[("independent", person.simulation.branch_name)] += 1
            return person.filled_array(7.0)

    system = CountryTaxBenefitSystem()
    variables = (helper_flow, helper_result, helper_independent)
    system.add_variables(*variables)

    def build():
        simulation = SimulationBuilder().build_default_simulation(system, count=2)
        if on_disk:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                simulation.memory_config = MemoryConfig(max_memory_occupation=0)
            # Holders take their storage configuration when first created.
            for variable in variables:
                population = simulation.get_variable_population(variable.__name__)
                population._holders.pop(variable.__name__, None)
        return simulation

    year = periods.period(year_string)
    january = year.first_month
    months = [january.offset(index) for index in range(12)]
    parent = build()
    parent.calculate("helper_independent", year)
    parent.set_input("helper_flow", january, [12.0, 24.0])
    parent_months = [january]
    if year_string == "0025":
        # Early-year fallback getters have a separate legacy parser issue.
        # Supply every month so this case exercises history and input replay.
        for month in months[1:]:
            parent.set_input("helper_flow", month, [0.0, 0.0])
        parent_months = months
    parent_inputs = {("helper_flow", "default", month) for month in parent_months}
    np.testing.assert_array_equal(parent.calculate("helper_flow", year), [12, 24])
    np.testing.assert_array_equal(parent.calculate("helper_result", year), [24, 48])

    branch = parent.get_branch("changed")
    branch.set_input("helper_flow", year, np.asarray(year_input))
    fresh = build()
    fresh.set_input("helper_flow", year, np.asarray(year_input))

    for name in ("helper_flow", "helper_result"):
        np.testing.assert_array_equal(
            branch.calculate(name, year), fresh.calculate(name, year)
        )
    for month in months:
        np.testing.assert_array_equal(branch.calculate("helper_flow", month), [10, 20])
    np.testing.assert_array_equal(branch.calculate("helper_independent", year), [7, 7])
    # The round-2 calculation frames keep independent inherited caches.
    assert calls[("independent", "changed")] == 0
    settled_calls = calls.copy()
    for _ in range(2):
        np.testing.assert_array_equal(
            branch.calculate("helper_result", year), [240, 480]
        )
    assert calls == settled_calls

    np.testing.assert_array_equal(parent.calculate("helper_flow", january), [12, 24])
    np.testing.assert_array_equal(parent.calculate("helper_flow", year), [12, 24])
    np.testing.assert_array_equal(parent.calculate("helper_result", year), [24, 48])

    branch_inputs = {("helper_flow", "changed", month) for month in months}
    assert branch_inputs | parent_inputs == branch._user_input_keys
    assert parent._user_input_keys == parent_inputs
    tier = "disk" if on_disk else "memory"
    holder = branch.get_holder("helper_flow")
    for month in months:
        assert holder._stores_user_input(month, "changed", tier)

    branch._invalidate_all_caches()

    assert branch._user_input_keys == branch_inputs | parent_inputs
    for month in months:
        assert holder._stores_user_input(month, "changed", tier)
        np.testing.assert_array_equal(branch.calculate("helper_flow", month), [10, 20])
    np.testing.assert_array_equal(branch.calculate("helper_result", year), [240, 480])
    np.testing.assert_array_equal(parent.calculate("helper_result", year), [24, 48])
    replayed_calls = calls.copy()
    branch.calculate("helper_result", year)
    assert calls == replayed_calls
