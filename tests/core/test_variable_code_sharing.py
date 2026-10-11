"""Systems loaded from the same variable files run the same code objects.

Each system still gets its own modules, classes and functions; only the code
compiled from a file is shared, and a changed file is compiled again.
"""

import os
import sys
import textwrap

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.taxbenefitsystems import tax_benefit_system as tbs_module

VARIABLE_FILE = textwrap.dedent(
    """
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import MONTH
    from policyengine_core.variables import Variable


    class shared_code_probe(Variable):
        value_type = float
        entity = Person
        definition_period = MONTH
        label = "Shared code probe"

        def formula(person, period, parameters):
            return person("salary", period) * 0 + {value}
    """
)


def _formula(system, name):
    return system.variables[name].get_formula()


def _system_with_probe(path):
    system = CountryTaxBenefitSystem()
    system.add_variables_from_file(str(path))
    return system


def _value(system):
    from policyengine_core.simulations import SimulationBuilder

    simulation = SimulationBuilder().build_from_entities(
        system, {"persons": {"a": {"salary": {"2024-01": 100}}}}
    )
    return simulation.calculate("shared_code_probe", "2024-01")[0]


def test_systems_share_code_but_not_modules_classes_or_functions():
    first = CountryTaxBenefitSystem()
    second = CountryTaxBenefitSystem()
    first_class = type(first.variables["disposable_income"])
    second_class = type(second.variables["disposable_income"])
    first_formula = _formula(first, "disposable_income")
    second_formula = _formula(second, "disposable_income")

    assert first_class is not second_class
    assert (
        sys.modules[first_class.__module__]
        is not (sys.modules[second_class.__module__])
    )
    assert first_formula is not second_formula
    assert first_formula.__globals__ is not second_formula.__globals__
    assert first_formula.__code__ is second_formula.__code__


def test_shared_code_is_compiled_from_the_current_source():
    CountryTaxBenefitSystem()
    assert tbs_module._variable_file_code
    for (path, mtime_ns, size), code in tbs_module._variable_file_code.items():
        if not os.path.exists(path):
            continue  # a temporary file from another test
        stat = os.stat(path)
        if (stat.st_mtime_ns, stat.st_size) != (mtime_ns, size):
            continue  # an older version of a file that changed
        with open(path, encoding="utf-8") as f:
            source = f.read()
        assert code == compile(source, path, "exec", dont_inherit=True), path


def test_a_changed_file_is_compiled_again(tmp_path):
    path = tmp_path / "probe.py"
    path.write_text(VARIABLE_FILE.format(value=1))
    first = _system_with_probe(path)
    second = _system_with_probe(path)
    path.write_text(VARIABLE_FILE.format(value=20))
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    third = _system_with_probe(path)

    assert _formula(first, "shared_code_probe").__code__ is (
        _formula(second, "shared_code_probe").__code__
    )
    assert _formula(third, "shared_code_probe").__code__ is not (
        _formula(first, "shared_code_probe").__code__
    )
    assert _value(first) == pytest.approx(1)
    assert _value(second) == pytest.approx(1)
    assert _value(third) == pytest.approx(20)
