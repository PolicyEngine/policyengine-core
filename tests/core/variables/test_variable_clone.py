"""Tests for ``Variable.clone()`` — policyengine-core#502.

Characterization of the existing contract for normal (non-reform) variables:
a clone preserves the variable's attributes, is the same class, and is
independent of the original. These pass on the current implementation and
guard the behaviour a later fix must keep.
"""

import policyengine_core.country_template as country_template
from policyengine_core.model_api import Reform, Variable
from policyengine_core.periods import MONTH

tax_benefit_system = country_template.CountryTaxBenefitSystem()


def test_clone_preserves_attributes():
    original = tax_benefit_system.get_variable("disposable_income")
    clone = original.clone()
    assert clone.value_type == original.value_type
    assert clone.entity.key == original.entity.key
    assert clone.definition_period == original.definition_period
    assert clone.label == original.label
    assert list(clone.formulas.keys()) == list(original.formulas.keys())


def test_clone_is_same_class():
    original = tax_benefit_system.get_variable("disposable_income")
    assert type(original.clone()) is type(original)


def test_clone_formulas_are_independent():
    # Fresh system so a (buggy) shared-container clone can't leak into other
    # tests: mutating the clone must not affect the original.
    system = country_template.CountryTaxBenefitSystem()
    original = system.get_variable("disposable_income")
    assert len(original.formulas) > 0
    clone = original.clone()
    clone.formulas.clear()
    assert len(original.formulas) > 0


def test_clone_update_variable_label_only():
    # A reform that overrides only the label inherits value_type/entity/
    # formula from the baseline; cloning it must keep that merged state.
    class disposable_income(Variable):
        label = "Updated label only"

    class reform(Reform):
        def apply(self):
            self.update_variable(disposable_income)

    system = reform(country_template.CountryTaxBenefitSystem())
    clone = system.get_variable("disposable_income").clone()

    assert clone.value_type == float  # inherited from baseline
    assert clone.entity.key == "person"
    assert clone.definition_period == MONTH
    assert clone.label == "Updated label only"  # the override
    assert len(clone.formulas) > 0  # inherited formula preserved


def test_clone_update_variable_adds_only():
    # A reform that redeclares a formula variable with adds only inherits the
    # baseline formula; cloning must keep the merged state.
    class disposable_income(Variable):
        adds = ["salary"]

    class reform(Reform):
        def apply(self):
            self.update_variable(disposable_income)

    system = reform(country_template.CountryTaxBenefitSystem())
    clone = system.get_variable("disposable_income").clone()

    assert clone.value_type == float
    assert clone.adds == ["salary"]
    assert len(clone.formulas) > 0  # inherited formula preserved
