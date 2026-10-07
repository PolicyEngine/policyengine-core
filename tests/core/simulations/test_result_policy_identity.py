"""Unit tests for non-mutating policy identity observations."""

from copy import copy

from policyengine_core.parameters import ParameterNode
from policyengine_core.taxbenefitsystems import TaxBenefitSystem


class ObservedPolicy(TaxBenefitSystem):
    """Stand in for a country's copy-on-write parameter descriptor."""

    @property
    def parameters(self):
        self.reads += 1
        return self.tree

    @parameters.setter
    def parameters(self, value):
        self.tree = value


def policy():
    system = object.__new__(ObservedPolicy)
    system.reads = 0
    system.variables = {}
    system.replace_parameters(
        ParameterNode("", data={"amount": {"values": {"2025-01-01": 1}}})
    )
    return system


def test_observing_identity_does_not_read_parameter_descriptor():
    system = policy()
    token = system.result_cache_token
    assert token == system.result_cache_token
    assert system.reads == 0


def test_shallow_wrapper_with_same_policy_has_same_identity():
    system = policy()
    wrapper = copy(system)
    assert wrapper is not system
    assert wrapper.result_cache_token == system.result_cache_token
    assert wrapper.reads == system.reads == 0


def test_observation_detects_shared_parameter_revision_without_getter():
    system = policy()
    wrapper = copy(system)
    before = system.result_cache_token
    system.tree.amount.update(period="2025", value=2)
    assert system.result_cache_token != before
    assert wrapper.result_cache_token == system.result_cache_token
    assert wrapper.reads == system.reads == 0


def test_installing_different_tree_changes_identity_without_getter():
    system = policy()
    before = system.result_cache_token
    system.replace_parameters(system.tree.clone())
    assert system.result_cache_token != before
    assert system.reads == 0


def test_copying_variable_registry_changes_identity():
    system = policy()
    wrapper = copy(system)
    wrapper.variables = dict(system.variables)
    assert wrapper.result_cache_token != system.result_cache_token
    assert wrapper.reads == system.reads == 0


def test_legacy_ordinary_root_assignment_is_observed_without_cache_rebind():
    system = object.__new__(TaxBenefitSystem)
    system.variables = {}
    system.replace_parameters(ParameterNode("", data={}))
    before = system.result_cache_token
    system.parameters = ParameterNode("", data={})
    assert system.result_cache_token != before


def test_legacy_clearing_ordinary_root_changes_identity():
    system = object.__new__(TaxBenefitSystem)
    system.variables = {}
    system.replace_parameters(ParameterNode("", data={}))
    before = system.result_cache_token
    system.parameters = None
    assert system.result_cache_token != before
