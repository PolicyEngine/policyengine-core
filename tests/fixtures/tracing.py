"""Helpers for tests of traced simulations."""

import copy

from policyengine_core.parameters import ParameterNodeAtInstant
from policyengine_core.simulations import SimulationBuilder
from tests.fixtures.branch_shared_arrays import SITUATION


def build_simulation(tax_benefit_system, trace=False):
    simulation = SimulationBuilder().build_from_entities(
        tax_benefit_system, copy.deepcopy(SITUATION)
    )
    simulation.trace = trace
    return simulation


def parameter_reads(trees):
    """Every parameter read recorded in ``trees``, as (name, branch name)."""
    reads = []

    def walk(node):
        reads.extend((read.name, read.branch_name) for read in node.parameters)
        for child in node.children:
            walk(child)

    for tree in trees:
        walk(tree)
    return reads


def assert_untraced(parameters, instant="2017-01-01"):
    """The parameter tree is as no simulation had ever traced it."""
    assert parameters.trace is False
    assert parameters.tracer is None
    assert parameters.branch_name is None
    assert type(parameters(instant)) is ParameterNodeAtInstant
    for node_at_instant in parameters._at_instant_cache.values():
        assert type(node_at_instant) is ParameterNodeAtInstant
