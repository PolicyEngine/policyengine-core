"""Simulations survive ``copy.deepcopy`` and a pickle round trip.

``copy`` and ``pickle`` rebuild an object by creating an empty instance and
then probing it for ``__setstate__`` before its ``__dict__`` is restored. A
``__getattr__`` that reads one of the instance's own attributes recurses on
that probe until the interpreter raises ``RecursionError``.
``Population.__getattr__`` did this through its projector lookup (it reads
``self.entity``), so ``copy.deepcopy(simulation)``,
``pickle.loads(pickle.dumps(simulation))`` and even ``copy.copy(population)``
failed on every simulation. The parameter wrappers that forward attribute
lookups (vectorial and tracing nodes) had the same recursion, and the
vectorial node also forwarded ``__deepcopy__`` to its numpy vector, so a deep
copy came back as a bare ``numpy.recarray``. Behind the recursion, a pickled
``EnumArray`` lost its ``possible_values``.

Pickles are for the process that wrote them: a tax-benefit system loads
variable files under module names no other process has.
"""

from __future__ import annotations

import copy
import io
import pickle

import numpy as np
import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.enums import Enum, EnumArray
from policyengine_core.parameters import VectorialParameterNodeAtInstant
from policyengine_core.populations import GroupPopulation, Population
from policyengine_core.tools import assert_near
from policyengine_core.tracers import FullTracer, TracingParameterNodeAtInstant
from tests.fixtures.simulation_copy import (
    COPIERS,
    FEB,
    JAN,
    DoubleIncomeTaxRate,
    build_simulation,
    rate_node,
)

# Population: lookups on an instance that copy or pickle has not filled yet.


@pytest.mark.parametrize("population_class", [Population, GroupPopulation])
def test_lookups_on_an_unfilled_population_raise_attribute_error(population_class):
    unfilled = population_class.__new__(population_class)

    assert getattr(unfilled, "__setstate__", None) is None
    assert not hasattr(unfilled, "household")
    with pytest.raises(AttributeError):
        unfilled.entity
    with pytest.raises(AttributeError):
        unfilled.simulation


def test_copy_protocol_lookups_on_a_population_find_nothing():
    simulation = build_simulation()

    assert getattr(simulation.persons, "__deepcopy__", None) is None
    assert getattr(simulation.household, "__setstate__", None) is None


def test_population_projector_shortcuts_still_resolve():
    simulation = build_simulation()

    assert_near(
        simulation.persons.household("rent", JAN), simulation.household("rent", JAN)
    )
    with pytest.raises(AttributeError, match="not a known attribute"):
        simulation.persons.not_an_entity


def test_shallow_copy_of_a_population_shares_its_holders():
    simulation = build_simulation()

    population = copy.copy(simulation.persons)

    assert population.entity is simulation.persons.entity
    assert population._holders is simulation.persons._holders


# Simulations.


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_simulation_calculates_what_the_original_does(copier):
    simulation = build_simulation(salaries=(1000, 2500))
    expected = simulation.calculate("income_tax", JAN)

    copied = copier(simulation)

    assert_near(copied.calculate("income_tax", JAN), expected)
    assert_near(
        copied.calculate("disposable_income", JAN),
        simulation.calculate("disposable_income", JAN),
    )
    assert copied.persons is not simulation.persons
    assert copied.persons.simulation is copied
    assert copied.household.simulation is copied
    assert copied.household.members is copied.persons
    assert copied.populations["person"] is copied.persons


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_inputs_set_on_a_copy_do_not_reach_the_original(copier):
    simulation = build_simulation()
    simulation.calculate("income_tax", JAN)

    copied = copier(simulation)
    copied.set_input("salary", FEB, [2000])

    assert_near(copied.calculate("income_tax", FEB), [300])
    assert simulation.persons.get_holder("salary").get_array(FEB) is None
    assert simulation.persons.get_holder("income_tax").get_array(FEB) is None


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_simulation_keeps_its_branches(copier):
    simulation = build_simulation()
    simulation.get_branch("child").set_input("salary", FEB, [2000])

    copied = copier(simulation)

    child = copied.branches["child"]
    assert child.parent_branch is copied
    assert child is not simulation.branches["child"]
    assert_near(child.calculate("income_tax", FEB), [300])


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_reform_simulation_keeps_the_reform(copier):
    reformed = DoubleIncomeTaxRate(CountryTaxBenefitSystem())
    simulation = build_simulation(tax_benefit_system=reformed)
    expected = simulation.calculate("income_tax", JAN)

    copied = copier(simulation)

    assert isinstance(copied.tax_benefit_system, DoubleIncomeTaxRate)
    assert_near(copied.calculate("income_tax", JAN), expected)
    assert_near(expected, [300])


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_traced_simulation_still_traces(copier):
    simulation = build_simulation()
    simulation.trace = True
    simulation.calculate("income_tax", JAN)

    copied = copier(simulation)
    copied.calculate("disposable_income", JAN)

    assert copied.tracer is not simulation.tracer
    assert "disposable_income<2025-01, (default)>" in copied.tracer.get_flat_trace()


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_simulation_keeps_cached_vectorial_parameter_nodes(copier):
    tax_benefit_system = CountryTaxBenefitSystem()
    tax_benefit_system.parameters.add_child("rate", rate_node())
    simulation = build_simulation(tax_benefit_system=tax_benefit_system)
    statuses = np.array(["couple", "single"])
    parameters = simulation.tax_benefit_system.get_parameters_at_instant("2015-01-01")
    assert_near(parameters.rate[statuses].owner, [500, 100])  # caches the node

    copied = copier(simulation)

    rate = copied.tax_benefit_system.get_parameters_at_instant("2015-01-01").rate
    assert isinstance(rate._vectorial_node, VectorialParameterNodeAtInstant)
    assert_near(rate[statuses].tenant, [700, 300])


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_simulation_keeps_enum_inputs(copier):
    simulation = build_simulation(
        salaries=(1000, 1000),
        households=[[0], [1]],
        occupancy=["owner", "free_lodger"],
    )

    copied = copier(simulation)

    status = copied.household("housing_occupancy_status", JAN)
    assert list(status.decode_to_str()) == ["owner", "free_lodger"]
    # The formula reads the enum from the stored array's ``possible_values``.
    housing_tax = copied.calculate("housing_tax", "2025")
    assert_near(housing_tax, simulation.calculate("housing_tax", "2025"))
    assert housing_tax[0] > 0
    assert housing_tax[1] == 0


# Enum arrays.


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickled_enum_array_keeps_its_possible_values(protocol):
    status = build_simulation(occupancy=["free_lodger"]).household(
        "housing_occupancy_status", JAN
    )

    restored = pickle.loads(pickle.dumps(status, protocol=protocol))

    assert type(restored) is EnumArray
    assert restored.possible_values is status.possible_values
    assert list(restored.decode_to_str()) == ["free_lodger"]
    assert (restored == status.possible_values.free_lodger).all()
    assert restored.dtype == status.dtype


def test_pickled_enum_array_without_possible_values_stays_without():
    restored = pickle.loads(pickle.dumps(EnumArray(np.array([0, 1]))))

    assert type(restored) is EnumArray
    assert restored.possible_values is None
    assert restored.tolist() == [0, 1]


@pytest.mark.parametrize(
    "module_name, qualified_name",
    [
        ("module_only_another_process_loaded", "HousingOccupancyStatus"),
        ("policyengine_core.enums", "NoSuchEnum"),
    ],
)
def test_enum_array_whose_enum_cannot_be_found_unpickles_without_it(
    module_name, qualified_name
):
    """An enum from a variable file has a module name only its process knows."""

    class Unreachable(Enum):
        a = "a"
        b = "b"

    Unreachable.__module__ = module_name
    Unreachable.__qualname__ = qualified_name
    array = EnumArray(np.array([1, 0, 1]), Unreachable)

    restored = pickle.loads(pickle.dumps(array))

    assert type(restored) is EnumArray
    assert restored.possible_values is None
    assert restored.tolist() == [1, 0, 1]


def test_enum_array_pickled_by_an_earlier_release_still_loads():
    """Earlier releases pickled an EnumArray as numpy pickles any ndarray."""
    status = build_simulation(occupancy=["owner"]).household(
        "housing_occupancy_status", JAN
    )

    class EarlierPickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, EnumArray):
                return np.ndarray.__reduce__(obj)
            return NotImplemented

    buffer = io.BytesIO()
    EarlierPickler(buffer).dump(status)

    restored = pickle.loads(buffer.getvalue())

    assert type(restored) is EnumArray
    assert restored.tolist() == status.tolist()


def test_copied_enum_array_keeps_its_possible_values():
    status = build_simulation(occupancy=["owner"]).household(
        "housing_occupancy_status", JAN
    )

    for copied in (copy.copy(status), copy.deepcopy(status)):
        assert type(copied) is EnumArray
        assert copied.possible_values is status.possible_values
        assert list(copied.decode_to_str()) == ["owner"]
        assert not np.shares_memory(copied, status)


# Parameter wrappers that forward attribute lookups.


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_vectorial_parameter_node_is_a_vectorial_node(copier):
    node = rate_node()("2015-01-01")[np.array(["single", "couple", "single"])]

    copied = copier(node)

    assert type(copied) is VectorialParameterNodeAtInstant
    assert copied._name == node._name
    assert copied._instant_str == node._instant_str
    assert_near(copied.owner, [100, 500, 100])
    assert_near(copied[np.array(["owner", "tenant", "tenant"])], [100, 700, 300])
    assert not np.shares_memory(copied.vector, node.vector)


def test_vectorial_parameter_node_still_forwards_to_its_vector():
    node = rate_node()("2015-01-01")[np.array(["single", "couple"])]

    assert set(node.dtype.names) == {"owner", "tenant"}
    assert node.shape == (2,)


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_tracing_parameter_node_still_traces(copier):
    tracer = FullTracer()
    tracer.record_calculation_start("v", "2015-01")
    node = TracingParameterNodeAtInstant(
        rate_node()("2015-01-01"), tracer, branch_name="default"
    )

    copied = copier(node)

    assert type(copied) is TracingParameterNodeAtInstant
    assert copied.single.owner == 100
    assert copied.tracer.trees[0].parameters[-1].name == "rate.single.owner"
    assert node.tracer.trees[0].parameters == []


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copied_tracing_wrapper_of_a_vectorial_node_keeps_both_types(copier):
    vectorial = rate_node()("2015-01-01")[np.array(["couple", "single"])]
    node = TracingParameterNodeAtInstant(vectorial, FullTracer(), "default")

    copied = copier(node)

    assert type(copied) is TracingParameterNodeAtInstant
    assert type(copied.parameter_node_at_instant) is VectorialParameterNodeAtInstant
    assert_near(copied.parameter_node_at_instant.owner, [500, 100])


@pytest.mark.parametrize(
    "wrapper_class", [VectorialParameterNodeAtInstant, TracingParameterNodeAtInstant]
)
def test_lookups_on_an_unfilled_parameter_wrapper_raise_attribute_error(
    wrapper_class,
):
    unfilled = wrapper_class.__new__(wrapper_class)

    assert getattr(unfilled, "__setstate__", None) is None
    assert not hasattr(unfilled, "anything")
