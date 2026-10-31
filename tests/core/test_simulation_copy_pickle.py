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
from policyengine_core.parameters import (
    ParameterNode,
    ParameterNodeAtInstant,
    VectorialParameterNodeAtInstant,
)
from policyengine_core.populations import GroupPopulation, Population
from policyengine_core.tools import assert_near
from policyengine_core.tracers import FullTracer, TracingParameterNodeAtInstant
from tests.fixtures.simulation_copy import (
    COPIERS,
    FEB,
    JAN,
    ChildEnumArray,
    DoubleIncomeTaxRate,
    build_simulation,
    rate_node,
)


class PropertyBackedPopulation(Population):
    @property
    def entity(self):
        return self._entity

    @entity.setter
    def entity(self, value):
        self._entity = value


class SlottedNodeState:
    __slots__ = ()

    def __getstate__(self):
        return self.__dict__, self.marker

    def __setstate__(self, state):
        attributes, self.marker = state
        self.__dict__.update(attributes)


class SlottedScalarNode(SlottedNodeState, ParameterNodeAtInstant):
    __slots__ = ("marker",)


class SlottedVectorialNode(SlottedNodeState, VectorialParameterNodeAtInstant):
    __slots__ = ("marker",)


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


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_property_backed_population_can_be_copied(copier):
    population = PropertyBackedPopulation(build_simulation().persons.entity)
    population.ids = ["person"]

    copied = copier(population)

    assert type(copied) is PropertyBackedPopulation
    assert copied.entity.key == population.entity.key
    assert copied.ids == population.ids


@pytest.mark.parametrize("protocol", range(6))
def test_property_backed_population_can_be_pickled(protocol):
    population = PropertyBackedPopulation(build_simulation().persons.entity)
    population.ids = ["person"]

    restored = pickle.loads(pickle.dumps(population, protocol=protocol))

    assert type(restored) is PropertyBackedPopulation
    assert restored.entity.key == population.entity.key
    assert restored.ids == population.ids


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
@pytest.mark.parametrize("array_class", [EnumArray, ChildEnumArray])
def test_pickled_enum_array_keeps_its_possible_values(protocol, array_class):
    status = build_simulation(occupancy=["free_lodger"]).household(
        "housing_occupancy_status", JAN
    )
    status = status.view(array_class)

    restored = pickle.loads(pickle.dumps(status, protocol=protocol))

    assert type(restored) is array_class
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


@pytest.mark.parametrize("protocol", range(6))
@pytest.mark.parametrize("array_class", [EnumArray, ChildEnumArray])
def test_enum_array_pickled_by_an_earlier_release_still_loads(protocol, array_class):
    """Earlier releases pickled an EnumArray as numpy pickles any ndarray."""
    status = build_simulation(occupancy=["owner"]).household(
        "housing_occupancy_status", JAN
    )
    status = status.view(array_class)

    class EarlierPickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, EnumArray):
                return np.ndarray.__reduce__(obj)
            return NotImplemented

    buffer = io.BytesIO()
    EarlierPickler(buffer, protocol=protocol).dump(status)

    restored = pickle.loads(buffer.getvalue())

    assert type(restored) is array_class
    assert restored.tolist() == status.tolist()
    assert not hasattr(restored, "possible_values")

    repickled = pickle.loads(pickle.dumps(restored, protocol=protocol))

    assert type(repickled) is array_class
    assert repickled.possible_values is None
    assert repickled.dtype == status.dtype
    assert repickled.shape == status.shape
    np.testing.assert_array_equal(repickled.view(np.ndarray), status.view(np.ndarray))


@pytest.mark.parametrize("array_class", [EnumArray, ChildEnumArray])
def test_copied_enum_array_keeps_its_possible_values(array_class):
    status = build_simulation(occupancy=["owner"]).household(
        "housing_occupancy_status", JAN
    )
    status = status.view(array_class)

    for copied in (copy.copy(status), copy.deepcopy(status)):
        assert type(copied) is array_class
        assert copied.possible_values is status.possible_values
        assert list(copied.decode_to_str()) == ["owner"]
        assert not np.shares_memory(copied, status)


# Parameter wrappers that forward attribute lookups.


def test_vectorial_parameter_field_named_vector_still_resolves():
    rate = ParameterNode(
        "rate",
        data={
            "single": {"vector": {"values": {"2015-01-01": 100}}},
            "couple": {"vector": {"values": {"2015-01-01": 500}}},
        },
    )
    vectorial = rate("2015-01-01")[np.array(["single", "couple"])]

    np.testing.assert_array_equal(vectorial["vector"], [100, 500])


@pytest.mark.parametrize("protocol", range(6))
@pytest.mark.parametrize("node_class", [SlottedScalarNode, SlottedVectorialNode])
def test_tracing_wrapper_of_a_slotted_node_can_be_pickled(protocol, node_class):
    source = rate_node()("2015-01-01")
    if node_class is SlottedVectorialNode:
        source = source[np.array(["single", "couple"])]
    inner = node_class.__new__(node_class)
    inner.__dict__.update(source.__dict__)
    inner.marker = "slotted"

    # The wrapper must preserve the protocol support of its wrapped node.
    unwrapped = pickle.loads(pickle.dumps(inner, protocol=protocol))
    wrapped = TracingParameterNodeAtInstant(inner, FullTracer(), "default")
    restored = pickle.loads(pickle.dumps(wrapped, protocol=protocol))

    assert type(restored) is TracingParameterNodeAtInstant
    restored_inner = restored.parameter_node_at_instant
    assert type(restored_inner) is type(unwrapped) is node_class
    assert restored_inner.marker == unwrapped.marker == "slotted"
    if node_class is SlottedVectorialNode:
        np.testing.assert_array_equal(restored_inner.owner, unwrapped.owner)
    else:
        assert restored_inner.single.owner == unwrapped.single.owner == 100


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
