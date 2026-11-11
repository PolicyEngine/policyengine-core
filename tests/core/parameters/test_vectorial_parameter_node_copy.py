"""Vectorial parameter nodes survive ``copy``, ``copy.deepcopy`` and pickle.

``VectorialParameterNodeAtInstant.__getattr__`` hands every attribute the node
lacks to its ``vector``, a ``numpy.recarray``. ``copy`` and ``pickle`` create
an empty instance and probe it for ``__setstate__`` before they restore its
``__dict__``. With no ``vector`` yet, ``__getattr__`` looked ``vector`` up
through itself until ``RecursionError``. ``copy.deepcopy`` looks
``__deepcopy__`` up on the instance, got the vector's, and returned a bare
``recarray``.

``ParameterNodeAtInstant`` caches its vectorial node, and
``Reform.modify_parameters`` deep-copies the baseline's parameter tree, so a
reform found a ``recarray`` where its cached node should be.
"""

import copy

import numpy as np
import pytest

from policyengine_core.enums import EnumArray
from policyengine_core.parameters import (
    ParameterNotFoundError,
    VectorialParameterNodeAtInstant,
)
from policyengine_core.reforms import Reform
from policyengine_core.tools import assert_near
from tests.fixtures.vectorial_parameter_nodes import (
    COPIERS,
    DEEP_COPIERS,
    INSTANT,
    Family,
    SlottedRecords,
    fancy_indexing_parameters,
)


def single_rate_node() -> VectorialParameterNodeAtInstant:
    return fancy_indexing_parameters()(INSTANT).rate.single[
        np.array(["owner", "tenant"])
    ]


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copy_is_a_vectorial_node_with_the_same_contents(copier):
    node = single_rate_node()

    copied = copier(node)

    assert type(copied) is VectorialParameterNodeAtInstant
    assert copied._name == node._name == "rate.single"
    assert copied._instant_str == node._instant_str == INSTANT
    assert copied.vector.dtype == node.vector.dtype
    assert copied.vector.tobytes() == node.vector.tobytes()
    assert_near(copied.z1, [100, 300])
    assert_near(copied[np.array(["z2", "z1"])], [200, 300])


@pytest.mark.parametrize("copier", DEEP_COPIERS.values(), ids=DEEP_COPIERS.keys())
def test_deep_copy_has_a_vector_of_its_own(copier):
    node = single_rate_node()

    copied = copier(node)
    copied.vector.z1[:] = 0

    assert not np.shares_memory(copied.vector, node.vector)
    assert_near(node.z1, [100, 300])


def test_shallow_copy_shares_the_vector():
    node = single_rate_node()

    assert copy.copy(node).vector is node.vector


# ``object`` has no such attribute, so a lookup reaches ``__getattr__``.
@pytest.mark.parametrize(
    "name",
    [
        "__copy__",
        "__deepcopy__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__slots__",
    ],
)
def test_copy_protocol_is_not_read_from_the_vector(name):
    assert not hasattr(single_rate_node(), name)


# ``object`` has these, so a lookup never reaches ``__getattr__``.
@pytest.mark.parametrize("name", ["__getstate__", "__reduce__", "__reduce_ex__"])
def test_copy_protocol_methods_are_the_nodes_own(name):
    node = single_rate_node()

    assert getattr(node, name).__self__ is node


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_node_with_a_slotted_vector_is_copied(copier):
    """The node's own ``__getstate__`` keeps the vector's ``__slots__`` out."""
    vector = single_rate_node().vector.view(SlottedRecords)
    node = VectorialParameterNodeAtInstant("rate.single", vector, INSTANT)
    assert vector.__slots__ == ("note",)

    copied = copier(node)

    assert type(copied) is VectorialParameterNodeAtInstant
    assert copied.vector.tobytes() == vector.tobytes()
    assert_near(copied.z1, [100, 300])


@pytest.mark.parametrize(
    "name", ["vector", "_name", "dtype", "__setstate__", "__array_struct__"]
)
def test_lookups_on_an_unfilled_node_raise_attribute_error(name):
    """``copy`` and ``pickle`` probe a node before they fill it in."""
    unfilled = VectorialParameterNodeAtInstant.__new__(VectorialParameterNodeAtInstant)

    with pytest.raises(AttributeError):
        getattr(unfilled, name)


def test_numpy_reads_the_vector_through_the_node():
    node = single_rate_node()

    array = np.asarray(node)

    assert type(array) is np.ndarray
    assert array.shape == node.vector.shape
    assert array.dtype.itemsize == node.vector.dtype.itemsize
    assert array.tobytes() == node.vector.tobytes()
    assert node.dtype == node.vector.dtype
    assert node.shape == node.vector.shape


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_numpy_reads_a_copy_as_it_reads_the_original(copier):
    node = single_rate_node()

    original, copied = np.asarray(node), np.asarray(copier(node))

    assert copied.dtype == original.dtype
    assert copied.shape == original.shape
    assert copied.tobytes() == original.tobytes()


def test_child_node_read_by_name_is_a_vectorial_node():
    """``__getattr__`` built the child with one of its three arguments."""
    node = fancy_indexing_parameters()(INSTANT).rate[
        np.array(["single", "couple", "single"])
    ]

    for owner in (node.owner, node["owner"]):
        assert type(owner) is VectorialParameterNodeAtInstant
        assert owner._name == node._name == "rate"
        assert owner._instant_str == INSTANT
        assert_near(owner.z1, [100, 500, 100])
        assert_near(owner[np.array(["z2", "z1", "z1"])], [200, 500, 100])
        with pytest.raises(ParameterNotFoundError):
            owner[np.array(["z1", "z9", "z1"])]


@pytest.mark.parametrize("copier", COPIERS.values(), ids=COPIERS.keys())
def test_copy_reads_enum_keys_as_the_original_does(copier):
    node = VectorialParameterNodeAtInstant.build_from_node(
        fancy_indexing_parameters()(INSTANT).rate
    )
    families = EnumArray(np.array([1, 0, 1]), Family)
    owners = np.array(["owner", "owner", "owner"])
    expected = node[families][owners].z2  # Fills the node's enum lookup cache.

    copied = copier(node)

    # The cache is keyed by ``id(enum)``, which in another process can belong
    # to a different enum.
    assert "_enum_lut_cache" not in copied.__dict__
    assert_near(expected, [600, 200, 600])
    assert_near(copied[families][owners].z2, expected)
    family_items = np.array([Family.couple, Family.single, Family.couple], dtype=object)
    assert_near(copied[family_items][owners].z2, expected)


@pytest.mark.parametrize("copier", DEEP_COPIERS.values(), ids=DEEP_COPIERS.keys())
def test_copied_parameter_tree_keeps_its_cached_vectorial_node(copier):
    parameters = fancy_indexing_parameters()
    families = np.array(["single", "couple", "couple"])
    housings = np.array(["owner", "tenant", "owner"])
    expected = parameters(INSTANT).rate[families][housings].z2
    assert type(parameters(INSTANT).rate._vectorial_node) is (
        VectorialParameterNodeAtInstant
    )

    copied = copier(parameters)

    rate = copied(INSTANT).rate
    assert type(rate._vectorial_node) is VectorialParameterNodeAtInstant
    assert_near(expected, [200, 800, 600])
    assert_near(rate[families][housings].z2, expected)


def test_reform_that_modifies_parameters_keeps_the_cached_vectorial_node(
    isolated_tax_benefit_system,
):
    system = isolated_tax_benefit_system
    names = np.array(["age_of_retirement", "age_of_majority", "age_of_retirement"])
    # Caches the vectorial node of ``general`` at this instant.
    expected = system.get_parameters_at_instant("2017-01-01").general[names]

    class KeepParameters(Reform):
        def apply(self):
            # Deep-copies the baseline's tree, cache included.
            self.modify_parameters(lambda parameters: parameters)

    general = KeepParameters(system).get_parameters_at_instant("2017-01-01").general

    assert type(general._vectorial_node) is VectorialParameterNodeAtInstant
    assert_near(general[names], expected)
