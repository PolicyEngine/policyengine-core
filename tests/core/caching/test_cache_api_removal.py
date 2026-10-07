"""Final cache API contracts; unit cases use no country model or filesystem."""

import copy
import operator
import pickle
from collections.abc import Mapping

import numpy as np
import pytest

from policyengine_core.caching import (
    InvalidCacheKeyError,
    InvalidCacheValueError,
)
from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.data_storage.immutable_array_cache import (
    CachedArrayEntry,
    ImmutableArrayCache,
)
from policyengine_core.entities import build_entity
from policyengine_core.parameters import ParameterNode
from policyengine_core.parameters.parameter_at_instant_cache import (
    ParameterAtInstantCache,
)
from policyengine_core.periods import period
from policyengine_core.simulations import Simulation
from policyengine_core.simulations.simulation_result_cache import (
    ResultCacheKey,
    SimulationResultCache,
    SuppliedInputKey,
)
from policyengine_core.taxbenefitsystems import TaxBenefitSystem


MONTH = period("2025-01")
RESULT_KEY = ResultCacheKey("income", MONTH)
INPUT_KEY = SuppliedInputKey("income", "default", MONTH)


@pytest.fixture(params=["arrays", "results"], ids=["immutable-arrays", "results"])
def populated_cache(request):
    if request.param == "arrays":
        cache = ImmutableArrayCache()
        key = "default:2025-01"
        value = CachedArrayEntry(np.array([4]))
    else:
        cache = SimulationResultCache()
        key, value = RESULT_KEY, "calculated"
    cache.put(key, value)
    return cache, key, value


@pytest.mark.parametrize(
    "mutation",
    [operator.setitem, operator.delitem],
    ids=["reject-replacement", "reject-deletion"],
)
def test_entries_observation_cannot_mutate_cache(populated_cache, mutation):
    cache, key, value = populated_cache
    snapshot = cache.entries
    assert isinstance(snapshot, Mapping)
    with pytest.raises(TypeError):
        if mutation is operator.setitem:
            mutation(snapshot, key, value)
        else:
            mutation(snapshot, key)
    assert cache.get(key) is value


@pytest.mark.parametrize("operation", ["replace", "clear"], ids=["replace", "clear"])
def test_entries_observation_is_a_detached_snapshot(populated_cache, operation):
    cache, key, value = populated_cache
    snapshot = cache.entries
    if operation == "replace":
        replacement = (
            CachedArrayEntry(np.array([9]))
            if isinstance(value, CachedArrayEntry)
            else "new"
        )
        cache.put(key, replacement)
        assert cache.get(key) is replacement
    else:
        cache.clear()
        assert len(cache) == 0
    assert snapshot[key] is value


def test_typed_bulk_entries_preserve_values_without_aliasing_mapping(populated_cache):
    cache, key, value = populated_cache
    supplied = {key: value}
    cache.replace_entries(supplied)
    supplied.clear()
    assert cache.get(key) is value


def test_empty_typed_bulk_entries_clear_only_owned_index(populated_cache):
    cache, key, value = populated_cache
    snapshot = cache.entries
    cache.replace_entries({})
    assert len(cache) == 0
    assert snapshot[key] is value


@pytest.mark.parametrize(
    "attribute,expected_type",
    [
        ("invalidated", frozenset),
        ("supplied_inputs", frozenset),
        ("input_contexts", tuple),
    ],
    ids=["pending-invalidations", "supplied-provenance", "input-context-stack"],
)
def test_result_metadata_observations_are_immutable(attribute, expected_type):
    cache = SimulationResultCache()
    cache.invalidate(RESULT_KEY)
    cache.record_supplied_input(*INPUT_KEY)
    with cache.supplied_input_context("default"):
        observation = getattr(cache, attribute)
        assert isinstance(observation, expected_type)
        with pytest.raises(AttributeError):
            observation.clear()
        assert cache.current_input_branch == "default"
    assert cache.invalidated == frozenset({RESULT_KEY})
    assert cache.supplied_input_keys() == frozenset({INPUT_KEY})


@pytest.mark.parametrize(
    "attribute",
    ["invalidated", "supplied_inputs", "input_contexts"],
    ids=["pending-invalidations", "supplied-provenance", "input-context-stack"],
)
def test_result_metadata_snapshots_do_not_change_after_mutation(attribute):
    cache = SimulationResultCache()
    snapshot = getattr(cache, attribute)
    if attribute == "invalidated":
        cache.invalidate(RESULT_KEY)
        assert cache.invalidated == frozenset({RESULT_KEY})
    elif attribute == "supplied_inputs":
        cache.record_supplied_input(*INPUT_KEY)
        assert cache.supplied_input_keys() == frozenset({INPUT_KEY})
    else:
        with cache.supplied_input_context("branch"):
            assert cache.current_input_branch == "branch"
            assert not snapshot
    assert not snapshot


@pytest.mark.parametrize(
    "method,attribute,key",
    [
        ("replace_invalidated", "invalidated", RESULT_KEY),
        ("replace_supplied_inputs", "supplied_inputs", INPUT_KEY),
    ],
    ids=["typed-pending-invalidations", "typed-supplied-inputs"],
)
def test_typed_bulk_metadata_detaches_caller_container(method, attribute, key):
    cache = SimulationResultCache()
    source = {key}
    getattr(cache, method)(source)
    source.clear()
    assert getattr(cache, attribute) == frozenset({key})
    getattr(cache, method)(set())
    assert not getattr(cache, attribute)


@pytest.mark.parametrize(
    "method,bad_key",
    [
        ("replace_entries", ("income", MONTH)),
        ("replace_entries", ResultCacheKey("", MONTH)),
        ("replace_invalidated", ("legacy_income", MONTH)),
        ("replace_invalidated", ResultCacheKey(7, MONTH)),
        ("replace_supplied_inputs", ("legacy_income", "default", MONTH)),
        ("replace_supplied_inputs", SuppliedInputKey("", "default", MONTH)),
        ("replace_supplied_inputs", SuppliedInputKey("income", "bad/branch", MONTH)),
        ("replace_supplied_inputs", SuppliedInputKey("income", "default", None)),
    ],
    ids=[
        "entries-legacy-tuple",
        "entries-empty-variable",
        "pending-legacy-tuple",
        "pending-nonstring-variable",
        "supplied-legacy-tuple",
        "supplied-empty-variable",
        "supplied-path-separator",
        "supplied-missing-period",
    ],
)
def test_bulk_schema_rejection_preserves_all_result_state(method, bad_key):
    cache = SimulationResultCache()
    cache.put(RESULT_KEY, "original")
    cache.invalidate(RESULT_KEY)
    cache.record_supplied_input(*INPUT_KEY)
    before = (
        dict(cache.entries),
        cache.invalidated,
        cache.supplied_input_keys(),
        cache.input_revision,
    )
    if method == "replace_entries":
        candidate = {ResultCacheKey("valid", MONTH): "new", bad_key: "bad"}
    else:
        valid = INPUT_KEY if method == "replace_supplied_inputs" else RESULT_KEY
        candidate = {valid, bad_key}
    with pytest.raises(InvalidCacheKeyError):
        getattr(cache, method)(candidate)
    after = (
        dict(cache.entries),
        cache.invalidated,
        cache.supplied_input_keys(),
        cache.input_revision,
    )
    assert after == before


def test_nested_supported_contexts_restore_parent_after_exception():
    cache = SimulationResultCache()
    with cache.supplied_input_context("parent"):
        with pytest.raises(RuntimeError, match="formula failed"):
            with cache.supplied_input_context("child"):
                assert cache.input_contexts == ("parent", "child")
                raise RuntimeError("formula failed")
        assert cache.current_input_branch == "parent"
    assert cache.current_input_branch is None
    assert cache.input_contexts == ()


def test_removed_context_replacement_cannot_shadow_supported_context_manager():
    cache = SimulationResultCache()
    with pytest.raises(AttributeError):
        getattr(cache, "replace_input_contexts")
    with pytest.raises(AttributeError):
        setattr(cache, "replace_input_contexts", lambda value: None)
    assert "replace_input_contexts" not in vars(cache)
    with cache.supplied_input_context("supported"):
        assert cache.current_input_branch == "supported"


@pytest.mark.parametrize(
    "raw_value",
    [np.array([9]), [9], 9],
    ids=["numpy-array", "python-list", "scalar"],
)
def test_raw_array_bulk_conversion_is_rejected_atomically(raw_value):
    cache = ImmutableArrayCache()
    original = cache.put_array("original", np.array([4]))
    with pytest.raises(InvalidCacheValueError):
        cache.replace_entries(
            {"valid": CachedArrayEntry(np.array([7])), "legacy": raw_value}
        )
    assert tuple(cache.entries) == ("original",)
    assert cache.entry("original") is original


def test_typed_result_with_unhashable_period_is_rejected_without_mutation():
    cache = SimulationResultCache()
    cache.put(RESULT_KEY, "original")
    with pytest.raises(InvalidCacheKeyError):
        cache.put(ResultCacheKey("income", []), "replacement")
    assert dict(cache.entries) == {RESULT_KEY: "original"}


@pytest.mark.parametrize(
    "attribute", ["_arrays", "_shared"], ids=["raw-array-index", "old-sharing-flags"]
)
def test_removed_storage_attributes_reject_reads_and_shadow_writes(attribute):
    storage = InMemoryStorage(False)
    storage.put(np.array([4]), MONTH, supplied=True)
    with pytest.raises(AttributeError):
        getattr(storage, attribute)
    with pytest.raises(AttributeError):
        setattr(storage, attribute, {})
    assert attribute not in vars(storage)
    np.testing.assert_array_equal(storage.get(MONTH), [4])
    assert storage.is_supplied(MONTH)


def test_storage_exposes_the_typed_entry_cache_for_observation():
    storage = InMemoryStorage(False)
    storage.put(np.array([4]), MONTH, supplied=True)
    assert isinstance(storage.entry_cache, ImmutableArrayCache)
    assert len(storage.entry_cache.entries) == 1
    entry = next(iter(storage.entry_cache.entries.values()))
    np.testing.assert_array_equal(entry.read(), [4])
    assert not entry.read().flags.writeable


def test_storage_entry_cache_cannot_be_replaced_by_assignment():
    storage = InMemoryStorage(False)
    storage.put(np.array([4]), MONTH, supplied=True)
    original = storage.entry_cache
    with pytest.raises(AttributeError):
        storage.entry_cache = ImmutableArrayCache()
    assert storage.entry_cache is original
    assert storage.is_supplied(MONTH)


def test_storage_hot_paths_do_not_materialize_entry_snapshots(monkeypatch):
    def fail_snapshot(_cache):
        raise AssertionError("ordinary reads and writes must not copy the index")

    monkeypatch.setattr(ImmutableArrayCache, "entries", property(fail_snapshot))
    storage = InMemoryStorage(False)
    storage.put(np.array([4]), MONTH, supplied=True)
    np.testing.assert_array_equal(storage.get(MONTH), [4])
    assert storage.has(MONTH)
    assert storage.is_supplied(MONTH)
    assert not storage.is_derived(MONTH)


def _new_parameter_owner(kind):
    node = ParameterNode("root", data={"amount": {"values": {"2025-01-01": 4}}})
    if kind == "node":
        return node
    person = build_entity("person", "people", "Person", is_person=True)
    system = TaxBenefitSystem(entities=[person])
    system.replace_parameters(node)
    return system


@pytest.mark.parametrize(
    "kind,attribute",
    [("node", "_at_instant_cache"), ("system", "_parameters_at_instant_cache")],
    ids=["parameter-node", "tax-benefit-system"],
)
def test_removed_parameter_cache_attributes_reject_shadow_state(kind, attribute):
    owner = _new_parameter_owner(kind)
    with pytest.raises(AttributeError):
        getattr(owner, attribute)
    with pytest.raises(AttributeError):
        setattr(owner, attribute, {})
    assert attribute not in vars(owner)
    view = (
        owner("2025-01-01")
        if kind == "node"
        else owner.get_parameters_at_instant("2025-01-01")
    )
    assert view.amount == 4


@pytest.mark.parametrize(
    "kind", ["node", "system"], ids=["parameter-node", "tax-benefit-system"]
)
def test_public_parameter_cache_is_typed_and_cannot_be_replaced(kind):
    owner = _new_parameter_owner(kind)
    cache = owner.parameter_cache
    assert isinstance(cache, ParameterAtInstantCache)
    with pytest.raises(AttributeError):
        owner.parameter_cache = ParameterAtInstantCache()
    assert owner.parameter_cache is cache


def _serializable_cache(kind):
    if kind == "arrays":
        cache = ImmutableArrayCache()
        cache.put_array("stored", np.array([4]))
    else:
        cache = InMemoryStorage(False)
        cache.put(np.array([4]), MONTH, supplied=True)
    return cache


def _read_stored(cache, kind):
    return cache.get_array("stored") if kind == "arrays" else cache.get(MONTH)


@pytest.mark.parametrize(
    "kind", ["arrays", "storage"], ids=["immutable-arrays", "memory-storage"]
)
@pytest.mark.parametrize(
    "copy_mode", ["shallow", "deep", "pickle"], ids=["copy", "deepcopy", "pickle"]
)
def test_current_copy_formats_keep_immutable_payloads_and_independent_indexes(
    kind, copy_mode
):
    source = _serializable_cache(kind)
    if copy_mode == "shallow":
        duplicate = copy.copy(source)
    elif copy_mode == "deep":
        duplicate = copy.deepcopy(source)
    else:
        duplicate = pickle.loads(pickle.dumps(source))
    read = _read_stored(duplicate, kind)
    np.testing.assert_array_equal(read, [4])
    with pytest.raises(ValueError):
        read.flags.writeable = True
    if kind == "arrays":
        duplicate.put_array("stored", np.array([9]))
    else:
        assert duplicate.is_supplied(MONTH)
        duplicate.put(np.array([9]), MONTH, supplied=True)
    np.testing.assert_array_equal(_read_stored(source, kind), [4])
    np.testing.assert_array_equal(_read_stored(duplicate, kind), [9])


@pytest.mark.parametrize(
    "kind", ["arrays", "storage"], ids=["immutable-arrays", "memory-storage"]
)
def test_current_serialized_state_declares_supported_schema(kind):
    cache = _serializable_cache(kind)
    assert cache.__getstate__()["schema_version"] == 1


@pytest.mark.parametrize(
    "kind", ["arrays", "storage"], ids=["immutable-arrays", "memory-storage"]
)
@pytest.mark.parametrize(
    "version", [None, 99], ids=["unversioned-legacy", "unsupported-version"]
)
def test_unsupported_serialized_state_leaves_existing_cache_unchanged(kind, version):
    cache = _serializable_cache(kind)
    state = dict(cache.__getstate__())
    if version is None:
        state.pop("schema_version", None)
    else:
        state["schema_version"] = version
    with pytest.raises(ValueError):
        cache.__setstate__(state)
    np.testing.assert_array_equal(_read_stored(cache, kind), [4])
    if kind == "storage":
        assert cache.is_supplied(MONTH)


@pytest.mark.parametrize(
    "kind", ["arrays", "storage"], ids=["raw-array-values", "raw-storage-dictionary"]
)
def test_malformed_current_serialized_payload_is_rejected_atomically(kind):
    cache = _serializable_cache(kind)
    state = dict(cache.__getstate__())
    state["schema_version"] = 1
    if kind == "arrays":
        state["entries"] = {"legacy": np.array([9])}
    else:
        state["_entry_cache"] = {"default:2025-01": np.array([9])}
    with pytest.raises(ValueError):
        cache.__setstate__(state)
    np.testing.assert_array_equal(_read_stored(cache, kind), [4])


@pytest.mark.parametrize(
    "key,is_eternal",
    [
        (7, False),
        (("default", MONTH), False),
        ("default", False),
        ("default:not-a-period", False),
        ("default:month:2025-01:1", False),
        ("default:2025-01", True),
    ],
    ids=[
        "integer-key",
        "tuple-key",
        "missing-delimiter",
        "invalid-period",
        "noncanonical-period",
        "dated-eternity-entry",
    ],
)
def test_storage_restore_rejects_wrong_domain_keys_without_changing_receiver(
    key, is_eternal
):
    storage = InMemoryStorage(False)
    storage.put(np.array([4]), MONTH, supplied=True)
    malformed = ImmutableArrayCache()
    malformed.put_array(key, np.array([9]))
    state = {
        "schema_version": 1,
        "_entry_cache": malformed,
        "is_eternal": is_eternal,
        "_inputs": frozenset({key}),
        "_supplied": frozenset({key}),
    }
    with pytest.raises(ValueError):
        storage.__setstate__(state)
    np.testing.assert_array_equal(storage.get(MONTH), [4])
    assert storage.is_supplied(MONTH)
    assert storage.get_known_periods() == [MONTH]
    assert storage.is_eternal is False


def test_storage_restore_accepts_canonical_eternity_and_preserves_provenance():
    source = InMemoryStorage(True)
    source.put(np.array([4]), MONTH, branch_name="nested_branch", supplied=True)
    receiver = InMemoryStorage(False)
    receiver.__setstate__(source.__getstate__())
    assert receiver.is_eternal is True
    assert receiver.is_supplied("2028", "nested_branch")
    np.testing.assert_array_equal(receiver.get("2028", "nested_branch"), [4])
    receiver.delete(branch_name="nested_branch")
    assert source.is_supplied(MONTH, "nested_branch")


class TestSimulationInitializationIntegration:
    """Additional integration coverage; excluded from the standalone unit count."""

    @staticmethod
    def make_simulation():
        system = _new_parameter_owner("system")
        return Simulation(
            tax_benefit_system=system, populations=system.instantiate_entities()
        )

    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("_fast_cache", {}),
            ("invalidated_caches", set()),
            ("_user_input_keys", set()),
            ("_user_input_contexts", []),
        ],
        ids=[
            "fast-results",
            "pending-invalidations",
            "supplied-inputs",
            "input-contexts",
        ],
    )
    def test_removed_simulation_attributes_cannot_shadow_current_cache(
        self, attribute, value
    ):
        simulation = self.make_simulation()
        cache = simulation.result_cache
        with pytest.raises(AttributeError):
            getattr(simulation, attribute)
        with pytest.raises(AttributeError):
            setattr(simulation, attribute, value)
        assert attribute not in vars(simulation)
        assert simulation.result_cache is cache

    def test_missing_initialization_is_not_silently_recovered(self):
        simulation = Simulation.__new__(Simulation)
        with pytest.raises(AttributeError):
            _ = simulation.result_cache
        assert "_result_cache" not in vars(simulation)
