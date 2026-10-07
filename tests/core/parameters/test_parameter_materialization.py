from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.errors import ParameterNotFoundError
from policyengine_core.parameters import (
    EagerParameterMaterializer,
    LazyParameterMaterializer,
    LazyParameterNodeAtInstant,
    ParameterAtInstantCache,
    ParameterMaterializer,
    ParameterNode,
    ParameterNodeAtInstant,
    ParameterTreeRevision,
    StaleParameterViewError,
)
from policyengine_core.periods import instant
from policyengine_core.tracers import FullTracer, TracingParameterNodeAtInstant


def make_parameter_tree() -> ParameterNode:
    return ParameterNode(
        "root",
        data={
            "amount": {
                "values": {
                    "2020-01-01": 100,
                    "2026-01-01": 200,
                }
            },
            "group": {
                "rate": {
                    "values": {
                        "2020-01-01": 0.2,
                        "2026-01-01": 0.3,
                    }
                },
                "threshold": {"values": {"2020-01-01": 1_000}},
            },
            "future": {"values": {"2030-01-01": 999}},
            "scale": {
                "brackets": [
                    {
                        "threshold": {"values": {"2020-01-01": 0}},
                        "rate": {"values": {"2020-01-01": 0.1}},
                    },
                    {
                        "threshold": {"values": {"2020-01-01": 1_000}},
                        "rate": {"values": {"2020-01-01": 0.2}},
                    },
                ]
            },
        },
    )


def test_parameter_nodes_use_lazy_materialization_by_default() -> None:
    node = make_parameter_tree()

    view = node("2026-01-01")

    assert type(node.parameter_materializer) is LazyParameterMaterializer
    assert type(view) is LazyParameterNodeAtInstant
    assert view.is_materialized is False


def test_explicit_eager_materialization_remains_available() -> None:
    node = make_parameter_tree()
    node.set_parameter_materializer(EagerParameterMaterializer())

    view = node("2026-01-01")

    assert type(node.parameter_materializer) is EagerParameterMaterializer
    assert type(view) is ParameterNodeAtInstant
    assert set(view._children) == {"amount", "group", "scale"}


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("initial-zero", id="initial-zero"),
        pytest.param("initial-positive", id="initial-positive"),
        pytest.param("advance-once", id="advance-once"),
        pytest.param("advance-twice", id="advance-twice"),
        pytest.param("validate-current", id="validate-current"),
        pytest.param("reject-stale", id="reject-stale"),
        pytest.param("shallow-independent", id="shallow-independent"),
        pytest.param("deep-independent", id="deep-independent"),
        pytest.param("pickle-independent", id="pickle-independent"),
        pytest.param("clone-independent", id="clone-independent"),
    ],
)
def test_parameter_tree_revision_behavior(scenario: str) -> None:
    revision = ParameterTreeRevision(4 if scenario == "initial-positive" else 0)

    if scenario == "initial-zero":
        assert revision.current == 0
    elif scenario == "initial-positive":
        assert revision.current == 4
    elif scenario == "advance-once":
        assert revision.advance() == revision.current == 1
    elif scenario == "advance-twice":
        revision.advance()
        assert revision.advance() == revision.current == 2
    elif scenario == "validate-current":
        revision.validate(0)
        assert revision.current == 0
    elif scenario == "reject-stale":
        revision.advance()
        with pytest.raises(StaleParameterViewError, match="revision 0"):
            revision.validate(0)
    else:
        if scenario == "shallow-independent":
            duplicate = copy.copy(revision)
        elif scenario == "deep-independent":
            duplicate = copy.deepcopy(revision)
        elif scenario == "pickle-independent":
            duplicate = pickle.loads(pickle.dumps(revision))
        else:
            duplicate = revision.clone()
        duplicate.advance()
        assert revision.current == 0
        assert duplicate.current == 1


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param(1.5, id="float"),
        pytest.param("1", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_parameter_tree_revision_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        ParameterTreeRevision(value)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("eager-name", id="eager-name"),
        pytest.param("lazy-name", id="lazy-name"),
        pytest.param("eager-explicit-view", id="eager-explicit-view"),
        pytest.param("eager-children-built", id="eager-children-built"),
        pytest.param("lazy-view", id="lazy-view"),
        pytest.param("lazy-unmaterialized", id="lazy-unmaterialized"),
        pytest.param("lazy-scalar-count", id="lazy-scalar-count"),
        pytest.param("lazy-repeat-no-count", id="lazy-repeat-no-count"),
        pytest.param("lazy-nested-view", id="lazy-nested-view"),
        pytest.param("lazy-nested-value", id="lazy-nested-value"),
        pytest.param("lazy-missing", id="lazy-missing"),
        pytest.param("lazy-iteration", id="lazy-iteration"),
        pytest.param("lazy-representation", id="lazy-representation"),
        pytest.param("lazy-node-name", id="lazy-node-name"),
        pytest.param("lazy-instant", id="lazy-instant"),
    ],
)
def test_parameter_materializer_standard_behavior(scenario: str) -> None:
    eager = EagerParameterMaterializer()
    lazy = LazyParameterMaterializer()
    node = make_parameter_tree()

    if scenario == "eager-name":
        assert eager.strategy_name == "eager"
        return
    if scenario == "lazy-name":
        assert lazy.strategy_name == "lazy"
        return
    if scenario.startswith("eager"):
        node.set_parameter_materializer(eager)
        view = node("2026-01-01")
        if scenario == "eager-explicit-view":
            assert type(view) is ParameterNodeAtInstant
        else:
            assert set(view._children) == {"amount", "group", "scale"}
        return

    node.set_parameter_materializer(lazy)
    view = node("2026-01-01")
    if scenario == "lazy-view":
        assert type(view) is LazyParameterNodeAtInstant
    elif scenario == "lazy-unmaterialized":
        assert view.is_materialized is False
        assert view.materialized_child_count == 0
    elif scenario == "lazy-scalar-count":
        assert view.amount == 200
        assert view.materialized_child_count == 1
    elif scenario == "lazy-repeat-no-count":
        assert view.amount == view.amount == 200
        assert view.materialized_child_count == 1
    elif scenario == "lazy-nested-view":
        assert type(view.group) is LazyParameterNodeAtInstant
        assert view.group.is_materialized is False
    elif scenario == "lazy-nested-value":
        assert view.group.rate == 0.3
        assert view.group.materialized_child_count == 1
    elif scenario == "lazy-missing":
        with pytest.raises(ParameterNotFoundError, match="root.future"):
            view.future
        assert view.materialized_child_count == 0
    elif scenario == "lazy-iteration":
        assert list(view) == ["amount", "group", "scale"]
    elif scenario == "lazy-representation":
        assert "amount:" in repr(view)
        assert "group:" in repr(view)
    elif scenario == "lazy-node-name":
        assert view._name == "root"
        assert view.materialized_child_count == 0
    else:
        assert view._instant_str == "2026-01-01"
        assert view.materialized_child_count == 0


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("unresolved-attribute", id="unresolved-attribute"),
        pytest.param("resolved-attribute", id="resolved-attribute"),
        pytest.param("item", id="item"),
        pytest.param("iteration", id="iteration"),
        pytest.param("tuple-iteration", id="tuple-iteration"),
        pytest.param("representation", id="representation"),
        pytest.param("name", id="name"),
        pytest.param("instant", id="instant"),
        pytest.param("children", id="children"),
        pytest.param("materialized-flag", id="materialized-flag"),
        pytest.param("materialized-count", id="materialized-count"),
        pytest.param("containment", id="containment"),
        pytest.param("missing-attribute", id="missing-attribute"),
        pytest.param("missing-item", id="missing-item"),
        pytest.param("nested-unresolved", id="nested-unresolved"),
        pytest.param("nested-resolved", id="nested-resolved"),
        pytest.param("traced-attribute", id="traced-attribute"),
        pytest.param("vector-index", id="vector-index"),
    ],
)
def test_lazy_parameter_view_rejects_access_after_mutation(operation: str) -> None:
    node = make_parameter_tree()
    view = node("2026-01-01")
    nested = view.group
    traced = TracingParameterNodeAtInstant(view, FullTracer(), "default")
    if operation in {"resolved-attribute", "nested-resolved"}:
        assert view.amount == 200
        assert nested.rate == 0.3

    node.amount.update(value=250, start=instant("2026-01-01"))

    operations = {
        "unresolved-attribute": lambda: view.scale,
        "resolved-attribute": lambda: view.amount,
        "item": lambda: view["amount"],
        "iteration": lambda: iter(view),
        "tuple-iteration": lambda: tuple(view),
        "representation": lambda: repr(view),
        "name": lambda: view._name,
        "instant": lambda: view._instant_str,
        "children": lambda: view._children,
        "materialized-flag": lambda: view.is_materialized,
        "materialized-count": lambda: view.materialized_child_count,
        "containment": lambda: "amount" in view,
        "missing-attribute": lambda: view.unknown,
        "missing-item": lambda: view["unknown"],
        "nested-unresolved": lambda: nested.threshold,
        "nested-resolved": lambda: nested.rate,
        "traced-attribute": lambda: traced.amount,
        "vector-index": lambda: view[np.array(["amount"])],
    }
    with pytest.raises(StaleParameterViewError):
        operations[operation]()


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("hit-same-revision", id="hit-same-revision"),
        pytest.param("advance-removes", id="advance-removes"),
        pytest.param("external-get-miss", id="external-get-miss"),
        pytest.param("external-length-empty", id="external-length-empty"),
        pytest.param("external-iteration-empty", id="external-iteration-empty"),
        pytest.param("external-containment-false", id="external-containment-false"),
        pytest.param("external-info-empty", id="external-info-empty"),
        pytest.param("bind-same-preserves", id="bind-same-preserves"),
        pytest.param("bind-new-removes", id="bind-new-removes"),
        pytest.param("invalid-bind-preserves", id="invalid-bind-preserves"),
        pytest.param("composite-key", id="composite-key"),
        pytest.param("shallow-independent", id="shallow-independent"),
        pytest.param("deep-independent", id="deep-independent"),
        pytest.param("pickle-independent", id="pickle-independent"),
        pytest.param(
            "pickle-drops-obsolete-entries",
            id="pickle-drops-obsolete-entries",
        ),
        pytest.param("clear-keeps-revision", id="clear-keeps-revision"),
        pytest.param("disabled-tracks-revision", id="disabled-tracks-revision"),
    ],
)
def test_parameter_cache_revision_behavior(scenario: str) -> None:
    revision = ParameterTreeRevision(3)
    cache = ParameterAtInstantCache[str, object](
        revision=revision,
        enabled=scenario != "disabled-tracks-revision",
    )
    value = object()
    cache.put("2026-01-01", value)

    if scenario == "hit-same-revision":
        assert cache.get("2026-01-01") is value
    elif scenario == "advance-removes":
        assert cache.advance_revision() == 4
        assert len(cache) == 0
    elif scenario.startswith("external"):
        revision.advance()
        if scenario == "external-get-miss":
            with pytest.raises(KeyError):
                cache.get("2026-01-01")
        elif scenario == "external-length-empty":
            assert len(cache) == 0
        elif scenario == "external-iteration-empty":
            assert list(cache) == []
        elif scenario == "external-containment-false":
            assert "2026-01-01" not in cache
        else:
            assert cache.cache_info().entries == 0
    elif scenario == "bind-same-preserves":
        cache.bind_revision(revision)
        assert cache.get("2026-01-01") is value
    elif scenario == "bind-new-removes":
        cache.bind_revision(ParameterTreeRevision(3))
        assert len(cache) == 0
    elif scenario == "invalid-bind-preserves":
        with pytest.raises(TypeError):
            cache.bind_revision(3)
        assert cache.get("2026-01-01") is value
    elif scenario == "composite-key":
        internal_key = next(iter(cache._entries))
        assert (internal_key.revision, internal_key.instant) == (3, "2026-01-01")
    elif scenario in {
        "shallow-independent",
        "deep-independent",
        "pickle-independent",
    }:
        if scenario == "shallow-independent":
            duplicate = copy.copy(cache)
        elif scenario == "deep-independent":
            duplicate = copy.deepcopy(cache)
        else:
            duplicate = pickle.loads(pickle.dumps(cache))
        duplicate.advance_revision()
        assert cache.current_revision == 3
        assert duplicate.current_revision == 4
    elif scenario == "pickle-drops-obsolete-entries":
        revision.advance()
        duplicate = pickle.loads(pickle.dumps(cache))
        assert duplicate.current_revision == 4
        assert len(duplicate) == 0
    elif scenario == "clear-keeps-revision":
        cache.clear()
        assert cache.current_revision == 3
    else:
        assert cache.get_or_create("2027-01-01", object) is not None
        assert len(cache) == 1
        revision.advance()
        assert cache.current_revision == 4


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("update-advances", id="update-advances"),
        pytest.param("update-clears-ancestor", id="update-clears-ancestor"),
        pytest.param("update-refreshes-system", id="update-refreshes-system"),
        pytest.param("add-root-advances", id="add-root-advances"),
        pytest.param("add-nested-advances", id="add-nested-advances"),
        pytest.param("clone-revision-distinct", id="clone-revision-distinct"),
        pytest.param("clone-number-preserved", id="clone-number-preserved"),
        pytest.param("clone-strategy-preserved", id="clone-strategy-preserved"),
        pytest.param("clone-cache-independent", id="clone-cache-independent"),
        pytest.param("switch-to-lazy-advances", id="switch-to-lazy-advances"),
        pytest.param("switch-to-eager-stales", id="switch-to-eager-stales"),
        pytest.param("children-share-context", id="children-share-context"),
        pytest.param("brackets-share-context", id="brackets-share-context"),
        pytest.param("new-view-uses-update", id="new-view-uses-update"),
        pytest.param("eager-view-keeps-snapshot", id="eager-view-keeps-snapshot"),
        pytest.param("clear-does-not-advance", id="clear-does-not-advance"),
        pytest.param("mapping-cache-uses-revision", id="mapping-cache-uses-revision"),
        pytest.param("system-sets-materializer", id="system-sets-materializer"),
        pytest.param(
            "shared-systems-share-revision", id="shared-systems-share-revision"
        ),
        pytest.param("clear-before-tree-assignment", id="clear-before-tree-assignment"),
        pytest.param("replace-accepts-none", id="replace-accepts-none"),
        pytest.param("replace-rejects-other-types", id="replace-rejects-other-types"),
        pytest.param("replace-clears-dated-views", id="replace-clears-dated-views"),
        pytest.param("replace-binds-revision", id="replace-binds-revision"),
        pytest.param("replace-does-not-share-cache", id="replace-does-not-share-cache"),
    ],
)
def test_parameter_tree_revision_and_materializer_context(scenario: str) -> None:
    node = make_parameter_tree()
    initial_revision = node.parameter_revision.current

    if scenario == "update-advances":
        node.amount.update(value=250, start=instant("2026-01-01"))
        assert node.parameter_revision.current == initial_revision + 1
    elif scenario == "update-clears-ancestor":
        node("2026-01-01")
        node.group.rate.update(value=0.4, start=instant("2026-01-01"))
        assert len(node._at_instant_cache) == 0
    elif scenario == "update-refreshes-system":
        system = CountryTaxBenefitSystem()
        before = system.get_parameters_at_instant("2026-01-01")
        system.parameters.general.age_of_majority.update(
            value=99,
            start=instant("2026-01-01"),
        )
        after = system.get_parameters_at_instant("2026-01-01")
        assert after.general.age_of_majority == 99
        assert after is not before
    elif scenario == "add-root-advances":
        node.add_child("new", ParameterNode("root.new", data={"x": {"2020-01-01": 1}}))
        assert node.parameter_revision.current == initial_revision + 1
    elif scenario == "add-nested-advances":
        node.group.add_child(
            "new", ParameterNode("root.group.new", data={"x": {"2020-01-01": 1}})
        )
        assert node.parameter_revision.current == initial_revision + 1
    elif scenario.startswith("clone"):
        node.set_parameter_materializer(LazyParameterMaterializer())
        node("2026-01-01")
        cloned = node.clone()
        if scenario == "clone-revision-distinct":
            assert cloned.parameter_revision is not node.parameter_revision
        elif scenario == "clone-number-preserved":
            assert cloned.parameter_revision.current == node.parameter_revision.current
        elif scenario == "clone-strategy-preserved":
            assert cloned.parameter_materializer is node.parameter_materializer
        else:
            assert cloned._at_instant_cache is not node._at_instant_cache
            assert len(cloned._at_instant_cache) == 0
    elif scenario == "switch-to-lazy-advances":
        node.set_parameter_materializer(LazyParameterMaterializer())
        assert node.parameter_revision.current == initial_revision + 1
    elif scenario == "switch-to-eager-stales":
        node.set_parameter_materializer(LazyParameterMaterializer())
        view = node("2026-01-01")
        node.set_parameter_materializer(EagerParameterMaterializer())
        with pytest.raises(StaleParameterViewError):
            view.amount
    elif scenario == "children-share-context":
        assert node.group.parameter_revision is node.parameter_revision
        assert node.group.parameter_materializer is node.parameter_materializer
    elif scenario == "brackets-share-context":
        bracket = node.scale.brackets[0]
        assert bracket.parameter_revision is node.parameter_revision
        assert bracket.parameter_materializer is node.parameter_materializer
    elif scenario == "new-view-uses-update":
        node.set_parameter_materializer(LazyParameterMaterializer())
        node.amount.update(value=250, start=instant("2026-01-01"))
        assert node("2026-01-01").amount == 250
    elif scenario == "eager-view-keeps-snapshot":
        node.set_parameter_materializer(EagerParameterMaterializer())
        view = node("2026-01-01")
        node.amount.update(value=250, start=instant("2026-01-01"))
        assert view.amount == 200
    elif scenario == "clear-does-not-advance":
        node("2026-01-01")
        node.clear_at_instant_caches()
        assert node.parameter_revision.current == initial_revision
    elif scenario == "mapping-cache-uses-revision":
        view = node("2026-01-01")
        node._at_instant_cache = {instant("2026-01-01"): view}
        assert node._at_instant_cache.revision_source is node.parameter_revision
    elif scenario == "system-sets-materializer":
        system = CountryTaxBenefitSystem()
        system.set_parameter_materializer(LazyParameterMaterializer())
        assert type(system.get_parameters_at_instant("2026-01-01")) is (
            LazyParameterNodeAtInstant
        )
    elif scenario == "clear-before-tree-assignment":
        system = CountryTaxBenefitSystem.__new__(CountryTaxBenefitSystem)
        system._parameters_at_instant_cache = ParameterAtInstantCache()
        system._parameters_at_instant_cache.put("2026-01-01", object())
        system.clear_parameter_caches()
        assert len(system._parameters_at_instant_cache) == 0
    elif scenario == "replace-accepts-none":
        system = CountryTaxBenefitSystem()
        system.replace_parameters(None)
        assert system.parameters is None
        assert system.get_parameters_at_instant("2026-01-01") is None
    elif scenario == "replace-rejects-other-types":
        system = CountryTaxBenefitSystem()
        with pytest.raises(TypeError, match="ParameterNode or None"):
            system.replace_parameters(object())
    elif scenario == "replace-clears-dated-views":
        system = CountryTaxBenefitSystem()
        system.get_parameters_at_instant("2026-01-01")
        system.replace_parameters(system.parameters.clone())
        assert len(system._parameters_at_instant_cache) == 0
    elif scenario == "replace-binds-revision":
        system = CountryTaxBenefitSystem()
        replacement = system.parameters.clone()
        system.replace_parameters(replacement)
        assert (
            system._parameters_at_instant_cache.revision_source
            is replacement.parameter_revision
        )
    elif scenario == "replace-does-not-share-cache":
        first = CountryTaxBenefitSystem()
        second = CountryTaxBenefitSystem()
        first.get_parameters_at_instant("2026-01-01")
        second.replace_parameters(first.parameters)
        assert second._parameters_at_instant_cache is not (
            first._parameters_at_instant_cache
        )
    else:
        first = CountryTaxBenefitSystem()
        second = CountryTaxBenefitSystem()
        second.share_parameters_from(first)
        assert (
            second._parameters_at_instant_cache.revision_source
            is first.parameters.parameter_revision
        )


def test_parameter_materializer_is_an_abstract_generic_contract() -> None:
    with pytest.raises(TypeError):
        ParameterMaterializer()


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("shallow-type", id="shallow-type"),
        pytest.param("shallow-resolved-value", id="shallow-resolved-value"),
        pytest.param("deep-type", id="deep-type"),
        pytest.param("deep-independent-revision", id="deep-independent-revision"),
        pytest.param("pickle-type", id="pickle-type"),
        pytest.param("pickle-resolved-value", id="pickle-resolved-value"),
        pytest.param("stale-copy-rejected", id="stale-copy-rejected"),
        pytest.param("stale-pickle-rejected", id="stale-pickle-rejected"),
    ],
)
def test_lazy_view_copy_and_serialization_contract(scenario: str) -> None:
    node = make_parameter_tree()
    view = node("2026-01-01")
    assert view.amount == 200

    if scenario.startswith("stale"):
        node.amount.update(value=250, start=instant("2026-01-01"))
        with pytest.raises(StaleParameterViewError):
            if scenario == "stale-copy-rejected":
                copy.copy(view)
            else:
                pickle.dumps(view)
        return
    if scenario.startswith("shallow"):
        duplicate = copy.copy(view)
    elif scenario.startswith("deep"):
        duplicate = copy.deepcopy(view)
    else:
        duplicate = pickle.loads(pickle.dumps(view))

    if scenario.endswith("type"):
        assert type(duplicate) is LazyParameterNodeAtInstant
    elif scenario == "deep-independent-revision":
        duplicate_revision = duplicate._LazyParameterNodeAtInstant__revision
        duplicate_revision.advance()
        assert node.parameter_revision.current != duplicate_revision.current
    else:
        assert duplicate.amount == 200
