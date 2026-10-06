# Cache architecture

Use this skill before changing cache ownership, keys, values, invalidation,
branching, tracing, or lifecycle behavior in PolicyEngine Core.

## Design rules

PolicyEngine caches must state their owner, key, value, lifetime, mutation
rules, invalidation behavior, concurrency behavior, metrics, and uncached
reference behavior. Cache implementations use the generic contracts in
`policyengine_core.caching`; concrete implementations remain next to the
subsystem that owns their domain behavior.

The generic cache contracts provide lifecycle, locking, metrics, and common
operations. Concrete caches implement protected storage hooks and only the
capabilities that apply to them. Do not add a domain-specific exception to a
generic base class. Add a capability contract or keep the behavior in the
concrete cache instead.

Cache values must not retain a simulation, tracer, holder, or caller-owned
mutable object unless that ownership is part of the documented contract.
Failed validation or construction must leave a cache unchanged. `close()` is
idempotent, and operations after closure fail explicitly.

## Cache types

| Type | Owner and lifetime | Key and value | Isolation and invalidation |
| --- | --- | --- | --- |
| A: model definition | A tax-benefit system and Python's imported modules, normally process-lived | Variables, formulas, entities, and parameter sources | This remains combined with policy state until Phase 7. Do not expand its responsibilities in earlier phases. |
| B: policy system and reform | One YAML runner invocation and one pytest worker | An ordered, typed policy configuration to a constructed tax-benefit system | Retention is bounded. Cached systems contain no case simulation or trace state. Fully uncached execution constructs every system again. |
| C: parameter at date | A parameter node or tax-benefit system policy state | A policy revision and instant to an ordinary parameter-at-date view | Cached values contain no tracer. Eager materialization is the default. Lazy views bind to one revision and reject access after mutation. |
| D: simulation input and result | One simulation, including its branch-local index | Variable, period, and branch to an immutable array entry; it also owns supplied-input provenance | Input or policy mutation conservatively removes every derived result. Branches may share immutable entries but never indexes or writable arrays. |
| E: YAML case execution reuse | One YAML case execution | Not a key/value cache; it controls how a case borrows a Type B system | Every case receives a new simulation and releases it after success or failure. Results must equal fully uncached execution and must not depend on case order. |

Trace state is not a cache type. It belongs to the active calculation. A
parameter cache may provide an ordinary view to a short-lived tracing adapter,
but it must never retain that adapter.

`SimulationMacroCache` is a separate persistent feature. Its current identity
is not a general content identity for arbitrary policy results. Keep it
disabled or narrowly scoped unless dataset contents, policy, configuration,
extensions, schema, and serialization identity are complete. Redesigning it
is outside Phases 1 through 6.

## Generic behavioral contracts

`BaseCache[K, V]` owns locking, open/closed state, common operations, and an
immutable metrics snapshot. Capability abstractions add construction,
capacity, revision, or branch behavior without storing separate mutable
state:

- `FactoryBackedCache[K, V]` defines `get_or_create` and uncached operation.
- `BoundedCache[K, V]` defines capacity and deterministic eviction.
- `RevisionAwareCache[K, V, R]` defines revision validation.
- `BranchableCache[K, V]` defines independent indexes over immutable values.

Every concrete cache must pass the common contract tests and the tests for its
capabilities. The common requirements include deterministic keys, exact
metrics, idempotent clearing and closure, no partial mutation after an error,
independent metrics snapshots, defined copying and serialization, and safe
concurrent public operations.

## Type B requirements

- Reform and extension order is significant.
- Parameter overrides use a recursively frozen, explicitly typed key.
- The default retains one baseline plus at most two derived systems.
- Capacity zero retains only the baseline.
- Disabled mode retains nothing, including the baseline.
- Completed cases, tracers, holders, and simulations are never retained.
- Metrics report hits, misses, successful builds, writes, deletions,
  evictions, current entries, and peak entries.

## Type C requirements

- Cached values are ordinary parameter-at-date nodes.
- Tracing adapters are created for the current calculation and are not cached.
- Policy mutation clears eager cache entries and advances the policy revision.
- Deep copies receive independent revisions and caches unless parameter state
  is deliberately shared.
- Eager materialization remains the production default.
- A lazy view checks its captured revision on every observable access and
  raises `StaleParameterViewError` after source mutation.

## Type D requirements

- Supplied inputs and calculated values have explicit, separate provenance.
- All mutation paths use one conservative invalidation operation.
- Invalidation preserves supplied inputs and removes derived values from fast
  lookup, holder memory, and disk storage together.
- Arrays are copied on insertion and stored read-only.
- Reads return read-only arrays. Callers that need a mutable value use
  `.copy()` and replace the stored value through a supported simulation API.
- A branch receives an independent index that may reference the same immutable
  entries. Replacement or deletion in one branch cannot change another.

## Type E requirements

- A cached policy system is borrowed, not transferred to a YAML case.
- The case restores any prior system backreference.
- Cleanup runs after successful execution, construction failure, calculation
  failure, and assertion failure.
- Successful and failed case simulations and tracers are collectible.
- Cached and fully uncached execution produce equal values.
- Serial order and pytest worker count do not affect values.

## Test requirements

Every production-code commit in the Phase 1 through 6 cache work adds at least
40 new pytest unit cases. Count collected cases, not source functions. A
parameterized case counts only when it has a descriptive ID, a distinct setup,
and an assertion for a distinct behavior. Integration tests do not count
toward the minimum.

Each production commit records `Tests-added: N` in its commit message and is
independently green. New unit tests are deterministic, network-free, use small
synthetic fixtures, and contain no skips or expected failures used to satisfy
the count.

## Deferred work

Phase 7 separates immutable model definitions, policy definitions, and
simulation execution state. That work includes Type A and is intentionally not
part of Phases 1 through 6. Dependency-aware selective invalidation and a
content-addressed persistent macro-data cache are also separate projects.
