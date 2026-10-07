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
idempotent. Mutations and value lookups after closure fail explicitly; metrics
and documented metadata observations remain inspectable.

## Cache types

| Type | Owner and lifetime | Key and value | Isolation and invalidation |
| --- | --- | --- | --- |
| A: model definition | A tax-benefit system and Python's imported modules, normally process-lived | Variables, formulas, entities, and parameter sources | This remains combined with policy state until Phase 7. Do not expand its responsibilities in earlier phases. |
| B: policy system and reform | One YAML runner invocation and one pytest worker | An ordered, typed policy configuration to a constructed tax-benefit system | Retention is bounded. Cached systems contain no case simulation or trace state. Fully uncached execution constructs every system again. |
| C: parameter at date | A parameter node or tax-benefit system policy state | A policy revision and instant to an ordinary parameter-at-date view | Cached values contain no tracer. Lazy materialization is the default. Lazy views bind to one revision and reject access after mutation; callers may explicitly select eager snapshots. |
| D: simulation input and result | One simulation, including its branch-local index | Variable, period, and branch to an immutable array entry; it also owns supplied-input provenance | Input or policy mutation conservatively removes every derived result. Branches may share immutable entries but never indexes or writable arrays. |
| E: YAML case execution reuse | One YAML case execution | Not a key/value cache; it controls how a case borrows a Type B system | Every case receives a new simulation and releases it after success or failure. Results must equal fully uncached execution and must not depend on case order. |
| F: persistent macro results | Per-call helper over dataset-folder files that outlive a simulation | Dataset name, variable, period, and branch filename; arrays plus Core/country version metadata | Existing identity is incomplete for arbitrary policies or changed datasets. Redesign and general reuse are deferred beyond Phases 1–6. |

Trace state is not a cache type. It belongs to the active calculation. A
parameter cache may provide an ordinary view to a short-lived tracing adapter,
but it must never retain that adapter.

Type F (`SimulationMacroCache`) is a separate persistent feature. Its current identity
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

Typed operations already provide their complete lifecycle guarantees in the
preparatory release. Bulk replacement holds the owner lock, validates every
converted key and value before changing state, counts each installed entry as
a write and only absent old keys as deletions, and updates peak size. Metadata
replacement does not count as a result-entry write. Metadata mutations and
input-context entry reject closed owners; context cleanup restores its stack
even if the owner closes inside it. Queries use the same lock.

Preparatory dictionary/set/list observations and raw-value/tuple adapters remain
available solely for old country compatibility. Their return values are still
live mutable objects; callers must use typed methods to obtain the guarantees
above. Final cleanup removes those escape paths without changing the guarantees.

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
- Policy mutation clears dated-view cache entries and advances the policy revision.
- Deep copies receive independent revisions and caches unless parameter state
  is deliberately shared.
- Lazy materialization is the production default; eager construction remains
  an explicit compatibility and profiling option.
- A lazy view checks its captured revision on every observable access and
  raises `StaleParameterViewError` after source mutation.

`ParameterTreeRevision` is the shared monotonic revision for one parameter
tree. Every descendant node and every `ParameterAtInstantCache` attached to
that tree uses the same revision object. Cache entries have the internal
identity `(revision, instant)`. `Parameter.update()` and structural
`ParameterNode.add_child()` mutations advance the root revision; ordinary
cache clearing does not.

`ParameterMaterializer[SourceT, InstantT, ViewT]` is the generic construction
strategy contract. `LazyParameterMaterializer` is the default. It returns
`LazyParameterNodeAtInstant`, which resolves one immediate child at a time,
supports traced and vector lookups, and checks its source revision before
every observable access. Use `EagerParameterMaterializer` explicitly through
`ParameterNode.set_parameter_materializer()` or
`TaxBenefitSystem.set_parameter_materializer()` when a complete dated
snapshot is required. Changing strategies invalidates existing dated views by
advancing the revision. Do not change construction strategy by modifying
private cache fields.

An eager view already returned to a caller remains a snapshot of its prior
policy values. A lazy view instead fails after mutation because continuing to
resolve children could otherwise combine values from different revisions.
Tax-benefit systems use `replace_parameters()` to install a tree with a fresh
dated-view cache. Systems that deliberately use `share_parameters_from()`
instead share the tree, revision, and dated-view cache; clones receive
independent revisions and caches.

## Type D requirements

- Supplied inputs are authoritative simulation state, not evictable results.
  Each memory or disk storage owns its supplied-input metadata alongside its
  immutable values. The simulation owns a separate query index and coordinates
  updates through `Holder.set_input`, `Holder.delete_arrays`, and its public APIs.
  A value inserted only through `put_in_cache` is not a supplied input, even
  when its legacy `derived` flag is false.
- The supplied-input key is `(variable_name, branch_name, Period)`. Storage
  canonicalizes eternity variables to `ETERNITY`. The result lookup key is
  `ResultCacheKey(variable_name, Period)` within one simulation owner. Pending
  invalidations use that same result key and are copied independently on clone.
- All supported input mutation paths invalidate that simulation's results.
  Invalidation removes non-supplied values from fast lookup, holder memory, and
  disk indexes together, without reading, copying, or replaying supplied arrays.
  Input loading does not scan previously loaded inputs after each insertion.
- Branches are snapshots of input state. Clearing, replacing, pruning, or
  deleting a parent's inputs or results never mutates an existing branch.
  Linked scenarios must be updated by explicitly invoking the public mutation
  API on each intended simulation; containment in `branches` is not consent.
- Deliberately shared policy trees remain shared. A simulation verifies policy
  identity and revision before result reuse and invalidates its own results
  when either changes. Shared input state and shared policy state are distinct.
  Supported variable registration, replacement, update, neutralization, and
  annualization also advance the owning system's revision. Country code that
  deliberately shares a raw variables dictionary between distinct systems must
  explicitly rebind and invalidate the affected simulations after changing it.
  Identity observation reads the Core-owned dated cache's bound revision, not
  a country's parameter accessor, which may trigger copy-on-write. Shallow
  wrappers sharing the same registry, tree revision, and variable revision
  have the same policy identity; a wrapper object alone is not a policy change.
- Arrays are copied on insertion and stored read-only.
- Reads return read-only arrays. Callers that need a mutable value use
  `.copy()` and replace the stored value through a supported simulation API.
- A branch receives an independent index that may reference the same immutable
  entries. Replacement or deletion in one branch cannot change another.
- `retain_supplied_inputs(variable_names)` removes all calculated values and
  unnamed supplied inputs in the receiving simulation only.
- `Simulation.input_revision` is a monotonic, owner-local integer advanced by
  successful supplied writes and actual deletions/pruning. Clones copy the
  revision and advance independently. Rejected/ignored inputs and calculated
  writes or result clearing do not advance it. A country helper reusing a
  derived comparison branch must compare the source input revision and rebuild
  its cached comparison when it changes; ordinary branches remain snapshots.
- `Holder.put_in_cache` cannot overwrite a supplied input, including when its
  legacy `derived=False` argument is used. Replacement requires `set_input`.
  `SimulationResultCache.clear()` clears result lookup entries only, not inputs;
  simulation-level invalidation coordinates holder storage and pending work.
- `rebind_tax_benefit_system()` binds populations and holders to the currently
  installed system and removes provenance for removed or incompatible variables.
  If that policy differs from the results' recorded policy, rebinding discards
  calculated values before accepting the new revision. Core cloning first
  synchronizes the source policy, then preserves only valid copied results.
  A clone also rebinds bound-method aliases to the clone rather than its source.
  Its `set_simulation_backreference=True` option is for a system privately
  owned by that simulation; the default preserves a shared system's owner.

## Release sequence

The migration has three stages and four pull requests: preparatory Core, US
and UK migrations, then final Core compatibility removal. Preparatory Core
provides the complete ownership and public-API guarantees; country migrations
remove direct cache mutation and correctness workarounds in that stage, not
after final cleanup. Country releases must require the actual published Core
version providing these APIs. Final cleanup waits for both country releases.

Compatibility mappings, old private attribute setters, and recovery of old
initialization layouts exist only during this transition. Core and migrated
country production code must not use those adapters. Removing them does not
change the public ownership guarantees or introduce Phase 7 model separation.

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
