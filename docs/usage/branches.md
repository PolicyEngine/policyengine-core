# Branches

A branch is a copy of a simulation that calculates under different inputs or
policy without changing the simulation it came from. Formulas use branches to
compare alternatives (for example, tax liability if itemizing), and marginal
tax rates use them to recalculate with slightly higher earnings.

```python
branch = simulation.get_branch("itemizing")
branch.set_input("tax_unit_itemizes", 2026, itemizes)
tax_if_itemizing = branch.calculate("income_tax", 2026)
```

## What a branch starts with

The first `get_branch(name)` call returns a new branch holding every value its
parent holds at that moment: inputs and calculated values alike. Later calls
with the same name return that branch as it is. What the parent stores after
the branch is created does not reach the branch, and what the branch stores
does not reach the parent. A branch of a branch reads its own values first,
then its parent's, then those of the simulation at the root.

## Inputs set on a branch

`set_input` on a branch stores the new value and drops every value the branch
holds that may have been calculated from the value it replaces. What the branch
calculates next therefore uses the input, as a simulation given the input
before calculating anything would, whatever its parent had calculated before
the branch was created.

To decide what to drop, every stored value carries a sequence number from one
counter shared by the whole process, so a larger number means a later store.
Each simulation also keeps a store history: for each variable and period, the
number of the first store its values may have been calculated from, and for
each variable, the first time one of its values was uprated or carried over
from another period. The history records:

- the simulation's own stores, including values a holder calculates but does
  not keep (`variables_to_drop`, the cache blacklist), values read from the
  macro cache, and the default a spiral returns;
- the number from which values may have been calculated from anything: the
  first value read from the macro cache (what it was calculated from was
  never calculated here) and the values restored from a dump (see below), so
  an input for any variable drops them;
- a copy of its parent's history, taken when the branch is created;
- the history of any simulation its formulas calculate in, taken in each time
  `calculate` there returns or raises (a formula that branches, sets an input
  and calculates in the branch hands back values calculated there); a branch
  calculated from a thread with no formula context hands its history to every
  ancestor with a calculation running.

When `set_input(variable, period, value)` is called on a branch, the branch
drops each value it holds, other than an input, whose number is at least the
earliest recorded store of the variable for a period that shares a day with
`period`, or the earliest recorded uprated or carried-over value of the
variable. It then forgets the records numbered from there on (its remaining
values were all stored earlier) and records again the inputs it keeps, unless a
formula is still running in the branch: such a formula may hold, in its own
variables, a value it read before the drop, so the records stay. Each drop
counts as an input change of the branch. A calculation running in the branch
then (one whose formula sets an input, say) may have read the replaced value,
so its result is kept neither in storage nor in the macro cache. Neither is any
result calculated from a value the branch gave before the change, in any
simulation: each calculation notes, for every other simulation it got a value
from (directly or through the calculations it called), that simulation's count
of input changes when the value's calculation began, and keeps its own result
only if none has changed since. (A calculation nested in another in the same
simulation shares that one's record, which can only make it keep less.) So a
parent formula calculating in a branch whose formula calls back into the parent
and then changes the branch's input does not keep what it got, nor does a
formula that read a branch and then calculated there something that changed the
branch's input, or set an input on it directly. A value calculated in a thread
started without the formula's context counts too: with no calculation above it
in that thread, every calculation running in the simulation's ancestors notes
it (`asyncio.to_thread` and `contextvars.copy_context` keep the context, so the
formula notes it directly). A calculation whose call into another simulation
settled before returning keeps its result, and unrelated simulations keep
caching. The outermost calculation running in each simulation that does not
keep its result (a `calculate`, or a direct `calculate_add`, whose terms run
within it) then runs again, inner calculations included, until a run reads
nothing that has changed since, and keeps that result, as a simulation given
the new inputs first would. After ten reruns it stops: a formula that keeps
changing inputs returns its last result without keeping it. The budget is per
simulation, so calculations nested across several such simulations can rerun
more. An input set again with the value the branch already reads for that very
period, as an input (not through a `set_input` helper that spreads it over
other periods, and not `-0.0` for `0.0`), changes nothing, so it drops nothing
and is no input change. An input stored for the very period being calculated
after the calculation began (by its own formula, say; under the branch's name
or any it reads, such as `default` through `Holder.set_input`) is the result,
as it would be had it been set first.

A custom `set_input` handler that calculates values between its own stores
calculates them from inputs it has not yet replaced, so if it calculated
anything (through `calculate`, `calculate_add` or `_calculate`), the branch
drops again, by the same rule, once the handler returns or raises (the inputs
it stored before raising stay).

If there is no such record, the branch drops nothing. That is the case when a
formula creates the branch while it is still calculating the variable the
branch overrides, unless another branch the formula created for the same
comparison already handed back a value calculated from that variable: then the
branch drops what came back and what was calculated after it.

A value written into a holder's storage directly, not through the storage's
`put`, has no number; an input for a variable holding such a value drops every
calculated value.

### Why this is sound

A formula's result is stored after everything the formula read was stored or
recorded, so a calculated value carries a larger number than each value it was
calculated from, directly or through other calculated values. Every value a
simulation holds was calculated there, inherited from its parent, handed back
by another simulation's `calculate`, or restored from a dump. In the first
three cases, for every variable it was calculated from, the simulation's
history holds a record of that variable for an overlapping period numbered no
later than the value read (for a value summed or divided from other periods,
the record may be of those periods), unless it was calculated from a
macro-cache read; restored values and values calculated from a macro-cache
read are covered by the record that values from a number on may depend on
anything. So any value that depends on the overridden variable at an
overlapping period was stored after the earliest record that applies. Uprating
and carry-over read which periods hold inputs, so a value they give can change
with an input for any period of the variable, which is why the first uprated
or carried-over value also counts. A result whose calculation was
running when an input changed is not kept unless it was calculated again from
the new inputs. An input set again with the value the branch
already reads for that period, as an input, changes no value read anywhere, so
dropping nothing then is sound; one set where the value was calculated still
drops, since it changes which periods hold inputs.

The rule can drop more than it needs to (a value stored later that does not
depend on the input is calculated again) but not less, within these limits:

- **Policy changes are not tracked.** A branch whose tax-benefit system or
  parameters differ from its parent's still holds the parent's values
  calculated under the parent's policy. Call `branch.drop_computed_arrays()`
  after changing them.
- **Formulas must not write into arrays they read.** A formula that changes a
  cached array in place (`array += x`) changes a value after its sequence
  number was assigned, so values calculated from it may be dropped too late or
  not at all.
- **Formulas should calculate, not inspect storage.** A formula that reads
  `holder.get_known_periods()`, `simulation.get_array()` or another
  simulation's storage directly, and acts on what it finds, depends on values
  that are not recorded.
- **Unrelated simulations in other threads.** A formula that calculates in a
  simulation other than its own branches, from a thread it starts without
  copying its context (`contextvars.copy_context`, which `asyncio.to_thread`
  does), does not take in that simulation's history, nor note what it read
  there. Its own branches are covered: their history, and what was read in
  them, go to every ancestor with a calculation running.
- **A branch a formula keeps between calls is a snapshot.** It holds what its
  parent held when it was created, so inputs set on the parent afterwards do
  not reach what the formula reads from it.
- **Inputs on the root simulation drop nothing.** `set_input` on a simulation
  that is not a branch keeps its earlier behaviour: values it already
  calculated stay.
- **Existing child branches keep their values.** An input set on a branch does
  not reach branches already created from it.
- **Values kept elsewhere stay.** A value one simulation calculated from
  another's (its branch's, its parent's, or another family's) and kept stays
  when that other simulation's input changes after the calculation ended: only
  calculations still running then are not kept, and run again. A rerun reads
  such a kept value as it is. Variables in `cache_blacklist` or
  `variables_to_drop` are still kept in the simulation's fast cache, as
  before, so this applies to them too.
- **Formulas that create a branch each run.** A formula that creates a branch,
  reads it, and then changes its input runs again until the budget is spent
  (each run reads a new branch from before the change), then returns its last
  result without keeping it.

With disk storage (`MemoryConfig`), a value dropped from a branch only leaves
its storage's index: the file stays until the storage directory is removed,
since a clone of the storage may still read it, and storing the key again
writes a file of the branch's own (see `OnDiskStorage._path_to_write`).

A simulation dump (`dump_simulation`) holds the values the simulation reads
(on a branch, its own and those it inherited) and records which were
calculated (`derived_periods.txt`). `restore_simulation` restores the others
as inputs and every calculated value under one later number. The dump does not
say what each value was calculated from, nor what was read without being kept,
so the restored simulation records that values from that number on may depend
on anything: an input set for any variable on a branch of it drops all of them.
A dump written before calculated values were recorded is restored with every
value as an input, as before, so such values never drop. With disk storage, a branch whose
name contains `_` cannot be dumped yet: `OnDiskStorage.get_known_periods`
splits its keys on every `_` (as on master).

Two related behaviours: a branch stops reading macro-cache files once an input
is set on it, even one it already reads, since they are keyed by branch name and
period but not by inputs (a file for its name may have been written by this
branch before the input, or by another simulation's branch of the same name),
and a branch that read one drops, on any input that changes what it reads,
everything calculated from the first read on; and
`requires_computation_after` is satisfied by a prerequisite requested before a
drop removed its values.

## Dropping calculated values

`simulation.drop_computed_arrays()` deletes every value the simulation holds
except inputs (the dataset or situation it was built from, values set with
`set_input` on it, and, for a branch, values set on the simulations it was
created from before it was created; also values cached with
`Holder.put_in_cache` without `derived=True`), and returns how many arrays it
deleted. Values a custom `set_input` handler calculates are not inputs. Use it on a
branch after changing its policy:

```python
branch = simulation.get_branch("pre_reform_rules", clone_system=True)
branch.tax_benefit_system.parameters.gov.some_rate.update(period="2026", value=0.2)
branch.drop_computed_arrays()
```
