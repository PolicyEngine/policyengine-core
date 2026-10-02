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
variables, a value it read before the drop, so the records stay. A
calculation that was running then (one whose formula sets an input, say) may
have read the replaced value, so its result is kept neither in storage nor in
the macro cache; nor is any result calculated meanwhile in another simulation,
which may have called into this one (a parent formula calculating in a branch
whose formula calls back into the parent, say). The outermost calculation
running in the simulation whose input changed (a `calculate`, or a direct
`calculate_add`, whose terms run within it) then runs again from the new
inputs, inner calculations included, until a run changes no input, and keeps
that result, so uprating and carry-over find its period as they would had the
inputs come first. After ten reruns it stops: a formula that keeps changing
inputs returns its last result without keeping it, and a later uprating or
carry-over may then not find that period. The budget is per simulation whose
input changes, so calculations nested across several such simulations can
rerun more. An input stored for the very period being
calculated after the calculation began (by its own formula, say; under the
branch's name or any it reads, such as `default` through `Holder.set_input`)
is the result, as it would be had it been set first.

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

A value written into a holder's storage without going through `set_input` or a
calculation has no number; an input for a variable holding such a value drops
every calculated value.

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
and carry-over read which periods hold values at all, which is why the first
uprated or carried-over value also counts. A result whose calculation was
running when an input changed is not kept unless it was calculated again from
the new inputs.

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
  does), does not take in that simulation's history. Its own branches are
  covered: their history goes to every ancestor with a calculation running.
- **A branch a formula keeps between calls is a snapshot.** It holds what its
  parent held when it was created, so inputs set on the parent afterwards do
  not reach what the formula reads from it.
- **Inputs on the root simulation drop nothing.** `set_input` on a simulation
  that is not a branch keeps its earlier behaviour: values it already
  calculated stay.
- **Existing child branches keep their values.** An input set on a branch does
  not reach branches already created from it.

With disk storage (`MemoryConfig`), every store writes a new file, named with
the sequence number and a token for the process (a forked child gets its own),
so a value recalculated in one simulation does not change a file another
simulation, or another process, still maps; files stay until the storage
directory is removed. `OnDiskStorage.restore` takes each key's most recently
written file, and on a timestamp tie the current process's own; between two
other processes' files written within one clock tick it cannot tell which came
last.

A simulation dump (`dump_simulation`) holds the values the simulation reads
(on a branch, its own and those it inherited) and records which were inputs.
`restore_simulation` restores those as inputs and every other value as
calculated under one later number. The dump does not say what each value was
calculated from, nor what was read without being kept, so the restored
simulation records that values from that number on may depend on anything:
an input set for any variable on a branch of it drops all of them. A dump
written before inputs were recorded is restored with every value as an
input, as before, so such values never drop. With disk storage, a branch whose
name contains `_` cannot be dumped yet: `OnDiskStorage.get_known_periods`
splits its keys on every `_` (as on master).

Two related behaviours: a branch stops reading macro-cache files once an input
is set on it, since they are keyed by branch name and period but not by inputs
(a file for its name may have been written by this branch before the input, or
by another simulation's branch of the same name), and a branch that read one
drops, on any input, everything calculated from the first read on; and
`requires_computation_after` is satisfied by a prerequisite requested before a
drop removed its values.

## Dropping calculated values

`simulation.drop_computed_arrays()` deletes every value the simulation holds
except inputs (the dataset or situation it was built from, values set with
`set_input` on it, and, for a branch, values set on the simulations it was
created from before it was created), and returns how many arrays it deleted.
Values a custom `set_input` handler calculates are not inputs. Use it on a
branch after changing its policy:

```python
branch = simulation.get_branch("pre_reform_rules", clone_system=True)
branch.tax_benefit_system.parameters.gov.some_rate.update(period="2026", value=0.2)
branch.drop_computed_arrays()
```
