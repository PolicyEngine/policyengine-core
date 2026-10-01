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

`get_branch` returns a branch holding every value its parent holds at that
moment: inputs and calculated values alike. What the parent stores after that
does not reach the branch, and what the branch stores does not reach the
parent. A branch of a branch reads its own values first, then its parent's,
then those of the simulation at the root.

## Inputs set on a branch

`set_input` on a branch stores the new value and drops every value the branch
holds that may have been calculated from the value it replaces. What the branch
calculates next therefore uses the input, as a simulation given the input
before calculating anything would, whatever its parent had calculated before
the branch was created.

To decide what to drop, every stored value carries a sequence number from one
counter shared by the whole process, so a larger number means a later store.
A simulation and all its branches also share a record of the first time each
variable was stored (or calculated without being kept) for each period, and of
the first time one of its values was uprated or carried over from another
period. When `set_input(variable, period, value)` is called on a branch, the
branch drops each value it holds, other than an input, whose number is at
least the earliest of:

- the first time the variable was stored for any period that shares a day with
  `period`, anywhere in the family; and
- the first time a value of the variable was uprated or carried over.

If neither has happened, the branch drops nothing. That is the usual case: a
formula creates the branch while it is still calculating the variable the
branch overrides, so nothing has stored that variable yet.

### Why this is sound

A formula's result is stored after everything the formula read has been
stored, so a calculated value carries a larger number than each stored value it
was calculated from, directly or through other calculated values. A value that
depends on the overridden variable at an overlapping period was therefore
stored after the first such value existed. A value derived from another period
of the same variable (uprating, carry-over) depends on which periods hold
values at all, which is why the first such derivation also counts. The record
covers the whole family because a formula can calculate in one branch and
return the result to another (`branch.calculate` inside a formula), and
because a branch may be deleted after its results were stored elsewhere.

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
- **Formulas should calculate, not test what is stored.** A formula that reads
  `holder.get_known_periods()` or `simulation.get_array()` and acts on whether
  a value exists depends on that value without its number being recorded.
- **Only the family is tracked.** A value calculated in a separate simulation
  (one not created with `get_branch`) and passed in is not covered.
- **Inputs on the root simulation drop nothing.** `set_input` on a simulation
  that is not a branch keeps its earlier behaviour: values it already
  calculated stay.
- **Existing child branches keep their values.** An input set on a branch does
  not reach branches already created from it.

## Dropping calculated values

`simulation.drop_computed_arrays()` deletes every value the simulation holds
except inputs (the dataset or situation it was built from, and values set with
`set_input` on it or on the simulations it was created from), and returns how
many arrays it deleted. Use it on a branch after changing its policy:

```python
branch = simulation.get_branch("pre_reform_rules", clone_system=True)
branch.tax_benefit_system.parameters.gov.some_rate.update(period="2026", value=0.2)
branch.drop_computed_arrays()
```
