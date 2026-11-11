# Writing parameters

**Parameters** are values that can change over time (but are the same for an entire country package, i.e. they can't be different for different entities). The `parameters/` folder of a country package contains the **parameter tree**, a tree of parameter nodes (objects that hold parameters as children) containing parameters or subnodes. A parameter node can be represented either as a folder, or a YAML file. Parameters are stored in parameter files. For example, a parameter tree might have the following structure in a country repo:

```{eval-rst}
.. code:: none

   parameters/
   ├── child_benefit/
   │   ├── basic.yaml
   │   ├── child.yaml
   |   ├── family.yaml
   │   └── index.yaml # Contains metadata for the child_benefit node.
   ├── income_tax/
   │   ├── basic.yaml
   │   ├── personal.yaml
   │   └── thresholds.yaml
   ├── national_insurance/
   │   ├── basic.yaml
   │   ├── employee.yaml
   │   └── thresholds.yaml
   └── universal_credit/
       ├── basic.yaml
       ├── child.yaml
       └── family.yaml
```

## Parameter values

Parameters are defined as a function of time, and are evaluated at a given instant. To achieve this, parameters must be defined with a **value history**: a set of time-dated values (where the value at a given instant is the last-set value before that instant). For example, the `child_benefit.basic.amount` parameter might have the following value history:

```yaml
values:
   2019-04-01: 20.00
   2019-04-02: 21.00
   2019-04-03: 22.00
   2019-04-04: 23.00
   2019-04-05: 24.00

```

## Metadata

Each parameter (or parameter node) can also set **metadata**: data that describes the parameter, such as its name, description, and units. While the metadata for each parameter and node is freeform, using the schemas defined in `policyengine_core.data_structures` will ensure consistency between country packages, and better maintainability.

Here's an example metadata specification for the `child_benefit.basic.amount` parameter:

```yaml
description: The UK provides this Child Benefit per child.

metadata:
    label: Child Benefit
    period: year
    unit: currency-GBP
    reference: 
        - label: GOV.UK | Child Benefit
          href: https://www.gov.uk/government/publications/child-benefit-rates-and-thresholds/child-benefit-rates-and-thresholds
```

### Metadata in copies

`clone()` on a parameter tree, which `TaxBenefitSystem.clone()`, reforms and simulation clones use, gives every node, parameter, dated value and scale in the copy its own metadata dict. The values in that dict are the same objects as in the source. Setting, updating or removing a key changes only the tree it is made on. To change a nested value, such as one entry of a `reference` list, replace the key with a new value instead of editing the list or dict in place, which would change every copy that shares it:

```python
parameter.metadata["reference"] = [*parameter.metadata["reference"], new_reference]
```

The same holds within one tree: a dated value written in dictionary form (`value:`, with or without `metadata:`) holds its parameter's metadata values, and `propagate_parameter_metadata` gives a node's descendants the node's values.

## Uprating

A parameter's `uprating` metadata extends its value history with another parameter, the index, such as a price index. When a tax-benefit system loads its parameters, `uprate_parameters` adds a value at each instant where the index has a value after the parameter's last explicit value (or, in chained mode, after `start_instant` when the metadata sets one). Uprating never changes an explicit value, so a new explicit value always replaces a projection.

### Chained mode

By default, uprating scales the last explicit value, at instant `L`, by the index's change since then:

```none
value(t) = value(L) × index(t) / index(L)
```

```yaml
metadata:
  uprating:
    parameter: gov.example.price_index
    rounding:
      type: downwards # or nearest, or upwards
      interval: 50
```

`rounding` rounds the uprated amount, as a floating-point number, to a multiple of `interval`; `nearest` takes an exact half to the even multiple. A bare float, such as `rounding: 0.01`, rounds to the nearest multiple. Chained mode also takes `start_instant`, to project from an instant other than the last explicit value; `at_defined_interval`, to uprate on a fixed cadence; and `parameter: self`, to extend the parameter's own growth.

### Base mode

Some statutes index a fixed base amount and round the increase rather than the increased amount. 26 U.S.C. 63(c)(7)(B)(ii), for example, increases the $23,625 head-of-household standard deduction by the cost-of-living adjustment since 2024 and rounds that increase down to a multiple of $50. A `base` block opts a parameter into this rule:

```yaml
metadata:
  uprating:
    parameter: gov.example.price_index
    base:
      period: 2025
      value: 23_625
      floor: true
    rounding:
      type: downwards
      interval: 50
```

For each instant `t` where the index has a value after the last explicit value, and no earlier than `base.period`, base mode computes

```none
value(t) = base + round(base × (index(t) / index(base.period) − 1))
```

applying `rounding`'s type and interval to the increase. A bare number, such as `rounding: 10`, rounds to the nearest multiple. Without `rounding`, `value(t) = base × index(t) / index(base.period)`.

- `base.period` is a year, such as `2025`, or a date, such as `1997-01-01`. It names the base in the index's own periods, since base mode reads the index there. Where the index's value at period `Y` measures calendar year `Y − 1`, as in the example above, a statute's base year `Y − 1` is `period: Y`.
- `base.value` defaults to the parameter's own value at `base.period`. When the parameter has a value there, a stated `base.value` must match it to one part in a billion, which catches a period one year off. State it when the parameter has no value at the base period, such as a 1997 base for a parameter whose history starts later; elsewhere, leaving it out lets the base follow a reform (see below).
- `base.floor: true` counts the index as no lower than its base-period level: base mode uses `max(index(t) / index(base.period), 1)` as the ratio, so a positive base never projects below itself. Statutes that add "the percentage (if any) by which" an index exceeds its base-year level floor the adjustment this way; 26 U.S.C. 1(f)(3), which sets the cost-of-living adjustment in the example above, is one. Without `floor`, an index below its base-period level gives a value below the base.
- Base mode takes only `parameter`, `base` and `rounding`. It raises an error for `start_instant`, `at_defined_interval`, `parameter: self` and any other key, rather than ignore them.
- Base mode computes the increase exactly, in rational arithmetic on the decimals the base, the index values and the interval print as, so an index written as `102.6` counts as 102.6. A 40% rise on $1,400 is exactly $560, though floats compute it as 559.9999999999998, so rounding down to $10 keeps all of it. `nearest` takes an exact half up: a $25 increase on a $10 interval becomes $30.
- An index computed in floating point, such as an average of monthly prices, prints with more than 15 significant digits and can be off by a few units in its last place. When any float input (the base, either index value or the interval) prints that way, base mode allows for it: an increase within 64 machine epsilons (about 1.4 × 10⁻¹⁴) of the uprated amount of a multiple of the interval, or of a half for `nearest`, counts as on it, up to 10⁻⁶ intervals. When every input is an int or a float that prints with 15 significant digits or fewer, as a value written in YAML with that many digits does, base mode allows nothing, and the arithmetic is exact.
- An infinite base, such as a scale's `-.inf` lowest threshold, stays infinite.

### Base mode and reforms

Base mode's projections depend on the explicit values only through the base amount, not on the values just before them. That matters where reforms apply before uprating, as in a `TaxBenefitSystem` that loads its parameters and receives a `reform`: it applies the reform, then uprates. Take the head-of-household example and a reform that sets $30,000 from some date on:

- From 2026 on, the reform sets 2026 to $30,000, and the base is still $23,625, so 2027 follows the statute again. Chained mode would instead project 2027 from $30,000.
- From 2020 on, the reform also sets the base period's value. With `base.value` left out, $30,000 becomes the base and the projections index it. A stated `base.value` of $23,625 no longer matches, and uprating raises an error that says to leave it out.

A reform that should hold its value in later years can give it a range covering them, such as `2026-01-01.2100-12-31`. Country packages that apply reforms after uprating, as PolicyEngine US does, see projected values as before: a reform replaces them from its start.

### How the modes differ

Without rounding, the modes agree up to floating-point error whenever the explicit values follow the index from the base: if `value(L) = base × index(L) / index(base)`, chained mode's `value(L) × index(t) / index(L)` equals base mode's `base × index(t) / index(base)`.

With rounding, they round different quantities. Chained mode rounds the whole amount, so every value it projects is a multiple of the interval. Base mode rounds the increase, so its values are the base plus a multiple of the interval, and a base off the interval's grid stays off it. Rev. Proc. 2026-35 prints two 2027 amounts of this kind, which chained mode cannot produce:

- Section 3.15 sets the head-of-household standard deduction at $24,925: the $23,625 base plus $1,300, a multiple of $50. $24,925 is not.
- Section 3.29 sets the long-term care premium limit for ages 41 to 50 at $945: the $375 base in 26 U.S.C. 213(d)(10)(A) plus $570, an increase that section 213(d)(10)(B)(i) rounds to the nearest $10. $945 is not a multiple of $10.

Chained mode also carries rounding forward when it starts from a published, rounded value. Take a $2,000 base, an increase rounded down to $100, and an index 4.9% above its base level in year 1 and 10% above it in year 2. The year-1 amount is $2,000, since $98 rounds down to $0. Base mode gives $2,200 for year 2. Chained mode from the year-1 amount gives $2,000 × 1.10 / 1.049 = $2,097.24, which rounds down to $2,000.

### Metadata for parameters

```{eval-rst}
.. autoclass:: policyengine_core.data_structures.ParameterMetadata
    :members:
```

### Metadata for parameter nodes

```{eval-rst}
.. autoclass:: policyengine_core.data_structures.ParameterNodeMetadata
    :members:
```

## Specifying references

```{eval-rst}
.. autoclass:: policyengine_core.data_structures.Reference
    :members:
```

## Units

```{eval-rst}
.. autoclass:: policyengine_core.data_structures.Unit
    :members:
```

## Other specifications

```{eval-rst}
.. automodule:: policyengine_core.data_structures.parameter_metadata
    :members:
    :show-inheritance:
    :exclude-members: ParameterMetadata, Reference, Unit

.. automodule:: policyengine_core.data_structures.parameter_node_metadata
    :members:
    :show-inheritance:
    :exclude-members: ParameterNodeMetadata
```
