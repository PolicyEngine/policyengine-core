# Simulations

The `policyengine_core.simulations` module contains the definition of `Simulation`, the singular most important class in the repo. `Simulations` combine the logic of a country package with data, and can use the country logic and parameters to calculate the values of unknown variables. The class `SimulationBuilder` can create `Simulation`s from a variety of inputs: JSON descriptions, or dataset arrays.

## Simulation

```{eval-rst}
.. autoclass:: policyengine_core.simulations.simulation.Simulation
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
```

## Microsimulation

```{eval-rst}
.. autoclass:: policyengine_core.simulations.microsimulation.Microsimulation
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
```

## SimulationBuilder

Country models can override `TaxBenefitSystem.preprocess_situation(situation,
default_period)` to fill missing groups before an entity-form situation is
built. The hook receives a deep copy after singular entity aliases have been
expanded, and returns a situation for the builder's normal validation. Both
`Simulation(situation=...)` and the YAML runner use this path. Dataset and
variable-only construction do not call the hook. The default hook returns its
input unchanged.

`Simulation.input_group_entities` is an immutable set of singular group keys
supplied before preprocessing. For datasets, it identifies groups with person
membership columns, before loader defaults are inserted. An explicitly empty
group mapping counts as supplied. Synthesized groups and the singleton groups
created for variable-only input do not. Country formulas can use this provenance
to distinguish reported relationships from defaults without inspecting group
sizes or identifier values. Population-only construction does not assert input
provenance; simulation clones preserve their source's set.

```{eval-rst}
.. autoclass:: policyengine_core.simulations.simulation_builder.SimulationBuilder
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
```
