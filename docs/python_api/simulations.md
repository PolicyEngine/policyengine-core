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
expanded, with any `axes` still present, and must return a dict for the
builder's normal validation. `default_period` is the builder's default period
as set, so it may be `None`, a string, an integer or a `Period`. Both
`Simulation(situation=...)` and the YAML runner use this path, and a `Reform`
delegates the hook to its baseline. Dataset and variable-only construction do
not call the hook. The default hook returns its input unchanged.

`Simulation.input_group_entities` is an immutable set of singular group keys
supplied before preprocessing. An explicitly empty group mapping counts as
supplied. Synthesized groups, the singleton groups created for variable-only
input, and the placeholder and default groups `IndividualSim` adds do not. For
datasets, it records groups declared by person membership columns or, in flat
files, by the group's ID column on each person row. Core's loaders require
membership for every group, so a loaded dataset records all of them; a country
that overrides `Simulation.build_from_dataset` must set the attribute itself.
Country formulas can use this provenance to distinguish reported relationships
from defaults without inspecting group sizes or identifier values.
Population-only construction does not assert input provenance; simulation
clones preserve their source's set.

```{eval-rst}
.. autoclass:: policyengine_core.simulations.simulation_builder.SimulationBuilder
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
```
