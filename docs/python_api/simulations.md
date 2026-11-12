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

```{eval-rst}
.. autoclass:: policyengine_core.simulations.simulation_builder.SimulationBuilder
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
```
