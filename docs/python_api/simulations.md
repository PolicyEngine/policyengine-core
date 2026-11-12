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
built. `SimulationBuilder.build_from_entities` passes it a deep copy of the
situation, with any `axes` still present, and it must return a dict for the
builder's normal validation. `default_period` is the builder's default period
as set, so it may be `None`, a string, an integer or a `Period`.
`Simulation(situation=...)` and the YAML runner reach it through
`build_from_dict`, which expands singular entity aliases first; a direct
`build_from_entities` call passes the caller's keys as given. The hook runs
before the builder validates entities, so it may receive a situation with no
persons. A `Reform` runs its country system's hook with the reform as the
system, so the hook sees reformed parameters and variables. Dataset and
variable-only construction do not call the hook. The default hook returns its
input unchanged.

```{eval-rst}
.. autoclass:: policyengine_core.simulations.simulation_builder.SimulationBuilder
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
```
