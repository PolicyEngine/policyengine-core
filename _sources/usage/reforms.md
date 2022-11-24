# Writing reforms

In PolicyEngine country packages, reforms are defined as Python objects that specify a way to modify a tax-benefit system. This can include changes to the parameter tree, or to variables. Generally, here's how to write a reform:

```python
from policyengine_core.reforms import Reform

class reform(Reform): # Inherit from the Reform class
    def apply(self): # You must define an apply method
        self.modify_parameters(modify_parameters)
        self.add_variable(new_variable_class)
        self.update_variable(overriding_variable_class)
        self.neutralize_variable(variable_name)
```

## Modifying parameters

To modify parameters, you should define `modify_parameters` method. This should be a function of a `ParameterNode` and should return the same `ParameterNode` (but modified). For example:

```python
def modify_parameters(parameters):
    parameters.benefits.child_benefit.amount.update(period="2019-01", value=100)
    return parameters
```

For most cases, you'll be able to do everything you need with the `update` method. This takes a `period` and a `value` and updates the parameter to that value for that period. You can also optionally specify a `start` and `stop` instant to specify the period over which the change should be applied.

## Modifying variables

To add a variable, define it in the same way the country package does. For example:

```python
class new_variable(Variable):
    value_type = float
    entity = Person
    label = "New variable"
    definition_period = YEAR
```

The `add_variable` method takes this class as its argument. It'll throw an error if the variable already exists. If you want to override an existing variable, use `update_variable` instead.

To neutralize a variable, use the `neutralize_variable` method. This takes the name of the variable as its argument, and ensures that this variable returns zero (or the default value) for all periods (essentially, removes all formulas).
