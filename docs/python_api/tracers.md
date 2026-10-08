# Tracers

The `policyengine_core.tracers` module contains classes used to represent tracers, which are used to track the computation involved in calculating variables.

## ComputationLog

```{eval-rst}
.. autoclass:: policyengine_core.tracers.computation_log.ComputationLog
    :members:
    :inherited-members:
    :show-inheritance:
```

## FlatTrace

Flat traces retain the first completed calculation's dependencies and parameters
when a later request reads its cache. When the period-recursion scheduler retries
a calculation, a completed accepted result replaces an abandoned or provisional
cut result for the same variable, period, and branch. The full computation and
performance logs still contain the retry attempts.

```{eval-rst}
.. autoclass:: policyengine_core.tracers.flat_trace.FlatTrace
    :members:
    :inherited-members:
    :show-inheritance:
```

## FullTracer

```{eval-rst}
.. autoclass:: policyengine_core.tracers.full_tracer.FullTracer
    :members:
    :inherited-members:
    :show-inheritance:
```

## PerformanceLog

```{eval-rst}
.. autoclass:: policyengine_core.tracers.performance_log.PerformanceLog
    :members:
    :inherited-members:
    :show-inheritance:
```

## SimpleTracer

```{eval-rst}
.. autoclass:: policyengine_core.tracers.simple_tracer.SimpleTracer
    :members:
    :inherited-members:
    :show-inheritance:
```

## TraceNode

```{eval-rst}
.. autoclass:: policyengine_core.tracers.trace_node.TraceNode
    :members:
    :inherited-members:
    :show-inheritance:
```

## TracingParameterNode

```{eval-rst}
.. autoclass:: policyengine_core.tracers.tracing_parameter_node_at_instant.TracingParameterNode
    :members:
    :inherited-members:
    :show-inheritance:
```

## TracingParameterNodeAtInstant

```{eval-rst}
.. autoclass:: policyengine_core.tracers.tracing_parameter_node_at_instant.TracingParameterNodeAtInstant
    :members:
    :inherited-members:
    :show-inheritance:
```
