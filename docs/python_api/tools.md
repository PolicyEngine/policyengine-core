# Tools

The `policyengine_core.tools` module contains miscellaneous utility functions, including for running YAML tests.

## run_tests

```{eval-rst}
.. autofunction:: policyengine_core.tools.test_runner.run_tests
```

The same runner backs `policyengine-core test <paths> -c <country_package>`,
where `--reform-cache-size N` sets `reform_cache_size`.

## dump_simulation

```{eval-rst}
.. autofunction:: policyengine_core.tools.simulation_dumper.dump_simulation
```

## restore_simulation

```{eval-rst}
.. autofunction:: policyengine_core.tools.simulation_dumper.restore_simulation
```
