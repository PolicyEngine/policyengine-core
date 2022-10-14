# Using the command-line interface

Use the `policyengine-core` command-line tool to run tests or manage data without writing Python code.

```{eval-rst}
.. argparse::
    :module: policyengine_core.scripts.policyengine_command
    :func: get_parser
    :prog: policyengine-core
```

```{eval-rst}
.. hint:: To list all the datasets for a country package, use `policyengine-core data datasets list`, passing in a country package as needed.
```