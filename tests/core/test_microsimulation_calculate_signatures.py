"""Each ``Microsimulation`` calculate method takes the arguments ``Simulation``'s takes.

``Population.__call__`` sends a formula's read to ``calculate``,
``calculate_add`` or ``calculate_divide`` with one set of keyword arguments.
``Microsimulation`` overrides those three methods and ``calculate_dataframe``.
When an override lacks one of the base method's arguments, Python rejects the
call with a ``TypeError`` in a ``Microsimulation``, and a ``Simulation``
computes the same call.

Any call that passes its arguments by keyword and binds to a ``Simulation``
calculate method also binds to the ``Microsimulation`` override. The test
checks that for every subset of each method's optional arguments.
"""

import inspect
from itertools import chain, combinations

import pytest

from policyengine_core.simulations import Microsimulation, Simulation


def calculate_overrides() -> list:
    """The ``Simulation`` calculate methods that ``Microsimulation`` overrides."""
    return sorted(
        name
        for name, member in vars(Microsimulation).items()
        if name.startswith("calculate")
        and inspect.isfunction(member)
        and inspect.isfunction(getattr(Simulation, name, None))
    )


def keyword_calls(name: str) -> list:
    """Every set of argument names a caller can pass to ``Simulation``'s method."""
    parameters = inspect.signature(getattr(Simulation, name)).parameters.values()
    required = [
        parameter.name
        for parameter in parameters
        if parameter.default is parameter.empty
    ]
    optional = [
        parameter.name
        for parameter in parameters
        if parameter.default is not parameter.empty
    ]
    subsets = chain.from_iterable(
        combinations(optional, size) for size in range(len(optional) + 1)
    )
    return [(*required, *subset) for subset in subsets]


CALLS = [
    (name, arguments)
    for name in calculate_overrides()
    for arguments in keyword_calls(name)
]


def test_microsimulation_overrides_the_four_calculate_methods():
    # An empty list would leave the property below with nothing to check.
    assert {
        "calculate",
        "calculate_add",
        "calculate_divide",
        "calculate_dataframe",
    } <= set(calculate_overrides())


@pytest.mark.parametrize(
    "name,arguments",
    CALLS,
    ids=[f"{name}({', '.join(arguments[1:])})" for name, arguments in CALLS],
)
def test_keyword_call_to_a_simulation_method_binds_to_the_override(name, arguments):
    call = dict.fromkeys(arguments)
    inspect.signature(getattr(Simulation, name)).bind(**call)

    try:
        inspect.signature(getattr(Microsimulation, name)).bind(**call)
    except TypeError as error:
        pytest.fail(
            f"Microsimulation.{name} rejects a call that Simulation.{name} "
            f"takes: {error}",
            pytrace=False,
        )
