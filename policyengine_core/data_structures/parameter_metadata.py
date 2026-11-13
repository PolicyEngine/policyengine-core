from typing import List, Union

from .reference import Reference


class UpratingIndex(str):
    """
    The name of a parameter whose values are used to uprate a parameter node.
    """


class SelfUprating(UpratingIndex):
    """
    A special value ("self") that indicates that the parameter node should be uprated based on the fixed rate increase between two points in its history.
    """


class UpratingRoundingConfig(dict):
    type: str
    """Either 'nearest', 'upwards' or 'downwards'."""

    interval: float
    """The interval to round to."""


class UpratingBase(dict):
    """
    The statutory base amount that base-mode uprating projects from.
    """

    period: Union[int, str]
    """
    The period of the base amount, in the uprating index's own periods: a year, such as 2025, or a date, such as 1997-01-01. Base mode measures each increase from the index's value here.
    """

    value: float
    """
    The base amount. Optional: it defaults to the parameter's value at ``period``, and when the parameter has a value there, a stated ``value`` must match it to one part in a billion.
    """

    floor: bool
    """
    Optional, default false. When true, base mode counts the index as no lower than its base-period level, so the index ratio is at least 1 and a positive base never projects below itself, as in statutes that add the percentage (if any) by which an index exceeds its base-year level.
    """


class UpratingSchema(dict):
    """
    A schema for uprating a parameter node.
    """

    parameter: UpratingIndex
    """
    The name of the parameter whose values are used to uprate the parameter node.
    """

    start_instant: int
    """
    The instant to add uprated values after. Chained mode only.
    """

    rounding: Union[UpratingRoundingConfig, float]
    """
    The rounding to apply. Chained mode rounds the uprated amount, taking an exact half to the even multiple; base mode rounds the increase over the base amount, computed exactly on the decimals the inputs print as, taking an exact half up. A bare float rounds to the nearest multiple of it in either mode, and base mode also reads a bare int that way.
    """

    base: UpratingBase
    """
    Opts in to base mode: each value after the last explicit one is the base amount plus the increase the index implies since the base period, with ``rounding`` applied to that increase and, with ``floor``, an index ratio of at least 1. Base mode takes only ``parameter``, ``base`` and ``rounding``.
    """


class ParameterMetadata:
    """A single legislative value that can change over time."""

    name: str
    """The name identifying this parameter. Should be snake-case and Python-safe."""

    label: str
    """The display text to use for the parameter. Should be short and less than a full sentence."""

    description: str
    """A longer description of the parameter."""

    documentation: str
    """Any relevant information about the parameter's *implementation* (not its legislative meaning- use the description for that field)."""

    reference: List[Reference]
    """A list of references to legislation or other documents that define this parameter."""

    unit: str
    """The real-world meaning of a particular value."""

    uprating: Union[UpratingIndex, UpratingSchema]
    """The schema for uprating this parameter.

    By default (chained mode), the process for uprating parameter $X$ based on parameter $Y$ is as follows:

    1. Look at $X$ and $Y$ at the same time, and move forward in history.

    2. If $Y$ increases by $z$%, then increase $X$ by $z$%.

    With ``base`` (base mode), each value after the last explicit one is a fixed base amount plus the increase $Y$ implies since the base period. See "Uprating" in the parameters guide.
    """
