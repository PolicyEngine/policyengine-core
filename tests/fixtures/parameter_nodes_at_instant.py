"""A parameter tree and an up-front reference for the at-instant tests.

``snapshot`` computes what a ``ParameterNodeAtInstant`` held when it built
every descendant in its constructor, straight from the parameter tree and
without using ``ParameterNodeAtInstant``. ``materialise`` reads the same
thing out of a ``ParameterNodeAtInstant``.
"""

from policyengine_core.parameters import (
    Parameter,
    ParameterNode,
    ParameterNodeAtInstant,
    ParameterScale,
)

INSTANTS = [
    "2009-12-31",
    "2010-01-01",
    "2014-06-30",
    "2015-01-01",
    "2017-03-01",
    "2018-01-01",
    "2020-01-01",
    "2029-12-31",
    "2030-01-01",
    "2035-07-01",
]


def _zone(amount, rate):
    return {
        "amount": {"values": {"2010-01-01": amount, "2020-01-01": amount * 2}},
        "rate": {"values": {"2010-01-01": rate}},
    }


TREE = {
    "flat_rate": {"values": {"2015-01-01": 0.1, "2020-01-01": 0.2}},
    # Not defined before 2030, so absent from earlier instants.
    "late": {"values": {"2030-01-01": 5}},
    "group": {
        "x": {"values": {"2010-01-01": 0.1}},
        "y": {"values": {"2018-01-01": 0.2}},
        # Every child undefined before 2018: an empty node, not an absent one.
        "inner": {
            "1": {"values": {"2018-01-01": 10}},
            "2": {"values": {"2018-01-01": 20, "2020-01-01": 30}},
        },
    },
    "scale": {
        "brackets": [
            {
                "threshold": {"2010-01-01": {"value": 0}},
                "rate": {"2010-01-01": {"value": 0.1}},
            },
            {
                "threshold": {"2010-01-01": {"value": 1000}},
                "rate": {
                    "2010-01-01": {"value": 0.2},
                    "2020-01-01": {"value": 0.3},
                },
            },
        ]
    },
    # Homogeneous children: vectorisable with ``node[array]``.
    "by_zone": {"z1": _zone(100, 0.1), "z2": _zone(200, 0.2), "z3": _zone(300, 0.3)},
}


def build_tree() -> ParameterNode:
    return ParameterNode("", data=TREE)


def describe(value):
    """A comparable form of one at-instant value."""
    if isinstance(value, (dict, int, float, bool, str)) or value is None:
        return value
    # Tax scales refuse ``==``.
    return (
        type(value).__name__,
        tuple(value.thresholds),
        tuple(getattr(value, "rates", getattr(value, "amounts", ()))),
    )


def snapshot(node: ParameterNode, instant_str: str) -> dict:
    """Every descendant's value at the instant, in the node's own order,
    leaving out children that have no value yet."""
    result = {}
    for name, child in node.children.items():
        if isinstance(child, ParameterScale):
            value = describe(child._get_at_instant(instant_str))
        elif isinstance(child, Parameter):
            value = child._get_at_instant(instant_str)
        else:
            value = snapshot(child, instant_str)
        if value is not None:
            result[name] = value
    return result


def materialise(node_at_instant: ParameterNodeAtInstant) -> dict:
    """The same nested mapping, read from a ``ParameterNodeAtInstant``."""
    return {
        name: (
            materialise(child)
            if isinstance(child, ParameterNodeAtInstant)
            else describe(child)
        )
        for name, child in node_at_instant._children.items()
    }


def ordered(mapping: dict):
    """A form that compares key order as well as content."""
    return [
        (key, ordered(value) if isinstance(value, dict) else value)
        for key, value in mapping.items()
    ]
