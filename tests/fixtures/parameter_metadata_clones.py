"""Parameter trees, copies and edits for the parameter clone tests.

``tree_data`` builds a tree with metadata on every kind of object: nodes,
parameters, dated values, scales and brackets. Some metadata values are
nested (a list of reference dicts, a dict holding a list), which is what a
clone shares with its source.

``deep_clone`` copies a tree the way Core did before clones shared nested
metadata: every metadata dict is deep-copied. It is the reference the
differential tests compare ``clone`` against. ``interpolate_in_place`` is the
same for ``interpolate_parameters``.

``snapshot`` turns a tree into plain data, so two trees, or one tree before
and after an edit, can be compared with ``==``.

``apply_edit`` makes one edit through an API that callers use on a tree:
``Parameter.update``, assigning a value, removing one, the metadata dict's
own methods, replacing a nested metadata value with a new one, and Core's
``propagate_parameter_metadata`` and ``interpolate_parameters``.
"""

import copy

from policyengine_core import commons
from policyengine_core.parameters import (
    Parameter,
    ParameterAtInstant,
    ParameterNode,
    ParameterScale,
    interpolate_parameters,
    propagate_parameter_metadata,
)
from policyengine_core.periods import instant

# ``snapshot`` reads each parameter at these instants.
INSTANTS = ["2014-06-01", "2016-01-01", "2018-07-01", "2020-01-01", "2022-01-01"]


def reference(label):
    return [{"title": f"{label} act", "href": f"https://example.gov/{label}"}]


def tree_data():
    """A tree with metadata, nested where it can be, on every kind of object."""
    return {
        "metadata": {"label": "Root", "reference": reference("root")},
        "benefit": {
            "metadata": {
                "label": "Benefit",
                "propagate_metadata_to_children": True,
                "notes": {"sources": ["benefit.pdf"]},
            },
            "amount": {
                "description": "An interpolated amount.",
                "metadata": {
                    "unit": "currency-USD",
                    "reference": reference("amount"),
                    "interpolation": {"interval": "year"},
                },
                "values": {
                    "2015-01-01": {
                        "value": 100,
                        "metadata": {"reference": reference("2015")},
                    },
                    "2019-01-01": 140,
                },
            },
            "eligible": {"2015-01-01": True, "2020-01-01": False},
        },
        "allowance": {
            "metadata": {
                "propagate_metadata_to_children": True,
                "interpolation": {"interval": "year"},
            },
            "single": {"values": {"2015-01-01": 100, "2019-01-01": 140}},
            "couple": {"values": {"2015-01-01": 200, "2019-01-01": 280}},
        },
        "tax": {
            "metadata": {"label": "Tax", "reference": reference("scale")},
            "brackets": [
                {
                    "metadata": {"notes": {"sources": ["bracket.pdf"]}},
                    "threshold": {"values": {"2015-01-01": 0}},
                    "rate": {
                        "metadata": {"reference": reference("rate")},
                        "values": {"2015-01-01": 0.1, "2018-01-01": 0.12},
                    },
                },
                {
                    "threshold": {"values": {"2015-01-01": 1_000}},
                    "rate": {"values": {"2015-01-01": 0.2}},
                },
            ],
        },
    }


def build(data=None):
    return ParameterNode(data=copy.deepcopy(tree_data() if data is None else data))


def deep_clone(item):
    """Clone ``item`` as Core did before clones shared nested metadata."""
    clone = commons.empty_clone(item)
    clone.__dict__ = item.__dict__.copy()
    clone.metadata = copy.deepcopy(item.metadata)
    if isinstance(item, ParameterAtInstant):
        return clone
    if isinstance(item, Parameter):
        clone.values_list = [deep_clone(value) for value in item.values_list]
        return clone
    if isinstance(item, ParameterScale):
        clone.brackets = [deep_clone(bracket) for bracket in item.brackets]
        for bracket in clone.brackets:
            bracket.parent = clone
        return clone
    clone.children = {key: deep_clone(child) for key, child in item.children.items()}
    for key, child in clone.children.items():
        setattr(clone, key, child)
        child.parent = clone
    clone._at_instant_cache = {}
    return clone


def interpolate_in_place(root):
    """``interpolate_parameters`` as Core had it, with its completion marker.

    The reference for the values interpolation gives. It marks a dict that
    several objects can share, so compare it only on parameters that each
    have their own.
    """
    for parameter in root.get_descendants():
        if isinstance(parameter, Parameter):
            if "interpolation" in parameter.metadata and not parameter.metadata[
                "interpolation"
            ].get("completed", False):
                interpolated_entries = []
                for i in range(len(parameter.values_list) - 1):
                    start = instant(parameter.values_list[::-1][i].instant_str)
                    num_intervals = 1
                    interval_size = parameter.metadata["interpolation"]["interval"]
                    parameter_dates = [
                        at_instant.instant_str for at_instant in parameter.values_list
                    ]
                    while (
                        str(start.offset(num_intervals, interval_size))
                        < parameter_dates[0]
                    ):
                        num_intervals += 1
                    for j in range(1, num_intervals):
                        start_str = str(start.offset(j, interval_size))
                        start_value = parameter.values_list[::-1][i].value
                        end_value = parameter.values_list[::-1][i + 1].value
                        new_value = (
                            start_value + (end_value - start_value) * j / num_intervals
                        )
                        interpolated_entries += [
                            ParameterAtInstant(
                                parameter.name, start_str, data=new_value
                            )
                        ]
                for entry in interpolated_entries:
                    parameter.values_list.append(entry)
                parameter.values_list.sort(key=lambda x: x.instant_str, reverse=True)
                parameter.metadata["interpolation"]["completed"] = True
    return root


def objects(root):
    """Every object in the tree, the root first, in a fixed order."""
    found = [root]
    for item in root.get_descendants():
        found.append(item)
        if isinstance(item, Parameter):
            found.extend(item.values_list)
    return found


def parameters(root):
    return [item for item in root.get_descendants() if isinstance(item, Parameter)]


def snapshot(item):
    """``item`` and everything below it as plain data."""
    metadata = copy.deepcopy(item.metadata)
    if isinstance(item, ParameterAtInstant):
        return ("value", item.instant_str, copy.deepcopy(item.value), metadata)
    if isinstance(item, Parameter):
        return (
            "parameter",
            metadata,
            [snapshot(value) for value in item.values_list],
            # Read through the parameter as well as its list.
            [item._get_at_instant(instant) for instant in INSTANTS],
            item.interpolated,
        )
    if isinstance(item, ParameterScale):
        return ("scale", metadata, [snapshot(bracket) for bracket in item.brackets])
    return (
        "node",
        metadata,
        {key: snapshot(child) for key, child in item.children.items()},
    )


def replaced(value):
    """A new value in place of a metadata value, leaving the old one alone."""
    if isinstance(value, dict):
        return {**value, "replaced": True}
    if isinstance(value, list):
        return [*value, "replaced"]
    return ("replaced", value)


def apply_edit(root, edit):
    """Make ``edit`` on the tree under ``root``.

    Positions index ``objects(root)`` or ``parameters(root)`` modulo their
    length, so any position names something in any tree.
    """
    kind = edit[0]
    if kind == "propagate":
        propagate_parameter_metadata(root)
        return
    if kind == "interpolate":
        interpolate_parameters(root)
        return
    if kind in ("update", "set value", "remove value"):
        parameter = parameters(root)[edit[1] % len(parameters(root))]
        if kind == "update":
            _, _, year, value = edit
            parameter.update(period=f"year:{year}:1", value=value)
        elif kind == "set value":
            _, _, position, value = edit
            values = parameter.values_list
            values[position % len(values)].value = value
        elif len(parameter.values_list) > 1:
            parameter.values_list.pop(edit[2] % len(parameter.values_list))
        return
    found = objects(root)
    metadata = found[edit[1] % len(found)].metadata
    if kind == "set key":
        metadata[edit[2]] = copy.deepcopy(edit[3])
    elif kind == "update keys":
        metadata.update(copy.deepcopy(edit[2]))
    elif kind == "setdefault key":
        metadata.setdefault(edit[2], copy.deepcopy(edit[3]))
    elif kind == "pop key":
        metadata.pop(edit[2], None)
    elif kind == "set interpolation":
        metadata["interpolation"] = {"interval": edit[2]}
    elif kind == "replace values":
        # Change every nested value the documented way: a new value per key.
        for key, value in list(metadata.items()):
            metadata[key] = replaced(value)
    else:
        raise ValueError(f"Unknown edit {edit!r}")
