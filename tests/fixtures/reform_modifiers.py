"""Systems, modifiers and reads for the ``modify_parameters`` tests.

``new_system`` gives a tax-benefit system holding the tree from
``parameter_metadata_clones.tree_data``, which has nodes, parameters, a scale
and brackets. Its brackets have no ``parent``, as loaded brackets don't.

``read`` turns everything a tree returns at an instant into plain data, through
each cache a read can come from: the root, the nodes below it and the brackets.
``warm`` fills those caches.

``make_edit`` makes one edit to a tree. It adds three kinds to
``parameter_metadata_clones.apply_edit``, none of which clears a cache: deleting
a bracket, appending one, and adding or removing a child. These with assigning
and removing a value are what ``Parameter.update`` doesn't see.

``make_edit`` also takes ``("read", position)``, which reads the tree at one of
``INSTANTS``: the only edit that fills a cache.

``modifier`` turns edits and an ending into a modifier function. The ending says
what it does once it has made its edits: hand back the tree it got, or a copy of
it, or something that isn't a tree, or raise.

``run_as_instance`` and ``run_in_place`` are the two ways a reform runs. See
#618.
"""

import copy

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.parameters import (
    ParameterNode,
    ParameterScale,
    ParameterScaleBracket,
)
from policyengine_core.parameters.parameter_node_at_instant import (
    ParameterNodeAtInstant,
)
from policyengine_core.reforms import Reform
from policyengine_core.taxscales import TaxScaleLike
from tests.fixtures.parameter_metadata_clones import (
    INSTANTS,
    apply_edit,
    build,
    parameters as parameters_of,
)

STRUCTURAL_EDITS = ("delete bracket", "append bracket", "add child", "remove child")


class ModifierFailed(Exception):
    """Raised by a modifier whose ending is ``"raises"``."""


def new_system():
    system = CountryTaxBenefitSystem()
    system.parameters = build()
    system._parameters_at_instant_cache = {}
    return system


def _read_at_instant(item):
    if isinstance(item, ParameterNodeAtInstant):
        return {key: _read_at_instant(item[key]) for key in item._children}
    if isinstance(item, TaxScaleLike):
        return (
            type(item).__name__,
            [float(threshold) for threshold in item.thresholds],
            [float(rate) for rate in getattr(item, "rates", [])],
            [float(amount) for amount in getattr(item, "amounts", [])],
        )
    return item


def _brackets(root):
    return [
        bracket
        for item in root.get_descendants()
        if isinstance(item, ParameterScale)
        for bracket in item.brackets
    ]


def read(root, instant_str):
    """What ``root`` and each bracket under it return at an instant."""
    return (
        _read_at_instant(root(instant_str)),
        [_read_at_instant(bracket(instant_str)) for bracket in _brackets(root)],
    )


def reads(root):
    return [read(root, instant_str) for instant_str in INSTANTS]


def system_reads(system):
    """Every instant as a system returns it, then as its tree returns it."""
    from_system = [
        _read_at_instant(system.get_parameters_at_instant(instant_str))
        for instant_str in INSTANTS
    ]
    return from_system, reads(system.parameters)


def warm(system, instant_strs=INSTANTS):
    """Fill the caches of ``system`` and its tree at ``instant_strs``."""
    for instant_str in instant_strs:
        system.get_parameters_at_instant(instant_str)
        read(system.parameters, instant_str)


def _nodes(root):
    return [root] + [
        item
        for item in root.get_descendants()
        if isinstance(item, ParameterNode)
        and not isinstance(item, ParameterScaleBracket)
    ]


def _scales(root):
    return [item for item in root.get_descendants() if isinstance(item, ParameterScale)]


def make_edit(root, edit):
    """Make ``edit`` on the tree under ``root``.

    Positions are taken modulo what's in the tree, so any position names
    something in any tree. An edit with nothing to act on does nothing.
    """
    kind = edit[0]
    if kind == "read":
        read(root, INSTANTS[edit[1] % len(INSTANTS)])
        return
    if kind not in STRUCTURAL_EDITS:
        if kind in ("update", "set value", "remove value") and not parameters_of(root):
            return
        apply_edit(root, edit)
        return
    if kind == "add child":
        _, position, value = edit
        node = _nodes(root)[position % len(_nodes(root))]
        name = f"added_{len(node.children)}"
        while name in node.children:
            name += "_"
        node.add_child(
            name,
            ParameterNode(data={"leaf": {"2015-01-01": value}}),
        )
    elif kind == "remove child":
        node = _nodes(root)[edit[1] % len(_nodes(root))]
        if node.children:
            name = sorted(node.children)[edit[2] % len(node.children)]
            del node.children[name]
            delattr(node, name)
    elif _scales(root):
        scale = _scales(root)[edit[1] % len(_scales(root))]
        if kind == "delete bracket":
            if scale.brackets:
                del scale.brackets[edit[2] % len(scale.brackets)]
        else:
            _, _, threshold, rate = edit
            scale.brackets.append(
                ParameterScaleBracket(
                    "new_bracket",
                    data={
                        "threshold": {"2015-01-01": {"value": threshold}},
                        "rate": {"2015-01-01": {"value": rate}},
                    },
                )
            )


def modifier(edits, ending="same tree"):
    """A modifier function that makes ``edits``, then does as ``ending`` says."""

    def modify(parameters):
        if ending == "deep copy":
            parameters = copy.deepcopy(parameters)
        elif ending == "clone":
            parameters = parameters.clone()
        for edit in edits:
            make_edit(parameters, edit)
        if ending == "no tree":
            return None
        if ending == "raises":
            raise ModifierFailed
        return parameters

    return modify


def apply_steps(system, steps):
    """Make each step of a reform's ``apply()`` on ``system``.

    A step is an edit made straight on the system's tree (``("direct", edit)``),
    a read of it (``("read", position)``), a modifier function (``("modify",
    edits, ending)``), or the dict form of ``modify_parameters`` (``("dict",
    year, value)``), which does nothing once an edit has removed its parameter.
    """
    for step in steps:
        if step[0] == "direct":
            make_edit(system.parameters, step[1])
        elif step[0] == "read":
            make_edit(system.parameters, step)
        elif step[0] == "modify":
            system.modify_parameters(modifier(step[1], step[2]))
        else:
            _, year, value = step
            benefit = system.parameters.children.get("benefit")
            changes = {}
            if benefit is not None and "amount" in benefit.children:
                changes["benefit.amount"] = {f"year:{year}:1": value}
            system.modify_parameters(changes)


def reform_with(steps):
    class reform(Reform):
        def apply(self):
            apply_steps(self, steps)

    return reform


def run_as_instance(system, steps):
    """Run a reform as ``MyReform(baseline)``; return the reform."""
    return reform_with(steps)(system)


def run_in_place(system, steps):
    """Run a reform with ``system.apply_reform_set``; return the system."""
    system.apply_reform_set(reform_with(steps))
    return system


def cached_nodes(root):
    """Each node of a tree with what it has cached, to check nothing replaced it."""
    return {
        id(item): (item, dict(item._at_instant_cache))
        for item in _nodes(root) + _brackets(root)
    }


def assert_keeps_its_cached_nodes(cached):
    for item, nodes in cached.values():
        assert item._at_instant_cache.keys() == nodes.keys()
        for instant_str, node in nodes.items():
            assert item._at_instant_cache[instant_str] is node
