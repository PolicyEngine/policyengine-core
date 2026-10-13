"""Copy only the parts of a parameter tree that a change will touch."""

from typing import Iterable, Set, Union

from policyengine_core import commons

from .parameter import Parameter
from .parameter_node import ParameterNode
from .parameter_scale import ParameterScale

TreeNode = Union[ParameterNode, ParameterScale, Parameter]


def copy_paths(root: ParameterNode, paths: Iterable[str]) -> ParameterNode:
    """Return a tree that shares every node of ``root`` except those on ``paths``.

    Each path names a parameter as ``set_parameter`` and
    ``ParameterNode.get_child`` do: dotted child names, with ``name[i]`` for
    bracket ``i`` of a scale. The returned root, every node between it and each
    named parameter, and each named parameter are new; every other node, with
    the at-instant values it has cached, is ``root``'s own.

    A full ``clone`` rebuilds all of a large tree (policyengine-us has over
    130,000 nodes) and drops every cached at-instant value, which costs
    seconds and a copy of the tree. Copying the paths instead costs a few
    nodes per path.

    Change the returned tree only at the named parameters, for example with
    ``Parameter.update``: an update clears the at-instant caches of the new
    nodes above it and marks them modified, and stops there. A node shared
    with ``root`` still names its parent in ``root``'s tree, so changing it
    would change ``root``'s tree too.

    Raises the ``ValueError`` of ``ParameterNode.get_child`` if a path names
    no parameter, before copying anything.
    """
    paths = list(paths)
    for path in paths:
        root.get_child(path)

    copied_root = _copy_node(root)
    copies = {id(copied_root)}
    for path in paths:
        node = copied_root
        for name in path.split("."):
            if "[" in name:
                scale_name, index = name.split("[")
                index = int(index[:-1])
                scale = _copied_child(node, scale_name, copies)
                bracket = scale.brackets[index]
                if id(bracket) not in copies:
                    bracket = _copy_node(bracket)
                    copies.add(id(bracket))
                    scale.brackets[index] = bracket
                    bracket.parent = scale
                node = bracket
            else:
                node = _copied_child(node, name, copies)
    return copied_root


def _copied_child(node: ParameterNode, name: str, copies: Set[int]) -> TreeNode:
    """``node``'s child ``name``, replaced by a copy unless it already is one."""
    child = node.children[name]
    if id(child) in copies:
        return child
    child = _copy_node(child)
    copies.add(id(child))
    node.children[name] = child
    setattr(node, name, child)
    child.parent = node
    return child


def _copy_node(node: TreeNode) -> TreeNode:
    """A copy of ``node`` that shares its children and their caches.

    A parameter is cloned outright, as it has no children to share. A node or
    scale gets its own children mapping or bracket list, holding the same
    children, and a node an empty at-instant cache, since a value below it is
    about to change.
    """
    if isinstance(node, Parameter):
        return node.clone()
    copied = commons.empty_clone(node)
    copied.__dict__ = node.__dict__.copy()
    if isinstance(node, ParameterScale):
        copied.brackets = list(node.brackets)
    else:
        copied.children = dict(node.children)
        copied._at_instant_cache = {}
    return copied
