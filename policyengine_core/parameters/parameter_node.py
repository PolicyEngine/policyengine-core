import copy
import functools
import os
import typing
from typing import Iterable, List, Type, Union

from policyengine_core import commons, parameters, tools
from policyengine_core.data_structures import Reference
from policyengine_core.periods.instant_ import Instant
from policyengine_core.tracers import TracingParameterNodeAtInstant

from .at_instant_like import AtInstantLike
from .parameter import Parameter
from .parameter_node_at_instant import ParameterNodeAtInstant
from .config import COMMON_KEYS, FILE_EXTENSIONS
from .helpers import (
    load_parameter_file,
    _compose_name,
    _validate_parameter,
    _parse_child,
    _load_yaml_file,
)

EXCLUDED_PARAMETER_CHILD_NAMES = ["reference", "__pycache__"]

# Attributes every ParameterNode sets on itself. A child with one of these
# names stays reachable through ``children`` and at-instant lookups, but is not
# set as an attribute, so it cannot replace the node's own state (for example a
# child named ``parent`` replacing the link to the holding node).
NODE_OWN_ATTRIBUTE_NAMES = frozenset(
    {
        "name",
        "children",
        "description",
        "documentation",
        "file_path",
        "metadata",
        "trace",
        "tracer",
        "branch_name",
        "_at_instant_cache",
        "parent",
        "modified",
    }
)


@functools.lru_cache(maxsize=None)
def _names_blocked_as_child_attributes(node_class: type) -> frozenset:
    # The node's own attributes, plus the methods and class attributes of its
    # classes (including subclasses such as ParameterScaleBracket). Metaclass
    # attributes such as ABCMeta.register or type.mro are not included: they
    # are not reachable from an instance. Computed once per class.
    names = set(NODE_OWN_ATTRIBUTE_NAMES)
    for klass in node_class.__mro__:
        names.update(klass.__dict__)
    return frozenset(names)


def _can_expose_child_as_attribute(node: "ParameterNode", child_name: str) -> bool:
    return child_name not in _names_blocked_as_child_attributes(type(node))


def _check_child_type(child) -> None:
    if not (
        isinstance(child, ParameterNode)
        or isinstance(child, Parameter)
        or isinstance(child, parameters.ParameterScale)
    ):
        raise TypeError(
            "child must be of type ParameterNode, Parameter, or Scale. Instead got {}".format(
                type(child)
            )
        )


class ParameterNode(AtInstantLike):
    """
    A node in the legislation `parameter tree <https://openfisca.org/doc/coding-the-legislation/legislation_parameters.html>`_.
    """

    _allowed_keys: typing.Optional[typing.Iterable[str]] = (
        None  # By default, no restriction on the keys
    )

    parent: "ParameterNode" = None
    """The parent of the node, or None if the node is the root of the tree."""

    def __init__(
        self,
        name: str = "",
        directory_path: str = None,
        data: dict = None,
        file_path: str = None,
    ):
        """
        Instantiate a ParameterNode either from a dict, (using `data`), or from a directory containing YAML files (using `directory_path`).

        :param str name: Name of the node, eg "taxes.some_tax".
        :param str directory_path: Directory containing YAML files describing the node.
        :param dict data: Object representing the parameter node. It usually has been extracted from a YAML file.
        :param str file_path: YAML file from which the `data` has been extracted from.


        Instantiate a ParameterNode from a dict:

        >>> node = ParameterNode('basic_income', data = {
            'amount': {
              'values': {
                "2015-01-01": {'value': 550},
                "2016-01-01": {'value': 600}
                }
              },
            'min_age': {
              'values': {
                "2015-01-01": {'value': 25},
                "2016-01-01": {'value': 18}
                }
              },
            })

        Instantiate a ParameterNode from a directory containing YAML parameter files:

        >>> node = ParameterNode('benefits', directory_path = '/path/to/country_package/parameters/benefits')
        """
        self.name: str = name
        self.children: typing.Dict[
            str,
            typing.Union[ParameterNode, Parameter, parameters.ParameterScale],
        ] = {}
        self.description: str = None
        self.documentation: str = None
        self.file_path: str = None
        self.metadata: dict = {}
        self.trace: bool = False
        self.tracer = None
        self.branch_name = None
        self._at_instant_cache: typing.Dict[Instant, ParameterNodeAtInstant] = {}
        self.parent = None

        if directory_path:
            self.file_path = directory_path
            for child_name in sorted(os.listdir(directory_path)):
                child_path = os.path.join(directory_path, child_name)
                if os.path.isfile(child_path):
                    child_name, ext = os.path.splitext(child_name)

                    # We ignore non-YAML files
                    if child_name.upper() == "README":
                        data = {}
                        metadata = {}

                        with open(child_path, "r") as f:
                            # Get the header as the label (making sure to remove the leading hash), and the rest as the description
                            lines = f.readlines()
                            metadata["label"] = lines[0].replace("# ", "").strip()
                            metadata["description"] = "".join(lines[1:]).strip()

                        self.metadata.update(metadata)

                    if ext not in FILE_EXTENSIONS:
                        continue

                    if child_name == "index":
                        data = _load_yaml_file(child_path) or {}
                        _validate_parameter(self, data, allowed_keys=COMMON_KEYS)
                        self.description = data.get("description")
                        self.documentation = data.get("documentation")
                        self.metadata.update(data.get("metadata", {}))
                    elif child_name not in EXCLUDED_PARAMETER_CHILD_NAMES:
                        child_name_expanded = _compose_name(name, child_name)
                        child = load_parameter_file(child_path, child_name_expanded)
                        self.add_child(child_name, child)

                elif os.path.isdir(child_path):
                    child_name = os.path.basename(child_path)
                    child_name_expanded = _compose_name(name, child_name)
                    child = ParameterNode(
                        child_name_expanded, directory_path=child_path
                    )
                    self.add_child(child_name, child)

        else:
            self.file_path = file_path
            _validate_parameter(
                self, data, data_type=dict, allowed_keys=self._allowed_keys
            )
            self.description = data.get("description")
            self.documentation = data.get("documentation")
            self.metadata.update(data.get("metadata", {}))
            for child_name, child in data.items():
                if (
                    child_name in COMMON_KEYS
                    or child_name in EXCLUDED_PARAMETER_CHILD_NAMES
                ):
                    continue  # do not treat reserved keys as subparameters.

                child_name = str(child_name)
                child_name_expanded = _compose_name(name, child_name)
                child = _parse_child(child_name_expanded, child, file_path)
                self.add_child(child_name, child)

        self.modified: bool = False

    def merge(self, other: "ParameterNode") -> None:
        """
        Merges another ParameterNode into the current node.

        In case of child name conflict, the other node child will replace the current node child.
        """
        for child_name, child in other.children.items():
            self.add_child(child_name, child)

    def add_child(self, name: str, child: Union["ParameterNode", Parameter]):
        """
        Add a new child to the node.

        :param name: Name of the child that must be used to access that child. Should not contain anything that could interfere with the operator `.` (dot). The child is also set as an attribute of the node, unless the name matches one of the node's own attributes (such as ``parent`` or ``children``) or a method or class attribute of its class; it then stays reachable through ``children`` and at-instant lookups.
        :param child: The new child, an instance of :class:`.ParameterScale` or :class:`.Parameter` or :class:`.ParameterNode`.
        """
        if name in self.children:
            raise ValueError("{} has already a child named {}".format(self.name, name))
        _check_child_type(child)
        self._link_child(name, child)

    def _replace_child(
        self,
        name: str,
        child: Union["ParameterNode", Parameter, "parameters.ParameterScale"],
    ) -> None:
        """
        Replace an existing child and link the replacement to this node.

        Internal: homogenization uses it to replace a placeholder Parameter
        with the ParameterNode it builds for a further breakdown level.

        Cached values of this node and its ancestors are cleared. The previous
        child is left as it is, and nodes are not marked as modified. The tree
        is left unchanged when an argument is rejected.

        :param name: Name of the existing child.
        :param child: The replacement, an instance of :class:`.ParameterScale`, :class:`.Parameter` or :class:`.ParameterNode`. It must not be this node or one of its ancestors.
        """
        # Check every argument before changing anything, so a rejected call
        # leaves the tree as it was.
        if name not in self.children:
            raise KeyError(
                "{} has no child named {} to replace".format(self.name, name)
            )
        if not isinstance(name, str):
            raise TypeError(
                "Child names must be strings. {} got {!r}".format(self.name, name)
            )
        _check_child_type(child)
        node_or_ancestor = self
        while node_or_ancestor is not None:
            if node_or_ancestor is child:
                raise ValueError(
                    "{} cannot hold itself or one of its ancestors ({}) as a "
                    "child".format(self.name, child.name)
                )
            node_or_ancestor = getattr(node_or_ancestor, "parent", None)

        self._link_child(name, child)
        # Cached at-instant nodes of this node and its ancestors still hold
        # the previous child's values.
        self.clear_parent_cache()

    def _link_child(self, name: str, child) -> None:
        # Store the child, expose it as an attribute when its name does not
        # collide with one of the node's own attributes or methods, and link it
        # upward so Parameter.update can clear caches and mark ancestors.
        self.children[name] = child
        if _can_expose_child_as_attribute(self, name):
            setattr(self, name, child)
        child.parent = self

    def __repr__(self) -> str:
        result = os.linesep.join(
            [
                os.linesep.join(["{}:", "{}"]).format(name, tools.indent(repr(value)))
                for name, value in sorted(self.children.items())
            ]
        )
        return result

    def get_descendants(self) -> Iterable[Union["ParameterNode", Parameter]]:
        """
        Return a generator containing all the parameters and nodes recursively contained in this `ParameterNode`
        """
        for child in self.children.values():
            yield child
            yield from child.get_descendants()

    def clone(self) -> "ParameterNode":
        clone = commons.empty_clone(self)
        clone.__dict__ = self.__dict__.copy()

        # Own dict, shared values: see ParameterAtInstant.clone.
        clone.metadata = copy.copy(self.metadata)
        clone.children = {key: child.clone() for key, child in self.children.items()}
        for child_key, child in clone.children.items():
            if _can_expose_child_as_attribute(clone, child_key):
                setattr(clone, child_key, child)
            child.parent = clone
        clone._at_instant_cache = {}

        return clone

    def _get_at_instant(self, instant: Instant) -> ParameterNodeAtInstant:
        if instant in self._at_instant_cache:
            return self._at_instant_cache[instant]
        node_at_instant = ParameterNodeAtInstant(self.name, self, instant)
        if self.trace:
            at_instant = TracingParameterNodeAtInstant(
                node_at_instant, self.tracer, self.branch_name
            )
        else:
            at_instant = node_at_instant
        self._at_instant_cache[instant] = at_instant
        return at_instant

    def attach_to_parent(self, parent: "ParameterNode"):
        self.parent = parent

    def clear_parent_cache(self):
        self._at_instant_cache.clear()
        if self.parent is not None:
            self.parent.clear_parent_cache()

    def mark_as_modified(self):
        self.modified = True
        if self.parent is not None:
            self.parent.mark_as_modified()

    def get_child(self, path: str) -> "ParameterNode":
        node = self
        for name in path.split("."):
            try:
                if "[" not in name:
                    node = node.children[name]
                else:
                    try:
                        name, index = name.split("[")
                        index = int(index[:-1])
                        node = node.children[name].brackets[index]
                    except:
                        raise ValueError(
                            "Invalid bracket syntax (should be e.g. tax.brackets[3].rate"
                        )
            except:
                raise ValueError(
                    f"Could not find the parameter {path} (failed at {name})."
                )
        return node
