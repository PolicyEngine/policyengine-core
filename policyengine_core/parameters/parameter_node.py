import copy
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


def _clear_at_instant_caches_upward(node) -> None:
    """Clear the nodes at instant cached by ``node`` and every node above it.

    Unlike ``clear_parent_cache``, this walks up without recursing and stops at
    a node it has seen. ``add_child`` calls it while a node loads, and until a
    loading node is attached to its own parent, a child named ``parent`` (as
    policyengine-us's ``gov.hhs.medicaid.eligibility.categories.parent``)
    replaces the node's ``parent`` attribute and points back at the node.
    """
    seen = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        cache = node.__dict__.get("_at_instant_cache")
        # A child can be named ``_at_instant_cache`` too.
        if isinstance(cache, dict):
            cache.clear()
        node = getattr(node, "parent", None)


class _AtInstantCache(dict):
    """A parameter node's nodes at instant, by instant.

    Reads add to it: a node at instant that resolves a child fills that
    child's cache. So a copy or a pickle takes its items in one step first,
    instead of going through the dictionary while another thread adds to it.
    It stays one object, so parameter nodes that share it, such as shallow
    copies, share its copy too. A pickle holds a plain ``dict``, which earlier
    versions of core can load, and which ``_get_at_instant`` replaces by one
    of these before it adds to it.
    """

    __slots__ = ()

    def __reduce_ex__(self, protocol):
        return (dict, (dict.copy(self),))

    def __deepcopy__(self, memo: dict) -> "_AtInstantCache":
        copied = type(self)()
        memo[id(self)] = copied
        for instant, node_at_instant in dict.copy(self).items():
            copied[copy.deepcopy(instant, memo)] = copy.deepcopy(node_at_instant, memo)
        return copied


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
        self._at_instant_cache: typing.Dict[Instant, ParameterNodeAtInstant] = (
            _AtInstantCache()
        )
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

        :param name: Name of the child that must be used to access that child. Should not contain anything that could interfere with the operator `.` (dot).
        :param child: The new child, an instance of :class:`.ParameterScale` or :class:`.Parameter` or :class:`.ParameterNode`.

        Nodes at instant cached by this node and by every node above it were
        built without the new child, so they are cleared.
        """
        if name in self.children:
            raise ValueError("{} has already a child named {}".format(self.name, name))
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
        # Before the child is set as an attribute, which a child named
        # ``_at_instant_cache`` or ``parent`` would replace.
        _clear_at_instant_caches_upward(self)
        self.children[name] = child
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
            setattr(clone, child_key, child)
            child.parent = clone
        clone._at_instant_cache = _AtInstantCache()

        return clone

    def _get_at_instant(self, instant: Instant) -> ParameterNodeAtInstant:
        cache = self._at_instant_cache
        if instant in cache:
            return cache[instant]
        node_at_instant = ParameterNodeAtInstant(self.name, self, instant)
        if self.trace:
            at_instant = TracingParameterNodeAtInstant(
                node_at_instant, self.tracer, self.branch_name
            )
        else:
            at_instant = node_at_instant
        # Building the node can read code that replaces this node's cache, so
        # store into the cache the node holds now, as before.
        cache = self._at_instant_cache
        if type(cache) is dict:
            # A plain ``dict`` set from outside core (policyengine-us's path
            # copies set one) changes as this node is read, so copying it
            # while another thread reads could fail. It is replaced, with its
            # entries, rather than written to.
            cache = self._at_instant_cache = _AtInstantCache(cache)
        cache[instant] = at_instant
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
