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
from .parameter_at_instant_cache import ParameterAtInstantCache
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
        self._at_instant_cache = ParameterAtInstantCache()
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

        clone.metadata = copy.deepcopy(self.metadata)
        clone.children = {key: child.clone() for key, child in self.children.items()}
        for child_key, child in clone.children.items():
            setattr(clone, child_key, child)
            child.parent = clone
        clone._at_instant_cache = {}

        return clone

    @property
    def _at_instant_cache(self) -> ParameterAtInstantCache:
        return self._at_instant_cache_store

    @_at_instant_cache.setter
    def _at_instant_cache(self, value) -> None:
        if isinstance(value, ParameterAtInstantCache):
            self._at_instant_cache_store = value
            return
        if not isinstance(value, typing.Mapping):
            raise TypeError("_at_instant_cache must be a mapping")
        cache = ParameterAtInstantCache()
        for instant_key, dated_node in value.items():
            if isinstance(dated_node, TracingParameterNodeAtInstant):
                dated_node = dated_node.parameter_node_at_instant
            cache.put(str(instant_key), dated_node)
        self._at_instant_cache_store = cache

    def get_plain_at_instant(self, instant: Instant) -> ParameterNodeAtInstant:
        """Return the ordinary dated view retained by this node."""

        return self._at_instant_cache.get_or_create(
            instant,
            lambda: ParameterNodeAtInstant(self.name, self, instant),
        )

    def _get_at_instant(self, instant: Instant) -> ParameterNodeAtInstant:
        node_at_instant = self.get_plain_at_instant(instant)
        if self.trace:
            return TracingParameterNodeAtInstant(
                node_at_instant, self.tracer, self.branch_name
            )
        return node_at_instant

    def clear_at_instant_caches(self, *, recursive: bool = True) -> None:
        """Clear this node's dated views and optionally child-node views."""

        self._at_instant_cache.clear()
        if recursive:
            for child in self.children.values():
                if isinstance(child, ParameterNode):
                    child.clear_at_instant_caches(recursive=True)

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
