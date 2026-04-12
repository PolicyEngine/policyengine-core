import os
import typing
import warnings

import yaml

from policyengine_core.warnings import LibYAMLWarning

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    message = [
        "libyaml is not installed in your environment.",
        "This can make OpenFisca slower to start.",
        "Once you have installed libyaml, run 'pip uninstall pyyaml && pip install pyyaml --no-cache-dir'",
        "so that it is used in your Python environment." + os.linesep,
    ]
    warnings.warn(" ".join(message), LibYAMLWarning)
    from yaml import (
        SafeLoader as Loader,
    )  # type: ignore # (see https://github.com/python/mypy/issues/1153#issuecomment-455802270)

ALLOWED_PARAM_TYPES = (float, int, bool, type(None), typing.List)
COMMON_KEYS = {"description", "metadata", "documentation"}
FILE_EXTENSIONS = {".yaml", ".yml"}


def date_constructor(_loader, node):
    return node.value


yaml.add_constructor("tag:yaml.org,2002:timestamp", date_constructor, Loader=Loader)


def dict_no_duplicate_constructor(loader, node, deep=False):
    explicit_keys = {}
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in explicit_keys:
                raise yaml.parser.ParserError(
                    "", node.start_mark, f"Found duplicate key '{key}'"
                )
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "", node.start_mark, f"Found unhashable key '{key}'"
            ) from exc
        explicit_keys[key] = True

    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node, deep=deep)
    mapping = {}

    for key, value in pairs:
        mapping[key] = value

    return mapping


yaml.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    dict_no_duplicate_constructor,
    Loader=Loader,
)
