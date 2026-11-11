"""Helpers for tests that copy and pickle vectorial parameter nodes."""

import copy
import os
import pickle

import numpy as np

from policyengine_core.enums import Enum
from policyengine_core.parameters import ParameterNode

FANCY_INDEXING_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core",
    "parameters_fancy_indexing",
)
INSTANT = "2015-01-01"

# ``rate.<family>.<housing>.<zone>`` in that directory's rate.yaml, at INSTANT.
FAMILIES = ("single", "couple")
HOUSINGS = ("owner", "tenant")
ZONES = ("z1", "z2")
RATE = {
    "single": {
        "owner": {"z1": 100, "z2": 200},
        "tenant": {"z1": 300, "z2": 400},
    },
    "couple": {
        "owner": {"z1": 500, "z2": 600},
        "tenant": {"z1": 700, "z2": 800},
    },
}


class Family(Enum):
    single = "Single"
    couple = "Couple"


class Housing(Enum):
    owner = "Owner"
    tenant = "Tenant"


class SlottedRecords(np.recarray):
    """A vector whose class defines ``__slots__``.

    Pickle protocols 0 and 1 refuse an object whose ``__slots__`` they can
    read unless its class defines ``__getstate__``.
    """

    __slots__ = ("note",)


def fancy_indexing_parameters() -> ParameterNode:
    """A new tree, so that no test reads another's cached nodes."""
    return ParameterNode(directory_path=FANCY_INDEXING_DIRECTORY)


def _pickle_round_trip(protocol):
    def round_trip(obj):
        return pickle.loads(pickle.dumps(obj, protocol=protocol))

    return round_trip


COPIERS = {
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
    **{
        f"pickle-{protocol}": _pickle_round_trip(protocol)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    },
}
# The copiers whose copy has a vector of its own.
DEEP_COPIERS = {name: copier for name, copier in COPIERS.items() if name != "copy"}
