# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import pytest

from policyengine_core.model_api import *
from policyengine_core.parameters import (
    Parameter,
    ParameterNode,
    ParameterNotFoundError,
)
from policyengine_core.tools import assert_near

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

parameters = ParameterNode(directory_path=LOCAL_DIR)

P = parameters.rate("2015-01-01")


def get_message(error):
    return error.args[0]


def test_on_leaf():
    zone = np.asarray(["z1", "z2", "z2", "z1"])
    assert_near(P.single.owner[zone], [100, 200, 200, 100])


def test_on_node():
    housing_occupancy_status = np.asarray(
        ["owner", "owner", "tenant", "tenant"]
    )
    node = P.single[housing_occupancy_status]
    assert_near(node.z1, [100, 100, 300, 300])
    assert_near(node["z1"], [100, 100, 300, 300])


def test_double_fancy_indexing():
    zone = np.asarray(["z1", "z2", "z2", "z1"])
    housing_occupancy_status = np.asarray(
        ["owner", "owner", "tenant", "tenant"]
    )
    assert_near(P.single[housing_occupancy_status][zone], [100, 200, 400, 300])


def test_double_fancy_indexing_on_node():
    family_status = np.asarray(["single", "couple", "single", "couple"])
    housing_occupancy_status = np.asarray(
        ["owner", "owner", "tenant", "tenant"]
    )
    node = P[family_status][housing_occupancy_status]
    assert_near(node.z1, [100, 500, 300, 700])
    assert_near(node["z1"], [100, 500, 300, 700])
    assert_near(node.z2, [200, 600, 400, 800])
    assert_near(node["z2"], [200, 600, 400, 800])


def test_triple_fancy_indexing():
    family_status = np.asarray(
        [
            "single",
            "single",
            "single",
            "single",
            "couple",
            "couple",
            "couple",
            "couple",
        ]
    )
    housing_occupancy_status = np.asarray(
        [
            "owner",
            "owner",
            "tenant",
            "tenant",
            "owner",
            "owner",
            "tenant",
            "tenant",
        ]
    )
    zone = np.asarray(["z1", "z2", "z1", "z2", "z1", "z2", "z1", "z2"])
    assert_near(
        P[family_status][housing_occupancy_status][zone],
        [100, 200, 300, 400, 500, 600, 700, 800],
    )


def test_wrong_key():
    zone = np.asarray(["z1", "z2", "z2", "toto"])
    with pytest.raises(ParameterNotFoundError) as e:
        P.single.owner[zone]
    assert "'rate.single.owner.toto' was not found" in get_message(e.value)


P_2 = parameters.local_tax("2015-01-01")


def test_with_properties_starting_by_number():
    city_code = np.asarray(["75012", "75007", "75015"])
    assert_near(P_2[city_code], [100, 300, 200])


P_3 = parameters.bareme("2015-01-01")


def test_with_enum():
    class TypesZone(Enum):
        z1 = "Zone 1"
        z2 = "Zone 2"

    zone = np.asarray([TypesZone.z1, TypesZone.z2, TypesZone.z2, TypesZone.z1])
    assert_near(P.single.owner[zone], [100, 200, 200, 100])
