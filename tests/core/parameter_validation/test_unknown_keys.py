# -*- coding: utf-8 -*-
"""Tests for warning on unknown keys placed beside ``values:`` in parameter YAML.

Regression tests for issue #505: an ``uprating:`` block written as a sibling of
``values:`` (instead of nested under ``metadata:``) was silently dropped, freezing
indexed parameters. The loader now emits a ``ParameterKeyWarning`` naming the file,
the offending key, and the fix.
"""

import os
import warnings

import pytest

from policyengine_core.parameters import ParameterNode, load_parameter_file
from policyengine_core.warnings import ParameterKeyWarning

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_unknown_key_beside_values_warns_from_file():
    path = os.path.join(BASE_DIR, "unknown_key_beside_values.yaml")
    with pytest.warns(ParameterKeyWarning) as record:
        load_parameter_file(path, "unknown_key_beside_values")

    assert len(record) == 1
    message = str(record[0].message)
    # Names the offending key, the file, and points to the fix.
    assert "uprating" in message
    assert "unknown_key_beside_values.yaml" in message
    assert "metadata" in message


def test_uprating_under_metadata_does_not_warn():
    path = os.path.join(BASE_DIR, "uprating_under_metadata.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParameterKeyWarning)
        # Should not raise: the placement is correct.
        param = load_parameter_file(path, "uprating_under_metadata")
    assert param.metadata["uprating"] == "some.index"


def test_unknown_key_beside_values_in_dict_warns():
    with pytest.warns(ParameterKeyWarning, match="uprating"):
        ParameterNode(
            data={
                "my_param": {
                    "values": {"2020-01-01": 100},
                    "uprating": {"parameter": "some.index"},
                },
            }
        )


def test_metadata_beside_values_in_dict_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParameterKeyWarning)
        node = ParameterNode(
            data={
                "my_param": {
                    "values": {"2020-01-01": 100},
                    "metadata": {"uprating": "some.index"},
                },
            }
        )
    assert node.my_param.metadata["uprating"] == "some.index"


def test_unknown_key_on_scale_warns():
    with pytest.warns(ParameterKeyWarning, match="bogus"):
        ParameterNode(
            data={
                "scale": {
                    "bogus": "oops",
                    "brackets": [
                        {
                            "threshold": {"values": {"2020-01-01": 0}},
                            "rate": {"values": {"2020-01-01": 0.1}},
                        },
                    ],
                },
            }
        )


def test_bracket_metadata_does_not_warn():
    # Bracket-level metadata is a legitimate carrier (see issue #390).
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParameterKeyWarning)
        ParameterNode(
            data={
                "scale": {
                    "brackets": [
                        {
                            "threshold": {"values": {"2020-01-01": 0}},
                            "rate": {"values": {"2020-01-01": 0.1}},
                            "metadata": {"uprating": "some.index"},
                        },
                    ],
                },
            }
        )


def test_unknown_key_on_bracket_component_leaf_warns():
    with pytest.warns(ParameterKeyWarning, match="uprating"):
        ParameterNode(
            data={
                "scale": {
                    "brackets": [
                        {
                            "threshold": {
                                "values": {"2020-01-01": 0},
                                "uprating": "some.index",  # wrong: sibling of values
                            },
                            "rate": {"values": {"2020-01-01": 0.1}},
                        },
                    ],
                },
            }
        )


def test_unknown_key_on_bracket_warns():
    # A dict-valued unknown key on a bracket would otherwise be silently turned
    # into a phantom child node; the loader warns instead.
    with pytest.warns(ParameterKeyWarning, match="bogus"):
        ParameterNode(
            data={
                "scale": {
                    "brackets": [
                        {
                            "threshold": {"values": {"2020-01-01": 0}},
                            "rate": {"values": {"2020-01-01": 0.1}},
                            "bogus": {"values": {"2020-01-01": 1}},
                        },
                    ],
                },
            }
        )


def test_valid_parameter_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParameterKeyWarning)
        ParameterNode(
            data={
                "my_param": {
                    "description": "A parameter",
                    "documentation": "Docs here",
                    "values": {"2020-01-01": 100},
                    "metadata": {"unit": "currency-USD"},
                },
            }
        )
