"""Regression tests for issue #390: scale parameter threshold uprating.

Covers the placements that were silently dead on core before this change, plus
the placements that already worked (locked in so they never regress):

Working before (locked):
- per-leaf ``brackets[i].threshold.metadata.uprating``
- per-leaf ``brackets[i].amount.metadata.uprating``
- scale ``metadata.uprating`` + ``uprate_thresholds: true``

Fixed here:
- bracket-level ``brackets[i].metadata.uprating`` (value-side: amount/rate/...)
- bracket-level ``brackets[i].metadata.threshold.uprating`` (sub-key form)
- bracket-level ``brackets[i].metadata.amount.uprating`` (sub-key form)
- bracket-level ``metadata.uprating`` + ``metadata.uprate_thresholds: true``
- scale ``metadata.threshold_uprating`` (distinct index for all thresholds)
"""

import pytest

from policyengine_core.parameters import (
    ParameterNode,
    ParameterScale,
    uprate_parameters,
)


def _index():
    # A simple index growing 100 -> 110 (a 10% step in 2026).
    return {"values": {"2025-01-01": 100, "2026-01-01": 110}}


# ---------------------------------------------------------------------------
# Working forms locked as regressions (per issue #390 evidence comments).
# ---------------------------------------------------------------------------


def test_per_leaf_threshold_metadata_uprating_still_works():
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {"type": "marginal_rate"},
                "brackets": [
                    {
                        "threshold": {
                            "values": {"2025-01-01": 3000},
                            "metadata": {"uprating": "index"},
                        },
                        "rate": {"values": {"2025-01-01": 0.1}},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(3300)


def test_per_leaf_amount_metadata_uprating_still_works():
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {"type": "single_amount"},
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 0}},
                        "amount": {
                            "values": {"2025-01-01": 4000},
                            "metadata": {"uprating": "index"},
                        },
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].amount("2026-01-01") == pytest.approx(4400)


def test_scale_uprating_with_uprate_thresholds_still_works():
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {
                    "type": "marginal_rate",
                    "uprating": "index",
                    "uprate_thresholds": True,
                },
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 5000}},
                        "rate": {"values": {"2025-01-01": 0.1}},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(5500)


# ---------------------------------------------------------------------------
# Broken forms this change fixes.
# ---------------------------------------------------------------------------


def test_bracket_level_uprating_applies_to_amount():
    # brackets[i].metadata.uprating -> value side (amount), like scale-level bare.
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {"type": "single_amount"},
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 0}},
                        "amount": {"values": {"2025-01-01": 4000}},
                        "metadata": {"uprating": "index"},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].amount("2026-01-01") == pytest.approx(4400)


def test_bracket_level_uprating_does_not_touch_threshold_by_default():
    # Bare bracket uprating is value-side only; the threshold stays frozen unless
    # explicitly requested (mirrors scale-level semantics).
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {"type": "single_amount"},
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 2000}},
                        "amount": {"values": {"2025-01-01": 4000}},
                        "metadata": {"uprating": "index"},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(2000)


def test_bracket_level_threshold_subkey_uprating_applies_to_threshold():
    # brackets[i].metadata.threshold.uprating -> threshold leaf (issue body form).
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {"type": "marginal_rate"},
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 1000}},
                        "rate": {"values": {"2025-01-01": 0.1}},
                        "metadata": {"threshold": {"uprating": "index"}},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(1100)


def test_bracket_level_amount_subkey_uprating_applies_to_amount():
    # brackets[i].metadata.amount.uprating -> amount leaf (symmetric sub-key form).
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {"type": "single_amount"},
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 0}},
                        "amount": {"values": {"2025-01-01": 4000}},
                        "metadata": {"amount": {"uprating": "index"}},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].amount("2026-01-01") == pytest.approx(4400)


def test_bracket_level_uprate_thresholds_flag_applies_to_threshold():
    # brackets[i].metadata.uprating + metadata.uprate_thresholds -> threshold too.
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {"type": "marginal_rate"},
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 1000}},
                        "rate": {"values": {"2025-01-01": 0.1}},
                        "metadata": {
                            "uprating": "index",
                            "uprate_thresholds": True,
                        },
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(1100)


def test_scale_level_threshold_uprating_applies_to_all_thresholds():
    # scale metadata.threshold_uprating -> distinct index for every threshold.
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {
                    "type": "marginal_rate",
                    "threshold_uprating": "index",
                },
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 2000}},
                        "rate": {"values": {"2025-01-01": 0.1}},
                    },
                    {
                        "threshold": {"values": {"2025-01-01": 6000}},
                        "rate": {"values": {"2025-01-01": 0.2}},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(2200)
    assert root.scale.brackets[1].threshold("2026-01-01") == pytest.approx(6600)


def test_scale_threshold_uprating_leaves_amounts_frozen():
    # threshold_uprating must not touch the value side.
    root = ParameterNode(
        data={
            "index": _index(),
            "scale": {
                "metadata": {
                    "type": "single_amount",
                    "threshold_uprating": "index",
                },
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 2000}},
                        "amount": {"values": {"2025-01-01": 4000}},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(2200)
    assert root.scale.brackets[0].amount("2026-01-01") == pytest.approx(4000)


def test_scale_threshold_uprating_and_amount_uprating_combine():
    # Distinct indices for thresholds vs amounts on the same scale.
    root = ParameterNode(
        data={
            "index_low": {"values": {"2025-01-01": 100, "2026-01-01": 110}},
            "index_high": {"values": {"2025-01-01": 100, "2026-01-01": 120}},
            "scale": {
                "metadata": {
                    "type": "single_amount",
                    "uprating": "index_high",
                    "threshold_uprating": "index_low",
                },
                "brackets": [
                    {
                        "threshold": {"values": {"2025-01-01": 2000}},
                        "amount": {"values": {"2025-01-01": 4000}},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(2200)
    assert root.scale.brackets[0].amount("2026-01-01") == pytest.approx(4800)


def test_per_leaf_threshold_overrides_scale_threshold_uprating():
    # Individual threshold metadata should win over the scale-level default.
    root = ParameterNode(
        data={
            "index_low": {"values": {"2025-01-01": 100, "2026-01-01": 110}},
            "index_high": {"values": {"2025-01-01": 100, "2026-01-01": 120}},
            "scale": {
                "metadata": {
                    "type": "marginal_rate",
                    "threshold_uprating": "index_low",
                },
                "brackets": [
                    {
                        "threshold": {
                            "values": {"2025-01-01": 2000},
                            "metadata": {"uprating": "index_high"},
                        },
                        "rate": {"values": {"2025-01-01": 0.1}},
                    },
                    {
                        "threshold": {"values": {"2025-01-01": 6000}},
                        "rate": {"values": {"2025-01-01": 0.2}},
                    },
                ],
            },
        }
    )
    uprate_parameters(root)
    # Bracket 0 uses its own index_high (20% growth).
    assert root.scale.brackets[0].threshold("2026-01-01") == pytest.approx(2400)
    # Bracket 1 falls back to the scale threshold_uprating index_low (10%).
    assert root.scale.brackets[1].threshold("2026-01-01") == pytest.approx(6600)


def test_scale_get_descendants_reaches_all_bracket_leaves():
    # Lock in that uprate_parameters can see every bracket leaf (get_descendants
    # coverage), independent of the propagation path.
    from policyengine_core.parameters import Parameter

    scale = ParameterScale(
        "scale",
        data={
            "metadata": {"type": "marginal_rate"},
            "brackets": [
                {
                    "threshold": {"values": {"2025-01-01": 0}},
                    "rate": {"values": {"2025-01-01": 0.1}},
                },
                {
                    "threshold": {"values": {"2025-01-01": 1000}},
                    "rate": {"values": {"2025-01-01": 0.2}},
                },
            ],
        },
        file_path=None,
    )
    leaves = [d for d in scale.get_descendants() if isinstance(d, Parameter)]
    names = {leaf.name for leaf in leaves}
    assert names == {
        "scale[0].threshold",
        "scale[0].rate",
        "scale[1].threshold",
        "scale[1].rate",
    }
