# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference fixtures preserve node geometry and genuine branching."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("catheter_migration", ROOT / "scripts/qualify_catheter_migration.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "arena"))


def test_curved_fixture_is_resampled_and_has_a_real_branch():
    config = {
        "geometry": {
            "route_length": 0.9,
            "route_spacing": 0.002,
            "wall_spacing": 0.005,
            "curve_radius": 0.5,
            "start": [0, 0, 0.5],
            "branch_at_arc": 0.45,
            "branch_angle_rad": 0.7,
            "branch_length": 0.15,
        },
        "rod": {"length": 0.4, "segments": 120},
    }
    route, points, wall, edges = module.geometry(config, {})
    assert points.shape == (121, 3)
    assert np.ptp(points[:, 1]) > 0.1
    np.testing.assert_allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), 0.4 / 120, atol=2e-7)
    degree = np.bincount(edges.flatten(), minlength=len(wall))
    assert (degree == 3).sum() == 1
    assert (degree == 1).sum() == 3
    assert edges.max() == len(wall) - 1


def test_escape_comparison_keeps_particle_identity():
    wall = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    edges = np.asarray([[0, 1]])
    reference, _ = module.escapes(np.asarray([[[0.5, 1.1, 0.0], [0.5, 0.0, 0.0]]]), wall, edges, 1.0)
    numerical, _ = module.escapes(np.asarray([[[0.5, 0.0, 0.0], [0.5, 1.1, 0.0]]]), wall, edges, 1.0)
    assert reference.sum() == numerical.sum() == 1
    assert np.count_nonzero(numerical & ~reference) == 1


def test_arrival_uses_control_steps_instead_of_physics_steps():
    import pytest

    pytest.importorskip("torch")
    gates = {"arrival_tolerance_m": 0.008, "arrival_hold_control_steps": 15}
    frames = np.zeros((60, 2, 3), np.float32)
    target = np.zeros(3)
    assert not module.arrival_decisions(frames[:59], target, gates).any()
    assert module.arrival_decisions(frames, target, gates)[-1]
    frames[-1, -1, 0] = 0.009
    assert not module.arrival_decisions(frames, target, gates).any()
