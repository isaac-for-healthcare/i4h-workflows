# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual CPU rod/contact solves, including motion in a curved lumen."""

import numpy as np
import pytest

from i4h_arena.medical.catheter_diagnostics import tube_surface_gaps_m
from i4h_arena.medical.catheter_initialization import initialize_rod_state
from i4h_arena.medical.newton_catheter_physics import _apply_physical_rotational_inertia, bend_radii_m


@pytest.fixture
def build_rod(tmp_path, monkeypatch):
    wp = pytest.importorskip("warp")
    from catheter_vasculature_solver import CathRodSolver, RodConfig, RodGeometryConfig, RodSolverConfig
    from catheter_vasculature_solver.vessel_deformation import (
        CenterlineData,
        CenterlineDynamicsParams,
        CenterlineVesselRuntime,
        build_centerline_tree,
    )

    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp"))

    def build(envs=1, iterations=8, edges=25, pitch=0.005, two_way=False):
        vessel_edges = max(60, edges + 20)
        angle = np.arange(vessel_edges + 1) * 2 * np.arcsin(pitch / 0.2)
        path = np.column_stack((0.1 * np.sin(angle), 0.1 * (1 - np.cos(angle)), np.zeros_like(angle))).astype(
            np.float32
        )
        radius = np.full(vessel_edges, 0.008, np.float32)
        tree = build_centerline_tree(
            CenterlineData(path[:-1], path[1:], np.zeros(vessel_edges, np.int32), radius, radius, radius, radius)
        )
        vessel = CenterlineVesselRuntime.from_tree(
            tree,
            device="cpu",
            num_envs=envs,
            params=CenterlineDynamicsParams(iterations=2),
            catheter_radius=0.0005,
            two_way=two_way,
        )
        config = RodConfig(
            geometry=RodGeometryConfig(num_segments=edges, segment_length=pitch, radius=0.0005),
            solver=RodSolverConfig(num_substeps=1, gravity=(0.0, 0.0, 0.0)),
            device="cpu",
        )
        rod = CathRodSolver(
            config,
            num_envs=envs,
            collision_mesh=None,
            track_start=path[0],
            track_dir=path[1] - path[0],
            track_length=edges * pitch,
            tip_num_edges=5,
            particle_radius=0.0005,
            segment_length=pitch,
            track_enabled=False,
            collision_enabled=False,
            floor_z=None,
            centerline_runtime=vessel,
            contact_coupling_iterations=iterations,
            containment_cleanup_iterations=128,
            containment_cleanup_rounds=8,
        )
        initialize_rod_state(rod, path[: edges + 1])
        _apply_physical_rotational_inertia(rod, radius_m=0.0005, segment_length_m=pitch)
        return rod, vessel

    return build


@pytest.mark.parametrize("envs", [1, 3])
def test_curved_hold_insert_stop_retract_preserves_shape_and_surface_clearance(build_rod, envs):
    rod, vessel = build_rod(envs=envs)
    initial = rod.position_array.numpy().reshape(envs, 26, 3).copy()
    touched_wall = False
    for step in range(360):
        speed = 0.009 if 120 <= step < 240 else (-0.009 if step >= 300 else 0.0)
        rod.apply_proximal_control(speed, 0.0, 1 / 120)
        rod.step(1 / 120)
        touched_wall |= vessel.contact_count > 0
    points = rod.position_array.numpy().reshape(envs, 26, 3)
    assert touched_wall, "The case must exercise contact, not just free-space bending"
    for env in range(envs):
        assert np.isfinite(points[env]).all()
        lengths = np.linalg.norm(np.diff(points[env], axis=0), axis=1)
        assert np.max(np.abs(lengths / 0.005 - 1)) < 0.015
        assert abs(lengths.sum() - 0.125) < 0.0005
        assert bend_radii_m(points[env]).min() > 0.03
        gaps = tube_surface_gaps_m(
            points[env],
            vessel.positions_per_env[env],
            vessel.tree.edges,
            vessel.radii.numpy().reshape(envs, -1)[env],
            0.0005,
        )
        assert gaps.max() < 0.0001
        # Feed follows the evolving proximal tangent, rather than a fixed
        # world axis; its net travel should still realize the commanded 4.5 mm.
        assert np.linalg.norm(points[env, 0] - initial[env, 0]) == pytest.approx(0.0045, abs=2.0e-6)
    assert np.isfinite(rod.velocity_array.numpy()).all()
    orientations = rod._ws.orientations.numpy() if envs == 1 else rod._bws.orientations.numpy()
    np.testing.assert_allclose(np.linalg.norm(orientations, axis=1), 1, atol=1.0e-6)


def test_global_contact_preserves_smooth_bending(build_rod):
    rod, _ = build_rod()
    for _ in range(30):
        rod.step(1 / 120)
    assert bend_radii_m(rod.position_array.numpy()).min() > 0.04


def test_batched_contact_matches_independent_rods_with_odd_edge_count(build_rod):
    single, _ = build_rod()
    batch, _ = build_rod(envs=3)
    for _ in range(25):
        single.step(1 / 120)
        batch.step(1 / 120)
    expected = single.position_array.numpy()
    for actual in batch.position_array.numpy().reshape(3, 26, 3):
        np.testing.assert_allclose(actual, expected, atol=3.0e-6)


@pytest.mark.parametrize("envs,two_way", [(1, False), (3, True)])
def test_full_length_120_edge_rod_contact_and_reaction_stay_bounded(build_rod, envs, two_way):
    pitch = 0.3032 / 120
    rod, vessel = build_rod(envs=envs, iterations=32, edges=120, pitch=pitch, two_way=two_way)
    for step in range(180):
        speed = 0.003 if 60 <= step < 120 else (-0.003 if step >= 150 else 0.0)
        rod.apply_proximal_control(speed, 0.0, 1 / 120)
        rod.step(1 / 120)
        if step % 30 == 29:
            points = rod.position_array.numpy().reshape(envs, 121, 3)
            wrench = rod.proximal_wrench().numpy()
            assert np.isfinite(wrench).all()
            for env in range(envs):
                lengths = np.linalg.norm(np.diff(points[env], axis=0), axis=1)
                assert abs(lengths.sum() - 0.3032) < 0.0005
                assert np.max(np.abs(lengths / pitch - 1)) < 0.015
                gap = tube_surface_gaps_m(
                    points[env],
                    vessel.positions_per_env[env],
                    vessel.tree.edges,
                    vessel.radii.numpy().reshape(envs, -1)[env],
                    0.0005,
                )
                assert gap.max() < 0.0001
    assert vessel.contact_count > 0
