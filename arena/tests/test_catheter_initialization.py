# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initial poses and resets checked against the actual CPU XPBD constraints."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from i4h_arena.medical.catheter_initialization import (
    initialize_rod_state,
    publish_reset_state,
    rod_frames_along_polyline,
)
from i4h_arena.medical.newton_catheter_physics import (
    CatheterRodHandle,
    CatheterRodSpec,
    _seed_shaft_rest_darboux,
    rest_darboux_along_polyline,
)


@pytest.fixture
def rod_factory(tmp_path, monkeypatch):
    wp = pytest.importorskip("warp")
    from catheter_vasculature_solver import CathRodSolver, RodConfig, RodGeometryConfig, RodSolverConfig

    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp"))

    def build(edges=24, length=0.005, envs=1):
        config = RodConfig(
            geometry=RodGeometryConfig(num_segments=edges, segment_length=length, radius=0.0005),
            solver=RodSolverConfig(num_substeps=1, gravity=(0.0, 0.0, 0.0)),
            device="cpu",
        )
        return CathRodSolver(
            config,
            num_envs=envs,
            collision_enabled=False,
            track_enabled=False,
            floor_z=None,
            collision_mesh=None,
            track_start=(0.0, 0.0, 0.0),
            track_dir=(1.0, 0.0, 0.0),
            track_length=edges * length,
            tip_num_edges=2,
            particle_radius=0.0005,
            segment_length=length,
        )

    return build


def _constraints(rod):
    import warp as wp
    from catheter_vasculature_solver.xpbd_rod_solver import _xr_update_constraints

    ws = rod._ws
    wp.launch(
        _xr_update_constraints,
        dim=ws.num_edges,
        inputs=[ws.positions, ws.orientations, ws.rest_lengths, ws.rest_darboux, ws.constraint_values],
        device="cpu",
    )
    return ws.constraint_values.numpy().reshape(-1, 6)


def _curve(edges=24, radius=0.1, pitch=0.005):
    # Equal chord lengths make length errors independent of resampling.
    angle = np.arange(edges + 1) * (2.0 * np.arcsin(pitch / (2.0 * radius)))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.full_like(angle, 0.2)))


@pytest.mark.parametrize("direction", [(1, 0, 0), (0, 1, 0), (0, 0, -1), (1, 2, -3)])
def test_arbitrary_straight_pose_satisfies_constraints_and_survives_a_step(rod_factory, direction):
    rod = rod_factory()
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    points = np.array([0.2, -0.1, 0.4]) + np.arange(25)[:, None] * 0.005 * direction
    initialize_rod_state(rod, points)

    constraints = _constraints(rod)
    assert np.max(np.abs(constraints[:, :3])) < 1.0e-7  # Metres.
    assert np.max(np.abs(constraints[:, 3:])) < 1.0e-6  # Float32 quaternion roundoff.
    rod.step(1.0 / 120.0)
    np.testing.assert_allclose(rod.position_array.numpy(), points, atol=2.0e-6)


def test_curved_pose_reduces_initial_shear_without_manufacturing_rest_curvature(rod_factory):
    rod = rod_factory()
    points = _curve()
    rod._ws.positions.assign(points.astype(np.float32))
    previous_error = np.linalg.norm(_constraints(rod)[:, :3], axis=1).max()

    initialize_rod_state(rod, points)

    new_error = np.linalg.norm(_constraints(rod)[:, :3], axis=1).max()
    assert new_error < previous_error / 20.0
    assert new_error < 0.0001  # Endpoint frames leave a small discrete residual.
    np.testing.assert_array_equal(rod._ws.rest_darboux.numpy(), 0.0)
    np.testing.assert_allclose(rod._ws.rest_lengths.numpy(), 0.005)
    assert np.linalg.norm(_constraints(rod)[:, 3:]) > 0.0  # Straight-rest device remains stressed.


def test_spatial_curve_has_continuous_frames_without_material_twist():
    u = np.linspace(0.0, 1.5 * np.pi, 80)
    points = np.column_stack((np.cos(u), np.sin(u), 0.2 * u))
    frames = rod_frames_along_polyline(points)
    np.testing.assert_allclose(np.linalg.norm(frames, axis=1), 1.0, atol=1.0e-7)
    assert np.all(np.sum(frames[:-1] * frames[1:], axis=1) > 0.0)
    relative = Rotation.from_quat(frames[:-1]).inv() * Rotation.from_quat(frames[1:])
    np.testing.assert_allclose(relative.as_rotvec()[:, 2], 0.0, atol=1.0e-7)
    axes = Rotation.from_quat(frames).apply([0.0, 0.0, 1.0])
    np.testing.assert_allclose(axes[0], (points[1] - points[0]) / np.linalg.norm(points[1] - points[0]), atol=1e-7)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((1, 3)),
        np.zeros((3, 2)),
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [np.nan, 0, 1]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
    ],
)
def test_invalid_seed_geometry_is_rejected(points):
    with pytest.raises(ValueError):
        rod_frames_along_polyline(points)


@pytest.mark.parametrize("envs", [1, 3])
def test_optional_rest_values_use_the_actual_constraint_convention(rod_factory, envs):
    import torch

    rod = rod_factory(envs=envs)
    points = _curve()
    initialize_rod_state(rod, points)
    # Preserve an independently authored distal precurve.
    for ws in (rod._ws, rod._bws):
        if ws is not None:
            rest = ws.rest_darboux.numpy().reshape(-1, 24, 3)
            rest[:, -2:, 0] = 0.12
            ws.rest_darboux.assign(rest.reshape(-1, 3))
    _seed_shaft_rest_darboux(rod, num_tip_edges=2, segment_length_m=0.005, positions_world_m=points)

    np.testing.assert_allclose(_constraints(rod)[:-2, 3:], 0.0, atol=2e-7)
    for ws in (rod._ws, rod._bws):
        if ws is not None:
            rest = ws.rest_darboux.numpy().reshape(-1, 24, 3)
            np.testing.assert_allclose(rest[:, -2:, 0], 0.12)
            assert np.linalg.norm(rest[:, :-2], axis=-1).max() < 0.03
    # Uniformly scaling geometry must not scale a dimensionless rest rotation.
    short = rest_darboux_along_polyline(torch.tensor(points), 0.005)
    long = rest_darboux_along_polyline(torch.tensor(points * 1000), 5.0)
    np.testing.assert_allclose(short.numpy(), long.numpy(), atol=1e-7)


@pytest.mark.parametrize("envs", [1, 3])
def test_reset_restores_all_pose_buffers_without_reallocation_or_neighbor_changes(rod_factory, envs):
    rod = rod_factory(envs=envs)
    points = _curve().astype(np.float32)
    frames = rod_frames_along_polyline(points)
    initialize_rod_state(rod, points)
    ws = rod._bws if rod._bws is not None else rod._ws
    reset_env = envs - 1
    pose_fields = ("positions", "predicted_positions", "orientations", "predicted_orientations", "prev_orientations")
    momentum_fields = ("velocities", "angular_velocities", "forces", "torques", "lambda_sum")
    pointers = {name: getattr(ws, name).ptr for name in (*pose_fields, *momentum_fields)}
    for name in (*pose_fields, *momentum_fields):
        array = getattr(ws, name)
        array.assign(np.full_like(array.numpy(), 0.3))

    rod.reset([reset_env])

    for name in (*pose_fields, *momentum_fields):
        array = getattr(ws, name)
        assert array.ptr == pointers[name]
        values = array.numpy().reshape(envs, -1, *array.numpy().shape[1:])
        if name in ("positions", "predicted_positions"):
            np.testing.assert_array_equal(values[reset_env], points)
        elif name in pose_fields:
            np.testing.assert_array_equal(values[reset_env], frames)
        else:
            np.testing.assert_array_equal(values[reset_env], 0.0)
        if envs > 1:
            np.testing.assert_array_equal(values[:-1], np.float32(0.3))


def test_scene_reset_publishes_both_newton_states_and_survives_readback(rod_factory, monkeypatch):
    import sys
    import types

    import warp as wp
    from catheter_vasculature_solver import RodParticleBridge, RodParticleRange

    rod = rod_factory(envs=3)
    points = _curve().astype(np.float32)
    initialize_rod_state(rod, points)
    particle_range = RodParticleRange(offset=2, count=75, num_envs=3)
    states = tuple(
        SimpleNamespace(
            particle_q=wp.array(np.full((80, 3), 0.7, dtype=np.float32), dtype=wp.vec3, device="cpu"),
            particle_qd=wp.array(np.full((80, 3), 0.4, dtype=np.float32), dtype=wp.vec3, device="cpu"),
        )
        for _ in range(2)
    )
    module = types.ModuleType("isaaclab_newton.physics")
    module.NewtonManager = SimpleNamespace(get_state_0=lambda: states[0], get_state_1=lambda: states[1])
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", module)
    handle = CatheterRodHandle(CatheterRodSpec(num_segments=24, num_envs=3))
    handle._rod, handle._particle_range = rod, particle_range

    handle.reset([1])

    for state in states:
        np.testing.assert_array_equal(state.particle_q.numpy()[27:52], points)
        np.testing.assert_array_equal(state.particle_qd.numpy()[27:52], 0.0)
        untouched = np.r_[0:27, 52:80]
        np.testing.assert_array_equal(state.particle_q.numpy()[untouched], np.float32(0.7))
        np.testing.assert_array_equal(state.particle_qd.numpy()[untouched], np.float32(0.4))
    RodParticleBridge(rod, particle_range).read_from(states[0])
    np.testing.assert_array_equal(rod.position_array.numpy().reshape(3, 25, 3)[1], points)
    # An empty selection must not touch either state.
    before = states[0].particle_q.numpy().copy()
    publish_reset_state(rod, particle_range, states, [])
    np.testing.assert_array_equal(states[0].particle_q.numpy(), before)


def _soft_runs(stiffness: np.ndarray) -> tuple[int, int]:
    """Softened edge count and how many separate softened stretches there are.

    A correct taper is one stretch at the distal end of each rod, so the run
    count is what distinguishes a tapered rod from the same profile stamped
    repeatedly along one.
    """
    ratio = stiffness[:, 0] / stiffness[:, 0].max()
    soft = (ratio < 0.95).astype(int)
    return int(soft.sum()), int(np.diff(soft).clip(min=0).sum() + soft[0])


@pytest.mark.parametrize("envs", [1, 4])
def test_the_tip_taper_reaches_the_buffers_the_solve_reads(rod_factory, envs):
    """Every rod gets one taper, in both workspaces.

    The batched workspace is a separate allocation seeded by tiling, and the
    batched solve reads only it, so tapering `_ws` alone leaves the tip as stiff
    as the shaft in every multi-rod run. The per-rod edge count has to come from
    each workspace too: `_ws` is one rod wide whatever the environment count is,
    so recovering it by division stamps the profile once per environment along
    that single rod -- and 120 edges divide evenly by the 8 and 4 environments
    the RL profiles ask for, so no size check refuses it.
    """
    import warp as wp

    from i4h_arena.medical.catheter_initialization import solver_workspaces
    from i4h_arena.medical.newton_catheter_physics import _taper_tip_bend_stiffness

    edges, tip = 24, 6
    rod = rod_factory(edges=edges, envs=envs)
    _taper_tip_bend_stiffness(rod, num_tip_edges=tip, tip_fraction=0.2)

    workspaces = solver_workspaces(rod)
    assert len(workspaces) == (2 if envs > 1 else 1)
    for workspace in workspaces:
        stiffness = wp.to_torch(workspace.bend_stiffness).numpy()
        rods = stiffness.shape[0] // edges
        softened, runs = _soft_runs(stiffness)
        assert runs == rods, f"{rods} rods should carry {rods} tapers, found {runs}"
        assert softened == rods * (tip - 1), (workspace, softened)
        # And it is the distal end of each rod that softened, not the proximal.
        per_rod = stiffness[:, 0].reshape(rods, edges)
        assert (per_rod[:, -1] < per_rod[:, 0]).all()
