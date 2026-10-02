# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical reference cases for the elastic solve and the transmitted wrench."""

import math

import numpy as np
import pytest

from i4h_arena.medical.newton_catheter_physics import _apply_physical_rotational_inertia

wp = pytest.importorskip("warp")
from catheter_vasculature_solver import RodConfig, RodGeometryConfig, RodSolverConfig, XPBDRodSolver  # noqa: E402


@pytest.mark.parametrize("edges", [20, 40])
def test_cantilever_matches_beam_deflection_and_root_load(edges):
    length, radius, force = 0.1, 0.0005, 1.0e-5
    config = RodConfig(
        geometry=RodGeometryConfig(num_segments=edges, segment_length=length / edges, radius=radius),
        solver=RodSolverConfig(num_substeps=1, gravity=(0.0, 0.0, 0.0)),
        device="cpu",
    )
    rod = XPBDRodSolver(config, floor_z=None, initial_height=0.0)
    _apply_physical_rotational_inertia(rod, radius_m=radius, segment_length_m=length / edges)
    load = np.zeros((edges + 1, 3), np.float32)
    load[-1, 1] = force
    rod._ws.forces.assign(load)
    for _ in range(1000):
        rod.step(0.01)
    ei = config.material.young_modulus * config.material.bend_stiffness * math.pi * radius**4 / 4
    expected = force * length**3 / (3 * ei)
    tip = rod.position_array.numpy()[-1]
    assert tip[1] == pytest.approx(expected, rel=0.03)
    wrench = rod.proximal_wrench().numpy()[0]
    np.testing.assert_allclose(wrench[:3], load[-1], rtol=0.03, atol=1e-8)
    np.testing.assert_allclose(wrench[3:], np.cross(tip, load[-1]), rtol=0.03, atol=1e-9)


@pytest.mark.parametrize("envs", [1, 3])
def test_twist_transmits_a_pure_couple_even_when_root_force_is_zero(envs):
    length, radius, angle, dt = 0.01, 0.0005, 0.1, 0.01
    config = RodConfig(
        geometry=RodGeometryConfig(num_segments=1, segment_length=length, radius=radius),
        solver=RodSolverConfig(num_substeps=1, gravity=(0.0, 0.0, 0.0)),
        device="cpu",
    )
    rod = XPBDRodSolver(config, num_envs=envs, floor_z=None)
    ws = rod._bws if envs > 1 else rod._ws
    # Local Z lies on world X. Rotate the distal frame about world X.
    q0 = wp.quat(0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    q1 = wp.mul(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle), q0)
    ws.orientations.assign(np.tile(np.array([list(q0), list(q1)], np.float32), (envs, 1)))
    # Static elastic multipliers = -dt² dU/dC. Compare the resulting torque
    # against the energy derivative, not against the implementation's Jacobian.
    shear = config.material.young_modulus / (2 * (1 + config.material.poisson_ratio))
    gj = shear * config.material.twist_stiffness * math.pi * radius**4 / 2
    multipliers = np.zeros((envs, 6), np.float32)
    multipliers[:, 5] = -(dt**2) * (4 * gj / length) * math.sin(angle / 2)
    ws.lambda_sum.assign(multipliers.ravel())
    rod._last_sub_dt = dt
    expected = gj / length * math.sin(angle)
    result = rod.proximal_wrench().numpy()
    np.testing.assert_allclose(result[:, :3], 0.0, atol=1e-12)
    np.testing.assert_allclose(result[:, 3], expected, rtol=1e-5)
    np.testing.assert_allclose(result[:, 4:], 0.0, atol=1e-10)


def test_global_linear_solve_matches_independent_dense_system():
    """Check the full linear system at the runtime's 120-edge resolution."""
    from catheter_vasculature_solver import xpbd_rod_solver as kernels
    from catheter_vasculature_solver.rod_linear_system import RodLinearSystem

    config = RodConfig(geometry=RodGeometryConfig(num_segments=120, segment_length=0.0025), device="cpu")
    rod = XPBDRodSolver(config, floor_z=None)
    _apply_physical_rotational_inertia(rod, radius_m=config.geometry.radius, segment_length_m=0.0025)
    ws = rod._ws
    wp.launch(
        kernels._xr_compute_jacobians,
        dim=120,
        inputs=[ws.orientations, ws.rest_lengths, ws.jacobian_pos, ws.jacobian_rot],
        device="cpu",
    )
    wp.launch(
        kernels._xr_compute_inv_inertia,
        dim=121,
        inputs=[ws.orientations, ws.quat_inv_masses, ws.inv_inertia_local_diag, ws.inv_inertia],
        device="cpu",
    )
    rng = np.random.default_rng(28)
    ws.constraint_values.assign(rng.normal(0, 1e-4, 720).astype(np.float32))
    ws.compliance.assign(np.tile([1e-10] * 3 + [1e4] * 3, 120).astype(np.float32))
    system = RodLinearSystem(120, 1, "cpu")
    actual = system.project(ws).numpy()
    jp = ws.jacobian_pos.numpy().reshape(120, 6, 6).astype(np.float64)
    jr = ws.jacobian_rot.numpy().reshape(120, 6, 6).astype(np.float64)
    inertia = ws.inv_inertia.numpy().reshape(121, 3, 3).astype(np.float64)
    masses = ws.inv_masses.numpy()
    jacobian = np.zeros((720, 726))
    mass = np.zeros((726, 726))
    for i in range(121):
        mass[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = masses[i] * np.eye(3)
        mass[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = inertia[i]
    for edge in range(120):
        for side in range(2):
            row, col = edge * 6, (edge + side) * 6
            jacobian[row : row + 6, col : col + 3] = jp[edge, :, side * 3 : side * 3 + 3]
            jacobian[row : row + 6, col + 3 : col + 6] = jr[edge, :, side * 3 : side * 3 + 3]
    a = jacobian @ mass @ jacobian.T + np.diag(ws.compliance.numpy().astype(float) + 1e-6)
    b = -ws.constraint_values.numpy().astype(float)
    expected = np.linalg.solve(a, b)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-11)
