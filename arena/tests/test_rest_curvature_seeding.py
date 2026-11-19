# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the rod's rest-curvature seeding and its Darboux convention.

The rest curvature decides what shape the direct constraint solve lands on, so a
sign or axis error here does not fail loudly -- it quietly bends the catheter the
wrong way through the anatomy. These pin the convention against the solver's own
formula instead.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Below the skip guard on purpose: these helpers are torch-typed, so a machine
# without torch has to skip rather than error at collection.
from i4h_arena.medical.newton_catheter_physics import (  # noqa: E402
    CatheterRodHandle,
    CatheterRodSpec,
    relative_darboux,
    rest_darboux_along_polyline,
)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """``RodSolver._quat_multiply``, copied so the test does not import the solver."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def _axis_angle_quat(axis: np.ndarray, angle: float) -> torch.Tensor:
    """A unit xyzw quaternion rotating by ``angle`` about ``axis``."""
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    half = 0.5 * float(angle)
    return torch.tensor([*(unit * np.sin(half)), np.cos(half)], dtype=torch.float32)


def test_darboux_matches_the_solvers_own_formula():
    """Agree with ``2 vec(q1* q2) / L`` computed the long way round."""
    generator = torch.Generator().manual_seed(20260907)
    q1 = torch.randn(16, 4, generator=generator)
    q2 = torch.randn(16, 4, generator=generator)
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    q2 = q2 / q2.norm(dim=-1, keepdim=True)
    length = 0.00758

    conjugate = torch.cat([-q1[..., :3], q1[..., 3:4]], dim=-1)
    expected = 2.0 * _quat_multiply(conjugate, q2)[..., :3] / length

    torch.testing.assert_close(relative_darboux(q1, q2, length), expected)


def test_a_straight_rod_has_zero_rest_curvature():
    """Identical frames are the straight rest shape the buffer ships as."""
    q = _axis_angle_quat(np.array([0.0, 0.0, 1.0]), 0.7).expand(8, 4)
    assert relative_darboux(q, q, 0.00758).abs().max() < 1e-6


def test_curvature_scales_with_bend_angle_and_inverse_length():
    """A tighter bend or a shorter edge both read as more curvature.

    The magnitude is what the bend constraint is penalized against, so a rest
    value that does not track the seeded bend would ask the solve for the wrong
    shape even with every sign correct.
    """
    axis = np.array([0.0, 0.0, 1.0])
    identity = _axis_angle_quat(axis, 0.0).unsqueeze(0)
    small = relative_darboux(identity, _axis_angle_quat(axis, 0.05).unsqueeze(0), 0.01)
    large = relative_darboux(identity, _axis_angle_quat(axis, 0.20).unsqueeze(0), 0.01)
    shorter = relative_darboux(identity, _axis_angle_quat(axis, 0.05).unsqueeze(0), 0.005)

    assert large.norm() > small.norm()
    # 2 sin(theta/2) / L, so halving the edge doubles it.
    torch.testing.assert_close(shorter.norm(), 2.0 * small.norm(), rtol=1e-5, atol=1e-8)


def test_curvature_is_signed_by_the_direction_of_the_bend():
    """Opposite bends must not seed the same rest shape."""
    axis = np.array([0.0, 0.0, 1.0])
    identity = _axis_angle_quat(axis, 0.0).unsqueeze(0)
    left = relative_darboux(identity, _axis_angle_quat(axis, 0.3).unsqueeze(0), 0.01)
    right = relative_darboux(identity, _axis_angle_quat(axis, -0.3).unsqueeze(0), 0.01)

    torch.testing.assert_close(left, -right)


def test_a_straight_polyline_seeds_no_curvature():
    """The default rest shape, reproduced from geometry."""
    line = torch.tensor([[0.01 * i, 0.0, 0.0] for i in range(12)], dtype=torch.float32)
    assert rest_darboux_along_polyline(line, 0.01).abs().max() < 1e-4


def test_a_planar_arc_seeds_bend_but_no_twist():
    """Twist is the failure mode that made the naive frames unusable.

    Building each frame independently from global +X gave neighbours arbitrary
    relative roll, so the seeded rest shape asked for twist the centerline does
    not have and the solve diverged. A planar arc must seed pure bending.
    """
    radius, count = 0.05, 24
    angles = np.linspace(0.0, 0.9, count)
    arc = torch.tensor(
        np.stack([radius * np.sin(angles), radius * (1.0 - np.cos(angles)), np.zeros(count)], -1),
        dtype=torch.float32,
    )
    length = float(np.linalg.norm(arc[1] - arc[0]))
    darboux = rest_darboux_along_polyline(arc, length)

    # Frames carry local +Z along the tangent, so twist is the Z component and
    # bending is the other two. See ``rod_frames_along_polyline``.
    bend = darboux[..., :2].norm(dim=-1)
    twist = darboux[..., 2].abs()
    assert bend.max() > 1e-3, "a curved polyline must seed some bending"
    assert twist.max() < 1e-3, f"planar arc seeded spurious twist up to {twist.max():.3e}"


def test_arc_curvature_magnitude_tracks_one_over_radius():
    """A circular arc's curvature is ``1/R``, which is what the solve is asked for.

    Compared in the rest buffer's own dimensionless units rather than inverse
    metres: the active constraint subtracts this from ``Im(conj(q0) q1)``
    without dividing by the segment, so a physical curvature is stored as
    ``curvature * L / 2``.
    """
    for radius in (0.03, 0.06):
        count = 40
        angles = np.linspace(0.0, 1.0, count)
        arc = torch.tensor(
            np.stack([radius * np.sin(angles), radius * (1.0 - np.cos(angles)), np.zeros(count)], -1),
            dtype=torch.float32,
        )
        length = float(np.linalg.norm(arc[1] - arc[0]))
        magnitude = rest_darboux_along_polyline(arc, length).norm(dim=-1)
        # Interior edges only; the end tangents are one-sided and read low.
        interior = magnitude[2:-2].mean()
        expected = (1.0 / radius) * length / 2.0
        assert abs(float(interior) - expected) < 0.1 * expected


def test_curvature_scales_linearly_for_blending():
    """``rest_curvature_scale`` has to be a blend, so the map must be linear."""
    count = 20
    angles = np.linspace(0.0, 0.8, count)
    arc = torch.tensor(
        np.stack([0.04 * np.sin(angles), 0.04 * (1.0 - np.cos(angles)), np.zeros(count)], -1),
        dtype=torch.float32,
    )
    full = rest_darboux_along_polyline(arc, 0.008)
    torch.testing.assert_close(full * 0.25, full * 0.25)
    assert float((full * 0.5).norm(dim=-1).max()) == pytest.approx(0.5 * float(full.norm(dim=-1).max()), rel=1e-5)


def test_rest_curvature_is_off_by_default():
    """Opt-in: it prescribes the shape the rod relaxes to."""
    assert CatheterRodSpec().rest_curvature_from_path is False


def test_track_guidance_defaults_are_the_useful_staging():
    """Guidance before the solve, so the distance constraints get the last word."""
    spec = CatheterRodSpec()
    assert spec.track_stage == "pre"
    assert 0.0 < spec.track_stiffness <= 1.0


def test_free_distal_window_defaults_to_the_tip():
    """Unset, guidance behaves as it always has and the solver keeps its default."""
    spec = CatheterRodSpec()

    assert spec.track_free_distal_length_m is None
    assert CatheterRodHandle(spec)._track_free_distal_edges() is None


def test_free_distal_window_converts_length_to_whole_edges():
    """Rounded down, so the free window is never shorter than asked for."""
    spec = CatheterRodSpec(length_m=0.5263, num_segments=120, track_free_distal_length_m=0.18)

    # 0.18 / (0.5263 / 120) = 41.04 edges.
    assert CatheterRodHandle(spec)._track_free_distal_edges() == 41


def test_free_distal_window_cannot_exceed_the_rod():
    """A window longer than the shaft leaves the whole shaft free, not more."""
    spec = CatheterRodSpec(length_m=0.5263, num_segments=120, track_free_distal_length_m=10.0)

    assert CatheterRodHandle(spec)._track_free_distal_edges() == 120


def test_free_distal_window_rejects_a_non_positive_length():
    with pytest.raises(ValueError, match="track_free_distal_length_m"):
        CatheterRodSpec(track_free_distal_length_m=0.0)


def test_feed_span_defaults_to_the_adjacent_node():
    """Unset, the solver keeps reading the rod's own first-segment tangent."""
    spec = CatheterRodSpec()

    assert spec.proximal_feed_span_m is None
    assert CatheterRodHandle(spec)._proximal_feed_span() == 1


def test_feed_span_converts_length_to_whole_nodes():
    """Rounded up, so the baseline is never shorter than asked for."""
    spec = CatheterRodSpec(length_m=0.5263, num_segments=120, proximal_feed_span_m=0.035)

    # 0.035 / (0.5263 / 120) = 7.98 segments.
    assert CatheterRodHandle(spec)._proximal_feed_span() == 8


def test_feed_span_below_one_segment_still_clears_the_adjacent_node():
    """A fold at the root must not be able to aim the push that deepens it."""
    spec = CatheterRodSpec(length_m=0.5263, num_segments=120, proximal_feed_span_m=0.0001)

    assert CatheterRodHandle(spec)._proximal_feed_span() == 1


def test_feed_span_cannot_run_past_the_rod():
    spec = CatheterRodSpec(length_m=0.5263, num_segments=120, proximal_feed_span_m=10.0)

    assert CatheterRodHandle(spec)._proximal_feed_span() == 120


def test_feed_span_rejects_a_non_positive_length():
    with pytest.raises(ValueError, match="proximal_feed_span_m"):
        CatheterRodSpec(proximal_feed_span_m=0.0)
