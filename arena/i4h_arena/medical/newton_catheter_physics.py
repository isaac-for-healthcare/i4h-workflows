# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Place the catheter rod under Isaac Lab's Newton manager.

A solver belongs to the physics manager, not to a sensor. This module builds the
``NewtonCfg`` that selects the rod manager and registers the one callback that
has to run while the Newton model is still open, so the rod's particles exist on
the model before it is finalized.

Ordering is the whole difficulty. ``NewtonManager.start_simulation`` dispatches
``PhysicsEvent.MODEL_INIT``, then finalizes the builder, then builds the solver.
Particles added after finalize are invisible to the model, and a solver built
before the particles exist has nothing to drive, so registration happens on
``MODEL_INIT`` and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # Annotation-only. Arena is on the light discovery path and must not pull
    # torch in at import time.
    import torch

CONTAINMENT_STAGE_ENV_VAR = "I4H_CATHETER_CONTAINMENT"
DAMPING_ENV_VAR = "I4H_CATHETER_DAMPING"
CLEANUP_SWEEPS_ENV_VAR = "I4H_CATHETER_CLEANUP"
INTERIOR_CONTAINMENT_ENV_VAR = "I4H_CATHETER_INTERIOR"
CLEANUP_ROUNDS_ENV_VAR = "I4H_CATHETER_ROUNDS"
SEGMENT_COUNT_ENV_VAR = "I4H_CATHETER_SEGMENTS"

#: Segment count the catheter's bend stiffness was tuned at. Refining past it
#: has to be compensated, which is what :func:`mesh_invariant_bend_stiffness`
#: is for. This is a calibration reference, not the shipped resolution.
REFERENCE_NUM_SEGMENTS = 40

#: Segments the catheter ships with. Measured against 40 on the s0011 route,
#: over a matched 55 s hold: single-sample jumps above 0.5 mm went from 3 to 0
#: and the worst from 1.13 mm to 0.23 mm, the drift band narrowed from 1.85 mm
#: to 1.03 mm, and containment improved from 0 of 41 particles outside at
#: -0.01 mm clearance to 0 of 121 at -1.72 mm.
#:
#: Chord stretch reads 100-128% here against 100-110% at 40, which looks like a
#: regression and is not one: the chord probe reports a fraction of the rest
#: segment length, and refining cut that length to a third. In millimetres the
#: excess is 1.51 mm against 1.61 mm. Compare chords across segment counts in
#: absolute terms or not at all.
DEFAULT_NUM_SEGMENTS = 120

#: ``XPBDRodSolverCfg.bend_stiffness``'s default, mirrored so the compensation
#: has a reference to scale without importing the solver package on the light
#: discovery path. ``test_the_reference_stiffness_matches_the_installed_cfg``
#: pins the two together.
REFERENCE_BEND_STIFFNESS = 0.1


def containment_stage_override(environ: Any = None) -> str | None:
    """``"pre"`` or ``"post"`` from ``I4H_CATHETER_CONTAINMENT``, else ``None``.

    An experiment switch, so the two stagings can be measured back to back
    against ``I4H_CATHETER_PROBE`` without editing the scene. Anything
    unrecognised is ignored rather than raising: a mistyped diagnostic should
    not decide how the physics runs.
    """
    raw = (environ if environ is not None else os.environ).get(CONTAINMENT_STAGE_ENV_VAR, "")
    stage = str(raw).strip().lower()
    return stage if stage in ("pre", "post") else None


def rod_damping_override(environ: Any = None) -> float | None:
    """Rod damping from ``I4H_CATHETER_DAMPING``, or ``None`` when unset.

    The predict step scales velocity by ``1 - damping``, so ``1.0`` is fully
    quasi-static: velocity and gravity are zeroed every substep and each step
    becomes a pure geometric projection. That is the setting Mosaic credits for
    their containment result, and it is the one worth pairing with
    ``containment_stage="pre"``.

    Values outside ``[0, 1]`` are ignored, since anything above one would flip
    the sign of the velocity it scales.
    """
    raw = (environ if environ is not None else os.environ).get(DAMPING_ENV_VAR, "")
    try:
        damping = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return damping if 0.0 <= damping <= 1.0 else None


def cleanup_sweeps_override(environ: Any = None) -> int | None:
    """Cleanup sweep count from ``I4H_CATHETER_CLEANUP``, or ``None`` when unset.

    Sweep count is the one containment knob with a monotone measured response
    and no measured cost, so it is worth being able to walk it against
    ``I4H_CATHETER_PROBE`` without a restart argument. ``0`` is meaningful --
    it disables the pass -- so it is accepted; negatives are not.
    """
    raw = (environ if environ is not None else os.environ).get(CLEANUP_SWEEPS_ENV_VAR, "")
    try:
        sweeps = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return sweeps if sweeps >= 0 else None


def cleanup_rounds_override(environ: Any = None) -> int | None:
    """Alternation rounds from ``I4H_CATHETER_ROUNDS``, or ``None`` when unset.

    ``1`` is meaningful -- it restores the original sequencing, where the sweeps
    run once after containment and overrule it -- so it is accepted. Zero and
    negatives are not, since there has to be at least one cleanup pass.
    """
    raw = (environ if environ is not None else os.environ).get(CLEANUP_ROUNDS_ENV_VAR, "")
    try:
        rounds = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return rounds if rounds >= 1 else None


def segment_count_override(environ: Any = None) -> int | None:
    """Rod segment count from ``I4H_CATHETER_SEGMENTS``, or ``None`` when unset.

    Refinement is the one knob that changes what the rod is able to represent
    rather than how hard it is pushed, so it is worth walking against
    ``I4H_CATHETER_PROBE`` without editing the embodiment. Pair it with
    :func:`mesh_invariant_bend_stiffness`, or the rod gets dramatically floppier
    as it refines.

    Values below two are rejected: a single segment has no interior joint and
    so no bend constraint at all.
    """
    raw = (environ if environ is not None else os.environ).get(SEGMENT_COUNT_ENV_VAR, "")
    try:
        segments = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return segments if segments >= 2 else None


def mesh_invariant_bend_stiffness(
    reference_stiffness: float,
    reference_segment_length_m: float,
    segment_length_m: float,
) -> float:
    """Bend stiffness that holds the rod's physical stiffness across refinement.

    The solver's bend compliance is ``1 / (E * bend_stiffness * L * dt^2)``, so
    one joint resists with ``k ~ E * bend_stiffness * L``. What that joint
    measures, though, is the raw relative-frame angle, not curvature: the
    Newton kernel differences ``vec(q0* q1)`` against the rest Darboux and never
    divides by ``L``. A rod held at curvature ``kappa`` therefore turns
    ``kappa * L`` at each joint and stores ``E * bend_stiffness * kappa^2 * L^3``
    there, and summing over the ``length / L`` joints leaves a total that scales
    with ``L^2`` instead of staying put.

    So the discretization is not stiffness-invariant on its own: tripling the
    segment count makes the catheter nine times floppier in bending. Scaling
    ``bend_stiffness`` by ``(L_ref / L)^2`` cancels the ``L^2`` exactly and
    leaves refinement doing only what refinement should -- letting the rod hold
    a tighter curve -- rather than quietly re-tuning the material.

    Args:
        reference_stiffness: Stiffness tuned at ``reference_segment_length_m``.
        reference_segment_length_m: Segment length that tuning was done at.
        segment_length_m: Segment length the rod will actually be built with.

    Returns:
        The compensated stiffness, equal to ``reference_stiffness`` when the two
        lengths match.
    """
    if not reference_segment_length_m > 0.0:
        raise ValueError(f"reference_segment_length_m must be positive, got {reference_segment_length_m}")
    if not segment_length_m > 0.0:
        raise ValueError(f"segment_length_m must be positive, got {segment_length_m}")
    ratio = float(reference_segment_length_m) / float(segment_length_m)
    return float(reference_stiffness) * ratio * ratio


def interior_containment_override(environ: Any = None) -> tuple[float, float] | None:
    """``(deadband, stiffness)`` from ``I4H_CATHETER_INTERIOR``, else ``None``.

    Written ``"deadband,stiffness"``, both fractions in ``[0, 1]`` -- so
    ``"0.5,0.25"`` leaves the inner half of the lumen free and pulls at quarter
    strength beyond it. The two only mean anything together, which is why they
    share one variable: a deadband with no stiffness does nothing, and a
    stiffness with the deadband at the wall is the one-sided behaviour.
    """
    raw = (environ if environ is not None else os.environ).get(INTERIOR_CONTAINMENT_ENV_VAR, "")
    parts = str(raw).split(",")
    if len(parts) != 2:
        return None
    try:
        deadband, stiffness = (float(part.strip()) for part in parts)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= deadband <= 1.0 and 0.0 <= stiffness <= 1.0):
        return None
    return deadband, stiffness


# Isaac's world is Z-up, while the standalone rod solver's config defaults to
# Y-down. Naming the world value here keeps the scene from inheriting a gravity
# vector pointing sideways.
GRAVITY_WORLD_Z_UP = (0.0, 0.0, -9.81)


def tip_bend_stiffness_profile(
    shaft_value: float, num_edges: int, num_tip_edges: int, tip_fraction: float
) -> np.ndarray:
    """Per-edge bend stiffness for a guidewire with a stiff shaft and a floppy tip.

    A real J-tip wire is built this way: the proximal shaft carries full stiffness
    so it stays pushable, and the distal edges are relieved so the tip deflects
    against a vessel wall instead of levering the shaft off the lumen axis. The
    blend is a raised cosine so the transition has no stiffness step for the
    constraint solve to ring on.

    Returns a ``(num_edges,)`` array running from ``shaft_value`` down to
    ``shaft_value * tip_fraction`` over the last ``num_tip_edges`` entries.
    """
    if not 0.0 < tip_fraction <= 1.0:
        raise ValueError(f"tip_fraction must be in (0, 1], got {tip_fraction}")
    if num_tip_edges < 0:
        raise ValueError(f"num_tip_edges must not be negative, got {num_tip_edges}")
    profile = np.full(int(num_edges), float(shaft_value), dtype=np.float32)
    taper = int(min(num_tip_edges, num_edges))
    if taper == 0 or tip_fraction == 1.0:
        return profile
    # One at the shaft end of the taper, zero at the very tip.
    u = np.arange(taper, dtype=np.float64) / max(taper - 1, 1)
    weight = 0.5 * (1.0 + np.cos(np.pi * u))
    profile[num_edges - taper :] = shaft_value * (tip_fraction + (1.0 - tip_fraction) * weight)
    return profile


def _taper_tip_bend_stiffness(solver: Any, *, num_tip_edges: int, tip_fraction: float) -> None:
    """Apply :func:`tip_bend_stiffness_profile` to a built solver's edges."""
    import warp as wp

    workspace = solver._ws
    stiffness = wp.to_torch(workspace.bend_stiffness)
    total_edges = int(stiffness.shape[0])
    num_envs = max(int(getattr(solver, "num_envs", 1)), 1)
    if total_edges % num_envs:
        raise RuntimeError(f"{total_edges} edges do not divide across {num_envs} environments")
    per_env = total_edges // num_envs
    shaft = stiffness.view(num_envs, per_env, 3)[0, 0].clone()
    profile = tip_bend_stiffness_profile(1.0, per_env, num_tip_edges, tip_fraction)
    scale = stiffness.new_tensor(profile).unsqueeze(-1)
    stiffness.view(num_envs, per_env, 3).copy_((shaft * scale).expand(num_envs, per_env, 3))


def segment_inverse_inertia(mass_kg: float, radius_m: float, segment_length_m: float) -> float:
    """Inverse transverse moment of inertia of one rod segment, ``12 / (m (3r^2 + L^2))``.

    The solid-cylinder value about a diameter through its centre of mass. This is
    the quantity the rod's bending response scales with: it says how much frame
    rotation a given off-axis push buys.
    """
    if not mass_kg > 0.0:
        raise ValueError(f"mass_kg must be positive, got {mass_kg}")
    if not radius_m > 0.0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")
    if not segment_length_m > 0.0:
        raise ValueError(f"segment_length_m must be positive, got {segment_length_m}")
    inertia = mass_kg * (3.0 * radius_m * radius_m + segment_length_m * segment_length_m) / 12.0
    return 1.0 / inertia


def _apply_physical_rotational_inertia(solver: Any, *, radius_m: float, segment_length_m: float) -> float:
    """Replace the solver's identity rotational inertia with the rod's own.

    ``XPBDRodSolver`` seeds ``inv_inertia_local_diag`` with ``(1, 1, 1)`` for
    every particle regardless of radius, length or density. That leaves a frame
    a wall cannot turn: containment displaces the nodes, the orientation stays
    where it was, and the next elastic solve pulls the nodes back onto the stale
    frame. It is why containment only persists when it runs last, and so why the
    chord lengths it mangles never get repaired.

    Written before the first step, because the value reaches the kernel by value
    and is therefore baked into the captured CUDA graph rather than re-read.

    Isotropic on purpose. The torsional axis is properly ``m r^2 / 2``, some 38x
    smaller here, but the solver source does not pin down which local axis
    carries the tangent, and guessing wrong would stiffen the rod across the bend
    instead of along it. Isotropic matches the configuration Mosaic measured as
    contained with rest lengths intact.

    Returns the inverse inertia written.
    """
    import warp as wp

    inv_inertia = None
    for workspace in (getattr(solver, "_ws", None), getattr(solver, "_bws", None)):
        diagonal = getattr(workspace, "inv_inertia_local_diag", None)
        if diagonal is None:
            continue
        if inv_inertia is None:
            inv_inertia = segment_inverse_inertia(
                _segment_mass_kg(workspace),
                radius_m,
                segment_length_m,
            )
        value = wp.vec3(inv_inertia, inv_inertia, inv_inertia)
        # Single-env workspaces hold a bare vec3; the batched one holds a
        # per-environment array of them.
        if isinstance(diagonal, wp.array):
            diagonal.fill_(value)
        else:
            workspace.inv_inertia_local_diag = value
    if inv_inertia is None:
        raise RuntimeError("solver exposes no workspace to write rotational inertia into")
    return inv_inertia


def nearest_on_polyline(points: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest point on ``path`` for each of ``points``.

    Returns the closest positions, the index of the owning path edge, and how
    far along that edge the closest point sits, in ``[0, 1]``. The fraction is
    handed back because the caller usually wants to interpolate something else
    along the same edge, such as the lumen radius.
    """
    if path.shape[0] < 2:
        raise ValueError(f"centerline needs at least two samples, got {path.shape[0]}")
    start, end = path[:-1], path[1:]
    along = end - start
    length_sq = np.einsum("ij,ij->i", along, along)
    # A degenerate edge collapses to its start point rather than dividing by
    # zero; duplicated samples do occur in extracted centerlines.
    length_sq = np.where(length_sq > 0.0, length_sq, 1.0)
    offset = points[:, None, :] - start[None, :, :]
    fraction = np.clip(np.einsum("nmj,mj->nm", offset, along) / length_sq, 0.0, 1.0)
    projected = start[None, :, :] + fraction[:, :, None] * along[None, :, :]
    edge = np.linalg.norm(points[:, None, :] - projected, axis=2).argmin(axis=1)
    rows = np.arange(points.shape[0])
    return projected[rows, edge], edge, fraction[rows, edge]


def containment_report(
    positions_world_m: Any,
    *,
    path_world_m: Any,
    lumen_radii_m: Any,
    segment_length_m: float,
) -> dict[str, float]:
    """How far the rod sits outside the lumen, and what it did to its own chords.

    ``positions_world_m`` is ``(num_points, 3)`` for a single environment.
    Penetration is measured against the wall interpolated at the nearest point on
    the centerline, signed so that negative is clearance and positive is through
    the wall. A particle on the axis of an 8 mm vessel therefore reports -8 mm.

    Chords come back as a percentage of ``segment_length_m`` because an
    inextensible rod stored with unequal chords is the specific damage that
    containment-after-solve does, and the number is meaningless in absolute mm.
    """
    points = np.asarray(positions_world_m, dtype=np.float64).reshape(-1, 3)
    path = np.asarray(path_world_m, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(lumen_radii_m, dtype=np.float64).reshape(-1)
    if points.shape[0] < 2:
        raise ValueError(f"need at least two rod particles, got {points.shape[0]}")
    if path.shape[0] < 2:
        raise ValueError(f"centerline needs at least two samples, got {path.shape[0]}")
    if radii.shape[0] != path.shape[0]:
        raise ValueError(f"{radii.shape[0]} radii for {path.shape[0]} centerline samples")
    if not segment_length_m > 0.0:
        raise ValueError(f"segment_length_m must be positive, got {segment_length_m}")

    closest, edge, span = nearest_on_polyline(points, path)
    wall = radii[:-1][edge] + span * (radii[1:][edge] - radii[:-1][edge])
    penetration = np.linalg.norm(points - closest, axis=1) - wall

    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    percent = 100.0 / float(segment_length_m)
    return {
        "worst_penetration_mm": float(penetration.max()) * 1000.0,
        "particles_outside": int((penetration > 0.0).sum()),
        "num_particles": int(points.shape[0]),
        "chord_min_pct": float(chords.min()) * percent,
        "chord_max_pct": float(chords.max()) * percent,
    }


def _segment_mass_kg(workspace: Any) -> float:
    """The mass the solver actually gave a segment, read back off the workspace.

    Taken from the inverse masses rather than recomputed from density so the
    inertia cannot disagree with the mass the solve uses. The root is pinned and
    carries zero inverse mass, hence the search for a free particle.
    """
    inv_masses = workspace.inv_masses
    # Warp arrays copy to host; a plain sequence is already there.
    values = np.asarray(inv_masses.numpy() if hasattr(inv_masses, "numpy") else inv_masses, dtype=np.float64)
    free = values[values > 0.0]
    if free.size == 0:
        raise RuntimeError("rod has no free particle to read a segment mass from")
    return 1.0 / float(free[0])


def relative_darboux(q1: "torch.Tensor", q2: "torch.Tensor", length_m: float) -> "torch.Tensor":
    """The Darboux vector between consecutive rod frames, ``2 vec(q1* q2) / L``.

    Mirrors ``RodSolver._compute_darboux`` so a rest value written from here is
    measured against the same convention the bend constraint uses. Quaternions
    are x, y, z, w, which is what the solver stores.

    Args:
        q1: ``(..., 4)`` frame at the proximal end of each edge.
        q2: ``(..., 4)`` frame at its distal end.
        length_m: Edge length the curvature is expressed per metre of.

    Returns:
        ``(..., 3)`` curvature and twist in the material frame.
    """
    import torch

    x1, y1, z1, w1 = -q1[..., 0], -q1[..., 1], -q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    vec = torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return 2.0 * vec / float(length_m)


def _quat_multiply(q1: "torch.Tensor", q2: "torch.Tensor") -> "torch.Tensor":
    """Hamilton product of xyzw quaternions, matching ``RodSolver._quat_multiply``."""
    import torch

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


def _minimal_rotation(source: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """The shortest xyzw rotation carrying unit ``source`` onto unit ``target``."""
    import torch

    dot = (source * target).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    quat = torch.cat([torch.cross(source, target, dim=-1), 1.0 + dot], dim=-1)
    # Antiparallel tangents leave the axis undetermined; any perpendicular will
    # do, and a vessel centerline doubling back inside one segment is not a case
    # worth choosing carefully for.
    flipped = (dot.squeeze(-1) < -0.999999).unsqueeze(-1)
    fallback = torch.zeros_like(quat)
    fallback[..., 1] = 1.0
    quat = torch.where(flipped, fallback, quat)
    return quat / (quat.norm(dim=-1, keepdim=True) + 1.0e-8)


def rest_darboux_along_polyline(positions_world_m: "torch.Tensor", segment_length_m: float) -> "torch.Tensor":
    """Per-edge rest curvature describing a polyline, with no spurious twist.

    Frames are parallel transported: the first is the shortest rotation taking
    local +X onto the first tangent, and each next one carries its predecessor
    along by the shortest rotation between consecutive tangents. Consecutive
    frames then differ by bending alone.

    That is the whole reason this exists rather than reusing
    ``orientations_xyzw_along_polyline``, which builds every frame independently
    as the shortest rotation from global +X. Those frames point the right way but
    their roll about the tangent jumps arbitrarily from one point to the next, so
    the relative Darboux between them is dominated by twist that the polyline
    does not actually have. Seeding from it drove the solve unstable -- the rod
    stretched to 3375 mm against a 303 mm rest length.

    Args:
        positions_world_m: ``(N, 3)`` particle positions, proximal to distal.
        segment_length_m: Rest length of one edge.

    Returns:
        ``(N - 1, 3)`` rest curvature and twist in each edge's material frame.
    """
    import torch

    positions = positions_world_m.reshape(-1, 3).to(dtype=torch.float32)
    if positions.shape[0] < 2:
        raise ValueError(f"need at least 2 particles to describe curvature, got {positions.shape[0]}")

    # Central differences mid-chain, one-sided at the ends, matching how
    # ``orientations_xyzw_along_polyline`` picks per-point tangents.
    tangents = torch.zeros_like(positions)
    tangents[0] = positions[1] - positions[0]
    tangents[-1] = positions[-1] - positions[-2]
    if positions.shape[0] > 2:
        tangents[1:-1] = positions[2:] - positions[:-2]
    tangents = tangents / (tangents.norm(dim=-1, keepdim=True) + 1.0e-8)

    axis = torch.zeros_like(tangents[0])
    axis[0] = 1.0
    frames = torch.zeros(positions.shape[0], 4, dtype=positions.dtype)
    frames[0] = _minimal_rotation(axis, tangents[0])
    for i in range(1, positions.shape[0]):
        transport = _minimal_rotation(tangents[i - 1], tangents[i])
        frames[i] = _quat_multiply(transport, frames[i - 1])
        frames[i] = frames[i] / (frames[i].norm() + 1.0e-8)

    return relative_darboux(frames[:-1], frames[1:], segment_length_m)


def _seed_shaft_rest_darboux(
    solver: Any,
    *,
    num_tip_edges: int,
    segment_length_m: float,
    positions_world_m: np.ndarray,
    scale: float = 1.0,
) -> None:
    """Make the rod's rest shape the curve it was seeded on, not a straight line.

    ``rest_darboux`` is the curvature the bend constraint measures against, and it
    ships as zeros along the shaft, which is a straight rest shape -- so the rod
    straightens out of curved anatomy and post-solve containment has to drag it
    back every step, which is what mangles the chords. Asking the solve for a rod
    that already follows the vessel is the mechanism that should avoid that,
    rather than fighting its output with a projection.

    It does not currently survive contact with the solve. See
    :attr:`CatheterRodSpec.rest_curvature_from_path` for the measurements: at
    full strength the rod diverges to ten times its rest length, and the
    curvature has to be scaled down to where it is effectively straight before
    the solve is stable. Kept because the seeding itself is correct and tested,
    and it is the shape of the fix once the solve can take it.

    The frames come from the seeded polyline rather than off the solver. Neither
    of the solver's own buffers can supply them at build time: ``orientations``
    is initialized to one frame for the whole rod, so consecutive frames are
    identical and their relative Darboux is zero, and ``positions`` is still
    empty because the rod syncs particles from the Newton state on each step.
    Seeding from either wrote back exactly the straight rest shape this is meant
    to replace, and measured identically to not running at all.

    Tip edges are left as they are. Their rest curvature is the steerable
    pre-bend that ``set_tip_bend`` owns, and a floppy tip free to deform is what
    lets the wire be steered at all.

    Args:
        solver: Built rod solver, single-env or batched.
        num_tip_edges: Trailing edges to leave to the tip.
        segment_length_m: Rest length of one edge.
        positions_world_m: ``(num_points, 3)`` seeded particle positions, in
            order from proximal to distal.
        scale: Fraction of the polyline's curvature to adopt. ``1.0`` asks for a
            rod whose relaxed shape is the vessel exactly, which makes the wire
            as reluctant to straighten as the anatomy is curved; below that it
            is a blend, and the rod still carries some preference for straight.
    """
    import torch
    import warp as wp

    seeded = torch.as_tensor(np.asarray(positions_world_m, dtype=np.float32)).reshape(-1, 3)
    full_curvature = rest_darboux_along_polyline(seeded, segment_length_m) * float(scale)

    # Both workspaces can exist, and the batched solve reads only the batched
    # buffers, so writing one of them alone would silently do nothing.
    for workspace, edges_attr in ((solver._bws, "num_edges_per_rod"), (solver._ws, "num_edges")):
        if workspace is None:
            continue
        num_edges = int(getattr(workspace, edges_attr, 0) or 0)
        shaft = num_edges - max(int(num_tip_edges), 0)
        if num_edges <= 0 or shaft <= 0:
            continue
        rest = wp.to_torch(workspace.rest_darboux)
        # Env count from the buffers rather than an attribute, whose name differs
        # between the two workspaces.
        num_envs = max(rest.shape[0] // num_edges, 1)
        rest = rest.view(num_envs, num_edges, 3)
        if full_curvature.shape[0] < shaft:
            raise RuntimeError(
                f"polyline gives {full_curvature.shape[0]} edge curvatures, " f"too few for {shaft} shaft edges"
            )
        rest[:, :shaft] = full_curvature[:shaft].to(device=rest.device, dtype=rest.dtype)


@dataclass
class CatheterRodSpec:
    """Scene-level description of the catheter rod and its vessel.

    Attributes:
        num_envs: Independent environments. Each gets its own rod and, when a
            vessel is present, its own deformable wall.
        device: Warp device string shared by the rod, the vessel, and Newton.
        origin_world_m: Proximal end of the catheter in Isaac world metres.
        track_direction_world: Insertion direction at the origin.
        length_m: Initial catheter length.
        num_segments: Rod segments; particles per environment is this plus one.
        radius_m: Catheter radius, used for both contact and rendering.
        patient_twin_manifest: Twin supplying the vessel centerline. Without
            one the rod runs against no wall.
        vessel_enabled: Set ``False`` to run the rod with contact off even when
            a twin is available, which is useful for isolating rod behaviour.
        gravity_world: World-frame gravity, Z-up by default.
        rigid_bodies_enabled: Set ``True`` when the scene also spawns rigid
            articulations, such as a robot arm carrying the catheter drive.
            The rod-only solver has no rigid integrator, so a scene with an
            arm needs the coupled MJWarp + XPBD manager instead.
        drive_body_name: Name of the rigid body holding the catheter's proximal
            end, such as an arm flange. Setting it turns on two-way coupling, so
            the holder feels the catheter's weight, stiffness, buckling and
            vessel contact instead of only pushing it. Requires
            ``rigid_bodies_enabled``, since a rod-only scene has no body to push
            on. Left unset, contact stays one-way.
        track_guidance: Hold the shaft on ``initial_path_world_m`` during the
            solve, leaving the distal tip free. Containment only tests whether a
            particle is inside the lumen, never where along it the particle
            belongs, so without this the shaft is free to slide and bunch:
            measured chords run from 3% to 380% of rest. Off by default because
            it prescribes the route the wire takes, which is only the intended
            behaviour when that route is known.
        track_stage: ``"pre"`` or ``"post"`` — which side of the constraint
            solve guidance runs on. ``"pre"`` is the useful one, since the
            distance constraints then get the last word and are what restore the
            spacing; ``"post"`` leaves guidance as the final say on position.
        rest_curvature_from_path: Take the bend constraint's rest curvature from
            the shape the rod was seeded in, instead of leaving it zero. Zero is
            a straight rest shape, which a direct constraint solve then
            straightens the rod into regardless of what the anatomy does; this
            asks the solve for a rod that follows the vessel to begin with.
            Requires ``initial_path_world_m``. The distal tip keeps its own rest
            pre-bend so it stays steerable.

            Off because it does not currently hold together: on the s0011 iliac
            route the constraint solve diverges, stretching the rod to 2.5-3.4 m
            against a 303 mm rest length. Scaling the curvature down only trades
            that for uselessness -- stable near ``0.05``, which is a 1.7 m radius
            and leaves 40 of 41 particles outside the lumen exactly as before.
            Making it work needs the solve to tolerate an anatomically curved
            rest shape, which is a solver-side change.
        containment_cleanup_iterations: Gauss-Seidel distance sweeps to run
            *after* post-solve containment, restoring the edge lengths that
            containment disturbs. Containment has to run after the solve to
            persist at all, but it is then the last word on position and
            overrides the stretch constraints the solve had satisfied, leaving
            chords at 7-364% of rest -- particles overlapping in clusters with
            long stretched gaps between them.

            Measured on the s0011 iliac route: 4 sweeps give 91-159%, 16 give
            100-119%, 32 give 100-112% and 64 give 100-107%. The cost is that
            worst-case wall penetration rises from 1.5 mm to about 3.2 mm, and
            notably it plateaus there rather than growing with sweep count, so
            more sweeps buy spacing accuracy without giving up more containment.
            ``0`` disables it.

            That measurement was taken over a short insertion. Stretch is not a
            fixed offset the sweeps pay off once -- it accumulates with inserted
            length, so a teleop run that drives most of the route reaches
            121-122% on the setting that measured 112%. Driven to a matched
            depth, 128 sweeps measured 100-109% against 64's 100-114% at the
            same containment, which is why it is the default.

            Sweeps alone cannot finish the job, and it is worth knowing why
            before reaching for a bigger number. They redistribute the excess
            arc length; they do not remove it. Pushing them harder pulls the
            correction back off the free distal tip and loads it into the
            mid-shaft instead, so the bend relocates rather than leaves. What
            lets it exist at all is one-sided containment -- see
            ``containment_interior_deadband``.
            ``I4H_CATHETER_CLEANUP`` walks this live against the probe.
        containment_cleanup_relaxation: Fraction of each length correction to
            apply per sweep, in ``(0, 1]``. Full strength converges fastest and
            is the default; 0.6 measured slightly worse at equal sweeps.
        containment_cleanup_rounds: How many times to alternate containment with
            the cleanup sweeps, rather than running containment once and
            spending every sweep after it.

            Sequencing them lets whichever ran last win, and the sweeps ran
            last. They equalize edge lengths knowing nothing about the vessel,
            so they pay for spacing by pushing particles laterally through the
            wall -- and 128 of them comfortably overrule the single containment
            pass before them. Measured that way: chords at a best-ever 100-104%
            with 13 of 41 particles outside the lumen, the worst containment
            recorded on this route. Alternating projects onto each constraint in
            turn, which converges toward satisfying both instead of only the
            last. The sweep budget is divided across rounds rather than
            multiplied, so this costs extra containment passes and no extra
            sweeps. ``1`` restores the original sequencing.
        containment_interior_deadband: How much of the radius the wire is free to
            occupy before containment starts pulling it back toward the vessel
            axis, as a fraction. ``1.0`` reaches the wall and is the one-sided
            behaviour, where a particle inside the lumen is corrected by nothing.

            One-sided containment is why sweeping harder relocates the bend
            rather than removing it. Excess arc length has to go somewhere, and
            with no preference for the axis anywhere inside the lumen, sideways
            is free -- so the sweeps only choose whether the tip or the shaft
            carries it. A deadband takes the free lateral room away without
            prescribing a route: inside it the shape is still the solve's answer.
        containment_interior_stiffness: How hard the interior pull acts per
            collision iteration, in ``[0, 1]``. ``0`` restores one-sided
            containment. At ``1`` a sample lands exactly on the deadband
            surface, so no value here can carry one past it toward the axis.
            ``I4H_CATHETER_INTERIOR`` walks both live against the probe.
        rest_curvature_scale: Fraction of the centerline's curvature to adopt as
            rest shape, when ``rest_curvature_from_path`` is on. Below ``1.0``
            the rod keeps some preference for straightening, which is what a
            real guidewire in a curved vessel has.
        track_stiffness: Blend toward the path per iteration, in ``[0, 1]``. A
            blend rather than a snap, so guidance argues with the elastic solve
            instead of overruling it as the hard containment projection does.
        physical_rotational_inertia: Write the rod's own segment inertia over the
            solver's identity default, which is physically wrong -- a 0.5 mm
            wire's frame is orders of magnitude easier to turn than a unit
            inertia claims.

            Off nonetheless, because it buys nothing measurable and the shipped
            configuration is the one that has been validated without it. It was
            expected to make ``containment_stage="pre"`` viable by letting the
            frame turn under a wall push and so hold a pre-solve correction. It
            does not: with it on, ``"pre"`` still measures +74 mm and 39 of 41
            particles outside, and ``"post"`` still measures +3 mm, both
            unchanged from identity inertia.
        containment_stage: ``"pre"`` or ``"post"`` — whether the lumen is
            enforced before or after the rod's constraint solve. The solve is
            direct, so whichever runs second wins: ``"post"`` keeps the wire
            inside the vessel but leaves the chord lengths it mangled, while
            ``"pre"`` preserves rest lengths and lets the solve carry particles
            back out through the wall.

            ``"pre"`` has been measured against every lever that looked
            relevant and none of them move it: physical rotational inertia,
            fully quasi-static damping, and a sparse centerline attraction all
            leave it at +74 mm with 39-40 of 41 particles outside. The reason is
            structural rather than a matter of tuning -- the rest shape is
            straight and containment is one-sided, acting on a particle only
            once it is already outside the wall, so the inward shove and the
            straightening solve simply balance well outside the lumen. Shaping
            the rod to the anatomy needs the solve itself to accept a curved
            rest configuration, which is solver-side work.
        tip_bend_fraction: Bend stiffness of the most distal edge as a fraction
            of the shaft's, cosine-blended over the tip. One leaves the rod
            uniform. Relieving the tip is how a real J-tip wire is built, and
            it lets the tip deflect off a wall rather than levering the shaft
            out of the lumen.
        soft_contact_overrides: Extra fields forwarded to the coupled solver
            cfg, which is where the particle-shape contact material lives.
        solver_overrides: Extra fields forwarded to ``XPBDRodSolverCfg``.
    """

    num_envs: int = 1
    device: str = "cuda:0"
    origin_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    track_direction_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    length_m: float = 0.4
    num_segments: int = DEFAULT_NUM_SEGMENTS
    radius_m: float = 0.0005
    patient_twin_manifest: str | None = None
    vessel_enabled: bool = True
    gravity_world: tuple[float, float, float] = GRAVITY_WORLD_Z_UP
    rigid_bodies_enabled: bool = False
    drive_body_name: str | None = None
    initial_path_world_m: tuple[tuple[float, float, float], ...] | None = None
    lumen_radii_m: tuple[float, ...] | None = None
    physical_rotational_inertia: bool = False
    containment_stage: str = "post"
    track_guidance: bool = False
    track_stage: str = "pre"
    track_stiffness: float = 0.35
    rest_curvature_from_path: bool = False
    rest_curvature_scale: float = 1.0
    containment_cleanup_iterations: int = 128
    containment_cleanup_relaxation: float = 1.0
    containment_cleanup_rounds: int = 8
    containment_interior_deadband: float = 0.5
    containment_interior_stiffness: float = 0.25
    tip_bend_fraction: float = 1.0
    soft_contact_overrides: dict[str, Any] = field(default_factory=dict)
    solver_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.num_segments < 1:
            raise ValueError(f"num_segments must be positive, got {self.num_segments}")
        if self.length_m <= 0.0:
            raise ValueError(f"length_m must be positive, got {self.length_m}")
        if self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")
        if self.drive_body_name is not None and not self.rigid_bodies_enabled:
            raise ValueError(
                f"drive_body_name={self.drive_body_name!r} asks the catheter's holder to feel "
                "its reaction, but rigid_bodies_enabled is False, so the scene has no rigid "
                "body to push on. Set rigid_bodies_enabled, or leave drive_body_name unset to "
                "keep contact one-way."
            )
        if self.containment_stage not in ("pre", "post"):
            raise ValueError(f"containment_stage must be 'pre' or 'post', got {self.containment_stage!r}")
        if not 0.0 < self.tip_bend_fraction <= 1.0:
            raise ValueError(f"tip_bend_fraction must be in (0, 1], got {self.tip_bend_fraction}")
        if self.lumen_radii_m is not None:
            if self.initial_path_world_m is None:
                raise ValueError("lumen_radii_m describes initial_path_world_m, which is unset")
            if len(self.lumen_radii_m) != len(self.initial_path_world_m):
                raise ValueError(
                    f"lumen_radii_m must have one entry per path sample, got "
                    f"{len(self.lumen_radii_m)} for {len(self.initial_path_world_m)}"
                )
            if min(self.lumen_radii_m) <= 0.0:
                raise ValueError("lumen_radii_m must be positive")
        direction = np.asarray(self.track_direction_world, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("track_direction_world must be non-zero")
        self.track_direction_world = tuple(float(value) for value in direction / norm)

    @property
    def num_points(self) -> int:
        return int(self.num_segments) + 1

    @property
    def segment_length_m(self) -> float:
        return float(self.length_m) / float(self.num_segments)

    @property
    def wants_vessel(self) -> bool:
        return bool(self.vessel_enabled) and self.patient_twin_manifest is not None

    def initial_positions(self) -> np.ndarray | None:
        """Rest positions along the vessel path, or ``None`` for a straight rod.

        Seeding the rod's shape from the centerline is initialization, not
        control: the catheter starts inside the lumen and XPBD takes over from
        there. Resampling it every step instead would overwrite the solver's
        own result and leave contact with nothing to act on.
        """
        if self.initial_path_world_m is None:
            return None
        from i4h_arena.medical.centerline import sample_polyline

        path = np.asarray(self.initial_path_world_m, dtype=np.float32)
        distances = np.linspace(0.0, float(self.length_m), self.num_points)
        return np.asarray(sample_polyline(path, distances), dtype=np.float32)


def rod_solver_cfg(spec: CatheterRodSpec) -> Any:
    """Build the ``XPBDRodSolverCfg`` that selects the rod manager.

    Containment against the deformable wall runs in the rod solver's own
    kernels, so the static-mesh collision path stays off; leaving both on would
    apply two independent wall constraints to the same catheter.

    ``sync_from_state`` stays on, and is what places the catheter in the
    patient. The rod solver builds itself as a straight rod along +X and has no
    way to start from a polyline, so the centerline reaches it only because the
    Newton builder is seeded with it and the solver reads that buffer back on
    its first step. Turning the sync off strands the catheter at the solver's
    default pose, well outside the anatomy.
    """
    from catheter_vasculature_solver.isaaclab_integration import XPBDRodSolverCfg

    fields: dict[str, Any] = {
        "num_segments": int(spec.num_segments),
        "segment_length": spec.segment_length_m,
        "radius": float(spec.radius_m),
        "gravity": tuple(float(value) for value in spec.gravity_world),
        "collision_enabled": False,
        "track_enabled": False,
    }
    # Refining the rod is not stiffness-neutral here, so a spec that has moved
    # off the reference segment count gets its bend stiffness rescaled to
    # describe the same physical catheter. Without this, asking for three times
    # the segments quietly asks for a wire nine times floppier, and the extra
    # resolution reads as a buckling regression. See
    # :func:`mesh_invariant_bend_stiffness`. At the reference count the field is
    # left alone rather than written back at its own default, so the shipped rod
    # is bit-for-bit what it was. An explicit override still wins, since
    # ``solver_overrides`` is applied last.
    if int(spec.num_segments) != REFERENCE_NUM_SEGMENTS:
        fields["bend_stiffness"] = mesh_invariant_bend_stiffness(
            REFERENCE_BEND_STIFFNESS,
            float(spec.length_m) / float(REFERENCE_NUM_SEGMENTS),
            spec.segment_length_m,
        )
    fields.update(spec.solver_overrides)
    return XPBDRodSolverCfg(**fields)


def coupled_solver_cfg(spec: CatheterRodSpec) -> Any:
    """Build the coupled MJWarp + XPBD rod cfg for a scene that has an arm.

    The rod solver integrates particles and nothing else, so as soon as the
    scene spawns an articulation the model needs a rigid integrator beside it.
    This cfg nests both and leaves substep order to the coupled manager, which
    runs collision, then MJWarp, then the rod.

    ``coupling_mode`` follows ``spec.drive_body_name``. Without one it stays
    ``one_way``, where rigid bodies push the catheter but never feel it. Naming
    the body that holds the proximal end selects ``two_way``, which also feeds
    the catheter's reaction back into ``body_f`` so the driving arm loads up
    against the wire's weight, stiffness and vessel contact. The body is passed
    by name rather than index because the solver resolves it against the Newton
    builder's labels, and an index would quietly mean a different link if the
    scene's body order ever changed.
    """
    from catheter_vasculature_solver.isaaclab_integration import CoupledMJWarpXPBDRodSolverCfg
    from isaaclab_newton.physics import MJWarpSolverCfg

    two_way = spec.drive_body_name is not None
    fields: dict[str, Any] = {
        "coupling_mode": "two_way" if two_way else "one_way",
        "rigid_solver_cfg": MJWarpSolverCfg(),
        "rod_solver_cfg": rod_solver_cfg(spec),
    }
    if two_way:
        fields["drive_body_name"] = spec.drive_body_name
    fields.update(spec.soft_contact_overrides)
    return CoupledMJWarpXPBDRodSolverCfg(**fields)


def newton_solver_cfg(spec: CatheterRodSpec) -> Any:
    """Pick the rod-only or coupled solver cfg to match what the scene spawns."""
    if spec.rigid_bodies_enabled:
        return coupled_solver_cfg(spec)
    return rod_solver_cfg(spec)


def newton_physics_cfg(spec: CatheterRodSpec) -> Any:
    """Build the ``NewtonCfg`` for a catheter scene.

    ``class_type`` is deliberately not set: ``NewtonCfg`` derives it from
    ``solver_cfg.class_type`` and rejects a manual value. That is also how the
    coupled manager gets selected: the cfg carries its own manager, so adding an
    arm changes the solver cfg and nothing else here.

    CUDA graph capture is disabled whenever a deformable vessel is in play. The
    vessel's containment allocates and resizes contact scratch as the catheter
    advances, which a captured graph cannot express. An arm disables it too,
    because MJWarp's contact counts vary with the arm's pose.
    """
    from isaaclab_newton.physics import NewtonCfg

    return NewtonCfg(
        solver_cfg=newton_solver_cfg(spec),
        use_cuda_graph=not (spec.wants_vessel or spec.rigid_bodies_enabled),
    )


class CatheterRodHandle:
    """Registers the rod with Newton and exposes what the scene needs after.

    The particle range and the built rod are only known once ``MODEL_INIT`` has
    fired, so readers must go through the properties rather than caching.
    """

    def __init__(self, spec: CatheterRodSpec):
        self._spec = spec
        self._particle_range: Any = None
        self._rod: Any = None
        self._vessel: Any = None
        self._callback: Any = None

    @property
    def spec(self) -> CatheterRodSpec:
        return self._spec

    @property
    def particle_range(self) -> Any:
        if self._particle_range is None:
            raise RuntimeError(
                "the rod has not been registered yet; PhysicsEvent.MODEL_INIT fires during "
                "scene construction, so read this only after the simulation is built"
            )
        return self._particle_range

    @property
    def rod(self) -> Any:
        if self._rod is None:
            raise RuntimeError("the rod has not been built yet; see particle_range")
        return self._rod

    @property
    def vessel(self) -> Any:
        """Deformable vessel runtime, or ``None`` when running without a wall."""
        return self._vessel

    def report_containment(self, positions_world_m: Any) -> dict[str, float] | None:
        """Containment and chord state for one environment's particles.

        ``None`` when the scene has no centerline to measure against, which is
        the straight-track case rather than a failure.
        """
        spec = self._spec
        if spec.initial_path_world_m is None or spec.lumen_radii_m is None:
            return None
        return containment_report(
            positions_world_m,
            path_world_m=np.asarray(spec.initial_path_world_m, dtype=np.float64),
            lumen_radii_m=np.asarray(spec.lumen_radii_m, dtype=np.float64),
            segment_length_m=float(spec.segment_length_m),
        )

    def reset(self, env_ids: Any = None) -> None:
        """Restore the listed environments in place.

        Rebuilding the solver instead would reallocate every buffer, which
        invalidates any captured CUDA graph and throws away the vessel's
        deformation state along with the rod's.

        The device index tensor IsaacLab builds is forwarded as-is; the solver
        brings it to the host itself.
        """
        if self._rod is None:
            return
        self._rod.reset(env_ids)

    def install(self) -> "CatheterRodHandle":
        """Subscribe to ``MODEL_INIT`` so the rod joins the model before finalize."""
        from isaaclab.physics import PhysicsEvent
        from isaaclab_newton.physics import NewtonManager

        _set_active_handle(self)
        self._callback = NewtonManager.register_callback(
            self._on_model_init,
            PhysicsEvent.MODEL_INIT,
            name="catheter_rod_particles",
            # The rod holds a reference to its own solver and vessel, so letting
            # this be collected as a weak ref would drop the registration.
            wrap_weak_ref=False,
        )
        return self

    def _on_model_init(self, _payload: Any = None) -> None:
        from catheter_vasculature_solver.isaaclab_integration import (
            NewtonXPBDRodManager,
            add_catheter_rod_to_builder,
            rod_config_from_solver_cfg,
        )
        from isaaclab_newton.physics import NewtonManager

        spec = self._spec
        solver_cfg = rod_solver_cfg(spec)
        # The rod's device is carried by RodConfig rather than the solver
        # constructor, and it has to match the Newton model's device.
        rod_config = rod_config_from_solver_cfg(solver_cfg, device=spec.device)

        builder = NewtonManager._builder
        if builder is None:
            raise RuntimeError(
                "MODEL_INIT fired with no Newton ModelBuilder, so the rod's particles have " "nowhere to go"
            )
        self._particle_range = add_catheter_rod_to_builder(
            builder,
            rod_config,
            positions=spec.initial_positions(),
            start=np.asarray(spec.origin_world_m, dtype=np.float32),
            direction=np.asarray(spec.track_direction_world, dtype=np.float32),
            num_envs=spec.num_envs,
        )
        self._rod = self._build_rod(rod_config, solver_cfg)
        NewtonXPBDRodManager.register_rod(self._particle_range, rod=self._rod)

    def _build_rod(self, rod_config: Any, solver_cfg: Any) -> Any:
        """Build the rod solver, with a deformable vessel when the twin has one."""
        from catheter_vasculature_solver import CathRodSolver

        spec = self._spec
        self._vessel = self._build_vessel()
        solver = CathRodSolver(
            rod_config,
            num_envs=spec.num_envs,
            collision_mesh=None,
            track_start=np.asarray(spec.origin_world_m, dtype=np.float32),
            track_dir=np.asarray(spec.track_direction_world, dtype=np.float32),
            track_length=float(spec.length_m),
            tip_num_edges=int(solver_cfg.tip_num_edges),
            particle_radius=float(spec.radius_m),
            segment_length=spec.segment_length_m,
            # Static-mesh collision stays off; the deformable centerline
            # supplies containment. Guidance is the separate question of where
            # along that centerline the shaft sits, which containment does not
            # answer, so it follows the path the rod was seeded on.
            collision_enabled=False,
            track_enabled=bool(spec.track_guidance),
            track_path=self._track_path(),
            track_stage=spec.track_stage,
            track_stiffness=float(spec.track_stiffness),
            centerline_runtime=self._vessel,
            centerline_containment_stage=spec.containment_stage,
            containment_cleanup_iterations=int(spec.containment_cleanup_iterations),
            containment_cleanup_rounds=int(cleanup_rounds_override() or spec.containment_cleanup_rounds),
            containment_cleanup_relaxation=float(spec.containment_cleanup_relaxation),
        )
        if spec.physical_rotational_inertia:
            _apply_physical_rotational_inertia(
                solver,
                radius_m=float(spec.radius_m),
                segment_length_m=float(spec.segment_length_m),
            )
        if spec.tip_bend_fraction < 1.0:
            _taper_tip_bend_stiffness(
                solver,
                num_tip_edges=int(solver_cfg.tip_num_edges),
                tip_fraction=spec.tip_bend_fraction,
            )
        seeded_positions = spec.initial_positions()
        if spec.rest_curvature_from_path and seeded_positions is not None:
            _seed_shaft_rest_darboux(
                solver,
                num_tip_edges=int(solver_cfg.tip_num_edges),
                segment_length_m=float(spec.segment_length_m),
                positions_world_m=seeded_positions,
                scale=float(spec.rest_curvature_scale),
            )
        return solver

    def _track_path(self) -> np.ndarray | None:
        """The route guidance holds the shaft on, or ``None`` when it is off.

        The same ordered centerline the rod is seeded from, so guidance keeps the
        shaft on the path it started on rather than introducing a second,
        disagreeing notion of where the vessel runs.
        """
        spec = self._spec
        if not spec.track_guidance or spec.initial_path_world_m is None:
            return None
        return np.asarray(spec.initial_path_world_m, dtype=np.float32)

    def _build_vessel(self) -> Any:
        if not self._spec.wants_vessel:
            return None
        from i4h_arena.medical.patient_twin import PatientTwin
        from i4h_arena.medical.vessel_deformation import centerline_vessel_from_twin

        deadband = self._spec.containment_interior_deadband
        stiffness = self._spec.containment_interior_stiffness
        override = interior_containment_override()
        if override is not None:
            deadband, stiffness = override
        vessel = centerline_vessel_from_twin(
            PatientTwin.load(self._spec.patient_twin_manifest),
            device=self._spec.device,
            num_envs=self._spec.num_envs,
            catheter_radius_m=self._spec.radius_m,
            interior_deadband=float(deadband),
            interior_stiffness=float(stiffness),
        )
        if vessel is None:
            raise ValueError(
                f"{self._spec.patient_twin_manifest} has no centerline artifacts, so the "
                "deformable vessel cannot be built; pass vessel_enabled=False to run without one"
            )
        return vessel


# Isaac Lab's physics managers are process-wide classmethod singletons, and the
# scene entity that renders the catheter is built from a config by the scene
# loader, so it has no constructor argument to receive the handle through. One
# active handle per process matches the manager it wraps.
_ACTIVE_HANDLE: CatheterRodHandle | None = None


def _set_active_handle(handle: CatheterRodHandle | None) -> None:
    global _ACTIVE_HANDLE
    _ACTIVE_HANDLE = handle


def active_handle() -> CatheterRodHandle | None:
    """Return the installed catheter rod handle, or ``None`` before install."""
    return _ACTIVE_HANDLE


def require_active_handle() -> CatheterRodHandle:
    handle = active_handle()
    if handle is None:
        raise RuntimeError(
            "no catheter rod is installed; the scene must call "
            "CatheterRodHandle(spec).install() before the simulation is built"
        )
    return handle


__all__ = [
    "GRAVITY_WORLD_Z_UP",
    "CatheterRodHandle",
    "CatheterRodSpec",
    "active_handle",
    "coupled_solver_cfg",
    "newton_physics_cfg",
    "newton_solver_cfg",
    "require_active_handle",
    "rod_solver_cfg",
]
