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

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Light: the vessel adapter defers its own solver import, so naming the wall
# defaults here keeps them from drifting apart from the builder that uses them.
from i4h_arena.medical.vessel_deformation import (
    VESSEL_ANGULAR_DAMPING,
    VESSEL_ENDPOINTS_LOCKED,
    VESSEL_LINEAR_DAMPING,
    VESSEL_RESPONSE,
)

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
REST_CURVATURE_ENV_VAR = "I4H_CATHETER_REST_CURVATURE"
TIP_EDGES_ENV_VAR = "I4H_CATHETER_TIP_EDGES"
TIP_LENGTH_ENV_VAR = "I4H_CATHETER_TIP_LENGTH_MM"
CONTACT_ITERATIONS_ENV_VAR = "I4H_CATHETER_CONTACT_ITERATIONS"
VESSEL_COMPLIANCE_ENV_VAR = "I4H_CATHETER_VESSEL"

BEND_STIFFNESS_ENV_VAR = "I4H_CATHETER_BEND"

#: Spatial resolution. Material stiffness is independent of this count.
DEFAULT_NUM_SEGMENTS = 120


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


def rest_curvature_override(environ: Any = None) -> float | None:
    """Optional unloaded-shaft curvature scale; zero disables seeding.

    This changes the material rest shape, independently of placement and
    orientation initialization. Full seeding is a diagnostic control unless
    the intended catheter is actually manufactured with the vessel's shape.
    """
    raw = (environ if environ is not None else os.environ).get(REST_CURVATURE_ENV_VAR, "")
    try:
        scale = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return scale if 0.0 <= scale <= 1.0 else None


def tip_bend_rest_component(polyline_turn_rad: float, num_tip_edges: int) -> float:
    """Per-edge rest Darboux value realizing a given turn of the tip polyline.

    Mirrors what the solver's tip-bend kernels write, so the mapping can be
    checked without a GPU. See
    :meth:`catheter_vasculature_solver.cath_rod_solver.CathRodSolver.set_tip_bend`
    for the derivation of the ``2n - 1``; the short version is that
    ``rest_darboux`` turns *frames* while the polyline follows the midpoints
    between them, losing half a hinge at the tip.

    Args:
        polyline_turn_rad: Requested turn of the tip polyline, in radians.
        num_tip_edges: Edges the bend is spread over.

    Returns:
        The value written into each tip edge's local-X rest curvature.
    """
    if num_tip_edges < 1:
        raise ValueError(f"num_tip_edges must be at least 1, got {num_tip_edges}")
    return math.sin(float(polyline_turn_rad) / float(2 * num_tip_edges - 1))


def tip_bend_polyline_turn_rad(rest_component: float, num_tip_edges: int) -> float:
    """The turn a per-edge rest curvature actually produces, inverting the above.

    Exists to measure the mapping rather than restate it: a test that only
    checks ``sin`` against ``asin`` proves nothing about whether the polyline
    turns by the requested amount. Out-of-range components clamp rather than
    raise, matching ``asin`` on a value the solve could have pushed slightly
    past one.
    """
    if num_tip_edges < 1:
        raise ValueError(f"num_tip_edges must be at least 1, got {num_tip_edges}")
    clamped = min(1.0, max(-1.0, float(rest_component)))
    return float(2 * num_tip_edges - 1) * math.asin(clamped)


def max_faithful_tip_bend_rad(num_tip_edges: int) -> float:
    """Largest request the mapping reproduces exactly, ``(2n - 1) pi / 2``.

    Past it ``asin(sin(x))`` folds back and a larger request realizes a smaller
    turn. Generous at the shipped ten edges (29.8 rad) and tight at one
    (``pi / 2``), which is the case worth guarding since the tip band is now
    sweepable down to a single edge.
    """
    if num_tip_edges < 1:
        raise ValueError(f"num_tip_edges must be at least 1, got {num_tip_edges}")
    return float(2 * num_tip_edges - 1) * math.pi / 2.0


def bend_stiffness_override(environ: Any = None) -> float | None:
    """Bend stiffness from ``I4H_CATHETER_BEND``, else ``None``.

    This dimensionless multiplier scales the physical section rigidity EI.
    It is applied after ``solver_overrides`` so a diagnostic sweep can override
    the scene's authored material. Segment length is accounted for by the
    solver's compliance; no mesh-dependent multiplier is needed here.

    Zero and negatives are ignored rather than treated as a floppy rod, since
    the sweep only moves upward and a zero here would silently remove the
    constraint being measured.
    """
    raw = (environ if environ is not None else os.environ).get(BEND_STIFFNESS_ENV_VAR, "")
    try:
        stiffness = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return stiffness if stiffness > 0.0 else None


def tip_edge_count_override(environ: Any = None) -> int | None:
    """Legacy diagnostic edge-count override for the distal tip.

    Prefer a physical tip length, which remains consistent on mesh refinement.
    Zero disables the steerable band for rest-curvature experiments.
    """
    raw = (environ if environ is not None else os.environ).get(TIP_EDGES_ENV_VAR, "")
    try:
        edges = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return edges if edges >= 0 else None


def seeded_rest_curvature_scale(
    *,
    from_path: bool,
    spec_scale: float,
    override: float | None,
) -> float | None:
    """Scale to seed the rest shape at, or ``None`` to leave the rod straight.

    Separated from the solver build so the precedence is testable without a
    solver. An override turns seeding on by itself, since the sweep would
    otherwise need the spec flag flipped first and the environment would only
    be able to change the strength of something already enabled.

    A scale of zero declines seeding however it arrives, which is what makes
    ``0`` the sweep's control arm rather than a request to seed nothing.
    """
    scale = float(spec_scale if override is None else override)
    if scale <= 0.0:
        return None
    return scale if (from_path or override is not None) else None


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


def vessel_compliance_override(environ: Any = None) -> tuple[float, float, float] | None:
    """``(response, linear_damping, angular_damping)`` from ``I4H_CATHETER_VESSEL``.

    Written ``"response,linear_damping,angular_damping"``, all three fractions
    in ``[0, 1]`` -- so ``"0.5,0.01,0.01"`` restates the defaults. They share a
    variable because they trade off against each other: how much of a contact
    correction the wall takes, and how much of the resulting motion survives to
    the next step. Returns ``None`` when unset or unparsable, leaving the
    spec's values alone, since a mistyped diagnostic should not pick the
    physics.
    """
    raw = (environ if environ is not None else os.environ).get(VESSEL_COMPLIANCE_ENV_VAR, "")
    parts = str(raw).split(",")
    if len(parts) != 3:
        return None
    try:
        response, linear_damping, angular_damping = (float(part.strip()) for part in parts)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= value <= 1.0 for value in (response, linear_damping, angular_damping)):
        return None
    return response, linear_damping, angular_damping


# Isaac's world is Z-up, while the standalone rod solver's config defaults to
# Y-down. Naming the world value here keeps a scene that wants weight from
# inheriting a gravity vector pointing sideways.
GRAVITY_WORLD_Z_UP = (0.0, 0.0, -9.81)

# A guidewire in blood is close to neutrally buoyant, so its weight in air is a
# load the real device does not carry -- and one the wall then has to resist.
# The reference endoluminal scene runs with gravity off for the same reason.
GRAVITY_NEUTRAL_BUOYANCY = (0.0, 0.0, 0.0)


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
    """Apply :func:`tip_bend_stiffness_profile` to a built solver's edges.

    Every workspace, because they hold separate allocations and the batched
    solve reads only the batched one. The rod length comes from each workspace's
    own edge count rather than from dividing its buffer by the environment
    count: the single-rod workspace is one rod wide whatever the environment
    count is, so dividing it repeats the profile once per environment along a
    single rod.
    """
    import warp as wp

    from .catheter_initialization import solver_workspaces, workspace_edges_per_env

    for workspace in solver_workspaces(solver):
        stiffness = wp.to_torch(workspace.bend_stiffness)
        total_edges = int(stiffness.shape[0])
        per_env = workspace_edges_per_env(workspace)
        if per_env <= 0 or total_edges % per_env:
            raise RuntimeError(f"{total_edges} edges are not a whole number of {per_env}-edge rods")
        rods = total_edges // per_env
        shaft = stiffness.view(rods, per_env, 3)[0, 0].clone()
        profile = tip_bend_stiffness_profile(1.0, per_env, num_tip_edges, tip_fraction)
        scale = stiffness.new_tensor(profile).unsqueeze(-1)
        stiffness.view(rods, per_env, 3).copy_((shaft * scale).expand(rods, per_env, 3))


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

    Local Z is the material tangent, as in initialization and stretch/shear
    constraints. Use cylinder transverse inertia for local X/Y and polar
    inertia ``m r^2 / 2`` for local Z.

    Returns the inverse inertia written.
    """
    import warp as wp

    from .catheter_initialization import solver_workspaces

    inv_inertia = None
    for workspace in solver_workspaces(solver):
        diagonal = getattr(workspace, "inv_inertia_local_diag", None)
        if diagonal is None:
            continue
        if inv_inertia is None:
            inv_inertia = segment_inverse_inertia(
                _segment_mass_kg(workspace),
                radius_m,
                segment_length_m,
            )
        polar_inverse = 2.0 / (_segment_mass_kg(workspace) * radius_m * radius_m)
        value = wp.vec3(inv_inertia, inv_inertia, polar_inverse)
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


def bend_radii_m(positions_world_m: Any) -> np.ndarray:
    """Radius of curvature at every interior node, in metres.

    The circumradius of each consecutive triple, which is what makes this
    comparable across segment counts: refining the rod leaves the circle through
    three samples of the same physical curve roughly where it was, while a
    per-node turning angle halves. That is the trap the chord percentages set,
    and the reason this is a radius rather than an angle.

    A radius is also the number the wire can be judged against. Anatomical
    curves run tens of millimetres; a guidewire folded on itself turns inside a
    few. Nothing else in :func:`containment_report` can tell those apart,
    because containment is a radial test that a fold staying inside the lumen
    never trips, and a fold does not change arc length either.

    Straight runs have no finite circle through them and come back as ``inf``
    rather than a large number that would read as a gentle bend. Coincident
    samples are ``inf`` for the same reason: no bend is defined there.
    """
    points = np.asarray(positions_world_m, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        return np.empty(0, dtype=np.float64)
    back = points[1:-1] - points[:-2]
    forward = points[2:] - points[1:-1]
    span = points[2:] - points[:-2]
    # ``|back x forward|`` is twice the triangle's area, so the circumradius
    # ``abc / 4A`` is the product of the side lengths over twice this.
    twice_area = np.linalg.norm(np.cross(back, forward), axis=1)
    sides = np.linalg.norm(back, axis=1) * np.linalg.norm(forward, axis=1) * np.linalg.norm(span, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        radii = sides / (2.0 * twice_area)
    return np.where(twice_area > 0.0, radii, np.inf)


def containment_report(
    positions_world_m: Any,
    *,
    path_world_m: Any,
    lumen_radii_m: Any,
    segment_length_m: float,
    kink_radius_m: float = 0.010,
) -> dict[str, float]:
    """How far the rod sits outside the lumen, and what it did to its own chords.

    ``positions_world_m`` is ``(num_points, 3)`` for a single environment.
    Penetration is measured against the wall interpolated at the nearest point on
    the centerline, signed so that negative is clearance and positive is through
    the wall. A particle on the axis of an 8 mm vessel therefore reports -8 mm.

    Chords come back as a percentage of ``segment_length_m`` because an
    inextensible rod stored with unequal chords is the specific damage that
    containment-after-solve does, and a single chord is meaningless in absolute
    mm.

    Summed over the rod it is not meaningless, which is what ``arc_excess_mm``
    is for: how much arc length the rod is carrying beyond its rest length. The
    chord     percentages describe how that excess is distributed and move when the
    cleanup sweeps redistribute it, so they can improve while the rod holds just
    as much excess as before. The total only moves when arc length is actually
    injected or removed, and it is comparable across segment counts, which the
    percentages are not. Insertion that arrives as excess here rather than as
    tip travel is insertion the operator does not get.

    ``min_bend_radius_mm`` is the one figure here that can see a fold. Neither
    penetration nor arc length can: a tip doubled back on itself sits well
    inside the lumen and carries exactly the arc length it had straight, so both
    report a healthy rod while the wire is kinked and the tip has stopped
    tracking. ``min_bend_radius_node`` says where, which is worth having because
    a tight radius at the distal end is a folded tip and the same radius
    mid-shaft is usually just anatomy.

    A minimum over a hundred-odd nodes is dominated by its worst one, though, so
    it cannot say whether the rod holds a single hairpin or is kinked
    throughout. Measured runs sit at the fold floor of half a segment length in
    every sample of every configuration, which is exactly the reading those two
    cases share. ``kinked_nodes`` and ``bend_radius_p05_mm`` separate them:
    a localized fold leaves the count in the low single digits and the
    percentile out at anatomical scale, while a rod-wide problem moves both.
    ``first_kinked_node`` and ``last_kinked_node`` then say whether the kinked
    nodes are one block or scattered, which is the difference between a buckled
    section that swallows insertion and a discretization that creases
    everywhere.

    Args:
        positions_world_m: ``(num_points, 3)`` particle positions.
        path_world_m: ``(num_samples, 3)`` centerline the lumen is measured on.
        lumen_radii_m: Vessel radius at each centerline sample.
        segment_length_m: Rest length of one edge.
        kink_radius_m: Bend radius at or below which a node counts as kinked.
            Defaults to 10 mm, which is under the tightest curve on the s0011
            route -- the raw centerline bottoms out at 13.1 mm and the rod's
            seeded shape at 14.4 mm -- so anatomy alone cannot trip it. In
            absolute metres rather than segment lengths, since the whole point
            is to compare a fold against the vessel rather than against the
            discretization.
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
    arc_length_m = float(chords.sum())
    rest_length_m = float(segment_length_m) * float(chords.shape[0])
    curvature_radii = bend_radii_m(points)
    tightest_m = float(curvature_radii.min()) if curvature_radii.size else float("inf")
    # Interior nodes start at 1, so the argmin shifts to index the rod itself.
    tightest_node = int(curvature_radii.argmin()) + 1 if curvature_radii.size else -1
    # Interpolating between two infinities gives a nan rather than an infinity,
    # and a rod straight enough for its 5th percentile to land there has no
    # kinks worth a number either way, so both collapse to ``inf``.
    spread_m = float("inf")
    if curvature_radii.size:
        with np.errstate(invalid="ignore"):
            percentile = float(np.percentile(curvature_radii, 5.0))
        if np.isfinite(percentile):
            spread_m = percentile
    tight = curvature_radii <= float(kink_radius_m)
    kinked = int(tight.sum())
    # Where those nodes sit, which separates a coil from scattered creases. The
    # count alone cannot: nine adjacent folded nodes are a buckled distal
    # section that swallows insertion, and nine spread over the rod are a
    # discretization problem. Interior nodes start at 1, matching
    # ``min_bend_radius_node``.
    if kinked:
        indices = np.flatnonzero(tight)
        first_kinked, last_kinked = int(indices[0]) + 1, int(indices[-1]) + 1
    else:
        first_kinked, last_kinked = -1, -1
    return {
        "worst_penetration_mm": float(penetration.max()) * 1000.0,
        "particles_outside": int((penetration > 0.0).sum()),
        "num_particles": int(points.shape[0]),
        "chord_min_pct": float(chords.min()) * percent,
        "chord_max_pct": float(chords.max()) * percent,
        "arc_length_mm": arc_length_m * 1000.0,
        "rest_length_mm": rest_length_m * 1000.0,
        "arc_excess_mm": (arc_length_m - rest_length_m) * 1000.0,
        "min_bend_radius_mm": tightest_m * 1000.0,
        "min_bend_radius_node": tightest_node,
        "bend_radius_p05_mm": spread_m * 1000.0,
        "kinked_nodes": kinked,
        "first_kinked_node": first_kinked,
        "last_kinked_node": last_kinked,
        "num_bend_nodes": int(curvature_radii.size),
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


def relative_darboux(q1: torch.Tensor, q2: torch.Tensor, length_m: float) -> torch.Tensor:
    """The Darboux vector between consecutive rod frames, ``2 vec(q1* q2) / L``.

    This is physical curvature in inverse metres. The active XPBD rest buffer
    instead stores dimensionless quaternion components; multiply by L/2 before
    writing there. Quaternions are x, y, z, w.

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


def rest_darboux_along_polyline(positions_world_m: torch.Tensor, segment_length_m: float) -> torch.Tensor:
    """Dimensionless XPBD rest values from local-Z transported rod frames.

    The active constraint subtracts rest_darboux from Im(conj(q0) * q1),
    without dividing by segment length. Physical curvature from
    relative_darboux must therefore be multiplied by L/2 before storage.
    This optional material rest shape is independent of initial placement.
    """
    import torch

    from .catheter_initialization import rod_frames_along_polyline

    if not segment_length_m > 0.0:
        raise ValueError("segment_length_m must be positive")
    frames = torch.as_tensor(
        rod_frames_along_polyline(positions_world_m.detach().cpu().numpy()),
        dtype=positions_world_m.dtype,
        device=positions_world_m.device,
    )
    return relative_darboux(frames[:-1], frames[1:], segment_length_m) * (0.5 * segment_length_m)


def _seed_shaft_rest_darboux(
    solver: Any,
    *,
    num_tip_edges: int,
    segment_length_m: float,
    positions_world_m: np.ndarray,
    scale: float = 1.0,
) -> None:
    """Optionally author the shaft's material rest shape from its seed path.

    Uses the same local-Z frames as initialization and the dimensionless
    quaternion components expected by XPBD. The distal tip keeps its own
    authored rest values. Disabled by default: patient placement alone does
    not imply that the unloaded catheter is shaped like the vessel.
    """
    import torch
    import warp as wp

    from .catheter_initialization import solver_workspaces, workspace_edges_per_env

    seeded = torch.as_tensor(np.asarray(positions_world_m, dtype=np.float32)).reshape(-1, 3)
    full_curvature = rest_darboux_along_polyline(seeded, segment_length_m) * float(scale)

    for workspace in solver_workspaces(solver):
        num_edges = workspace_edges_per_env(workspace)
        shaft = num_edges - max(int(num_tip_edges), 0)
        if num_edges <= 0 or shaft <= 0:
            continue
        rest = wp.to_torch(workspace.rest_darboux)
        rods = max(rest.shape[0] // num_edges, 1)
        rest = rest.view(rods, num_edges, 3)
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
        gravity_world: World-frame gravity. Zero by default, since a guidewire
            in blood is near neutrally buoyant; pass ``GRAVITY_WORLD_Z_UP`` for
            a scene that wants the rod to carry its own weight.
        vessel_endpoints_locked: Anchor the vessel wall's distal end as well as
            its root. Held at the root alone, a wall whose bend stiffness is
            1.0 does not bend under contact so much as swing about that single
            anchor, which is how the probe came to report the whole wire
            outside a lumen the wire had itself carried 15 mm off the anatomy.
        vessel_response: Share of each contact correction the wall absorbs,
            leaving the rest to the catheter. At ``1.0`` the wall yields
            completely and the wire is never actually contained by anything.
        vessel_linear_damping: Wall translational damping, in ``[0, 1]``.
            Undamped, the energy a contact puts into the wall stays there.
        vessel_angular_damping: Wall rotational damping, in ``[0, 1]``.
            ``I4H_CATHETER_VESSEL`` walks the response and both dampings live.
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
        track_free_distal_length_m: Length of shaft nearest the tip that
            guidance leaves free, which is what turns guidance from a whole-
            route prescription into a proximal rail. Unset, guidance holds
            everything but the steerable tip and the wire simply follows the
            seeded path. Set it longer than the route still to be navigated and
            guidance only holds shaft that has already been somewhere, which is
            the part that has no freedom left anyway: the operator keeps full
            authority over every millimetre still ahead of the tip.

            This removes the lateral freedom a proximal fold needs, and real
            procedures get the same effect for free: the wire there is inside
            an introducer and against vessel it already traversed. On its own
            it is not sufficient, because guidance skips the prescribed root --
            see ``proximal_feed_span_m`` for the other half.
        proximal_feed_span_m: Baseline over which the insertion direction is
            measured, from the root toward the tip. Insertion advances the root
            along this direction, so measuring it across one 4 mm segment lets
            a fold at the root aim the very push that deepens it, which is the
            loop that ends attempts here: node 1's bend radius fell from 61 mm
            to 15 mm over 400 steps of pushing, and the shaft then buckled
            rather than advancing. Measured over several centimetres instead,
            the direction barely registers the fold. Unset, the rod's own first
            segment is used, which is the solver's historical behaviour.
        rest_curvature_from_path: Optionally manufacture the shaft's unloaded
            rest shape from the seeded path, leaving the distal tip separate.
            Requires ``initial_path_world_m``. Disabled by default: initial
            placement in a curved vessel does not prescribe material curvature.
            Earlier sweeps used incompatible curvature units and frame axes;
            those divergence measurements do not apply to the corrected helper.
        containment_cleanup_iterations: Legacy position-only length sweeps,
            used only when ``contact_coupling_iterations=0``. Retained for
            controlled comparisons with the previous post-solve pipeline.
        containment_cleanup_relaxation: Fraction of each legacy length
            correction to apply, in (0, 1].
        containment_cleanup_rounds: Legacy alternations between contact and
            position-only length sweeps, dividing the total sweep budget.
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
        physical_rotational_inertia: Use cylinder inertia around the material
            axes. Enabled so wall contact can turn a frame consistently with
            the catheter's mass and dimensions; identity inertia is retained
            only as an explicit legacy comparison.
        containment_stage: Legacy pre/post ordering used when coupling is
            disabled. Coupled iterations reconcile contact and elasticity
            within the solve instead of selecting a final position override.
        drive_mount_local: Grip point in the drive body's local frame, in
            metres. The transmitted wrench is shifted from this point to the
            body's center of mass using its authored mass properties.
        contact_coupling_iterations: Number of alternating live-wall and
            global elastic solves per substep, updating stretch/shear and
            bend/twist together with positions and material frames. Zero
            selects the legacy global solve plus cleanup.
        tip_length_m: Distal steering span in metres, rounded up to whole
            segments. Default 25 mm preserves approximately ten edges at the
            current s0011 resolution. Set from the intended device geometry.
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
    gravity_world: tuple[float, float, float] = GRAVITY_NEUTRAL_BUOYANCY
    rigid_bodies_enabled: bool = False
    drive_body_name: str | None = None
    drive_mount_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_path_world_m: tuple[tuple[float, float, float], ...] | None = None
    lumen_radii_m: tuple[float, ...] | None = None
    physical_rotational_inertia: bool = True
    containment_stage: str = "post"
    track_guidance: bool = False
    track_stage: str = "pre"
    track_stiffness: float = 0.35
    track_free_distal_length_m: float | None = None
    proximal_feed_span_m: float | None = None
    rest_curvature_from_path: bool = False
    rest_curvature_scale: float = 1.0
    containment_cleanup_iterations: int = 128
    containment_cleanup_relaxation: float = 1.0
    containment_cleanup_rounds: int = 8
    containment_interior_deadband: float = 0.5
    containment_interior_stiffness: float = 0.25
    vessel_endpoints_locked: bool = VESSEL_ENDPOINTS_LOCKED
    vessel_response: float = VESSEL_RESPONSE
    vessel_linear_damping: float = VESSEL_LINEAR_DAMPING
    vessel_angular_damping: float = VESSEL_ANGULAR_DAMPING
    tip_bend_fraction: float = 1.0
    tip_length_m: float = 0.025
    contact_coupling_iterations: int = 32
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
        if not math.isfinite(self.tip_length_m) or not 0.0 < self.tip_length_m <= self.length_m:
            raise ValueError("tip_length_m must be finite, positive, and no longer than the catheter")
        for name in ("vessel_response", "vessel_linear_damping", "vessel_angular_damping"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite fraction in [0, 1], got {getattr(self, name)}")
        if (
            int(self.contact_coupling_iterations) != self.contact_coupling_iterations
            or self.contact_coupling_iterations < 0
        ):
            raise ValueError("contact_coupling_iterations must be a nonnegative integer")
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
        if self.track_free_distal_length_m is not None and self.track_free_distal_length_m <= 0.0:
            raise ValueError(f"track_free_distal_length_m must be positive, got {self.track_free_distal_length_m}")
        if self.proximal_feed_span_m is not None and self.proximal_feed_span_m <= 0.0:
            raise ValueError(f"proximal_feed_span_m must be positive, got {self.proximal_feed_span_m}")

    @property
    def num_points(self) -> int:
        return int(self.num_segments) + 1

    @property
    def segment_length_m(self) -> float:
        return float(self.length_m) / float(self.num_segments)

    @property
    def wants_vessel(self) -> bool:
        return bool(self.vessel_enabled) and self.patient_twin_manifest is not None

    def initial_positions(self) -> np.ndarray:
        """Initial particle positions along the vessel path or straight entry axis.

        Seeding the rod's shape from the centerline is initialization, not
        control: the catheter starts inside the lumen and XPBD takes over from
        there. Resampling it every step instead would overwrite the solver's
        own result and leave contact with nothing to act on.
        """
        if self.initial_path_world_m is None:
            offsets = np.linspace(0.0, float(self.length_m), self.num_points)[:, None]
            return np.asarray(
                np.asarray(self.origin_world_m) + offsets * np.asarray(self.track_direction_world),
                dtype=np.float32,
            )
        from i4h_arena.medical.centerline import sample_polyline

        path = np.asarray(self.initial_path_world_m, dtype=np.float32)
        distances = np.linspace(0.0, float(self.length_m), self.num_points)
        return np.asarray(sample_polyline(path, distances), dtype=np.float32)


def rod_solver_cfg(spec: CatheterRodSpec) -> Any:
    """Build the ``XPBDRodSolverCfg`` that selects the rod manager.

    Containment against the deformable wall runs in the rod solver's own
    kernels, so the static-mesh collision path stays off; leaving both on would
    apply two independent wall constraints to the same catheter.

    ``sync_from_state`` stays on so externally authored Newton positions and
    velocities are respected. Both the Newton builder and rod workspace are
    initialized from the same positions before the first step.
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
    # The rod solver uses EI/GJ and the half-angle strain's 4/L factor.
    # A dimensionless material multiplier now stays fixed under refinement.
    fields.update(spec.solver_overrides)
    # A physical span survives mesh refinement. Explicit edge counts remain a
    # diagnostic override; a length override takes precedence when both exist.
    tip_length = spec.tip_length_m
    raw_length = os.environ.get(TIP_LENGTH_ENV_VAR)
    if raw_length is not None:
        tip_length = float(raw_length) * 0.001
    if not math.isfinite(tip_length) or not 0.0 < tip_length <= spec.length_m:
        raise ValueError("tip_length_m must be finite, positive, and no longer than the catheter")
    fields.setdefault("tip_num_edges", min(spec.num_segments, max(1, math.ceil(tip_length / spec.segment_length_m))))
    # After ``solver_overrides`` rather than before, because this one is a live
    # diagnostic for a sweep and a scene's authored value would otherwise pin it.
    tip_edges = tip_edge_count_override()
    if tip_edges is not None:
        fields["tip_num_edges"] = tip_edges
    if raw_length is not None:
        fields["tip_num_edges"] = min(spec.num_segments, max(1, math.ceil(tip_length / spec.segment_length_m)))
    if not 0 <= fields["tip_num_edges"] <= spec.num_segments:
        raise ValueError("tip_num_edges must lie between zero and num_segments")
    stiffness = bend_stiffness_override()
    if stiffness is not None:
        fields["bend_stiffness"] = stiffness
        print(f"[catheter bend] stiffness overridden to {stiffness:g}", flush=True)
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
        fields["drive_mount_local"] = spec.drive_mount_local
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

    CUDA graph capture is off unconditionally, because the rod managers reject
    it outright: replay bypasses the Python pending-control latch in the rod's
    step, so a captured graph keeps reissuing the commands from capture time and
    ignores every later one. Capture was already impossible with a deformable
    vessel, whose containment resizes contact scratch as the catheter advances,
    or with an arm, whose MJWarp contact counts vary with its pose; the latch
    extends that to a bare rod, which is the only case this used to allow.
    """
    from isaaclab_newton.physics import NewtonCfg

    return NewtonCfg(solver_cfg=newton_solver_cfg(spec), use_cuda_graph=False)


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
        report = containment_report(
            positions_world_m,
            path_world_m=np.asarray(spec.initial_path_world_m, dtype=np.float64),
            lumen_radii_m=np.asarray(spec.lumen_radii_m, dtype=np.float64),
            segment_length_m=float(spec.segment_length_m),
        )
        if self._vessel is not None:
            from .catheter_diagnostics import tube_surface_gaps_m

            vessel = self._vessel
            gaps = tube_surface_gaps_m(
                positions_world_m,
                vessel.positions_per_env[0],
                vessel.edges.numpy()[: vessel.edges_per_env],
                vessel.radii.numpy()[: vessel.nodes_per_env],
                spec.radius_m,
                open_root=vessel.open_root,
                open_root_neighbor=vessel.open_root_neighbor,
            )
            report.update(
                live_worst_penetration_mm=float(gaps.max()) * 1000.0 if gaps.size else 0.0,
                live_samples_outside=int(np.count_nonzero(gaps > 0.0)),
                num_live_samples=int(gaps.size),
            )
        return report

    @property
    def reference_path_world_m(self) -> np.ndarray | None:
        """Ordered route for measuring tip progress, independent of wall contact."""
        path = self._spec.initial_path_world_m
        return None if path is None else np.asarray(path, dtype=np.float64)

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
        if self._particle_range is not None:
            from isaaclab_newton.physics import NewtonManager

            from .catheter_initialization import publish_reset_state

            publish_reset_state(
                self._rod,
                self._particle_range,
                (NewtonManager.get_state_0(), NewtonManager.get_state_1()),
                env_ids,
            )

    def install(self) -> CatheterRodHandle:
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
        positions = spec.initial_positions()
        self._particle_range = add_catheter_rod_to_builder(
            builder,
            rod_config,
            positions=positions,
            start=np.asarray(spec.origin_world_m, dtype=np.float32),
            direction=np.asarray(spec.track_direction_world, dtype=np.float32),
            num_envs=spec.num_envs,
        )
        self._rod = self._build_rod(rod_config, solver_cfg, positions)
        NewtonXPBDRodManager.register_rod(self._particle_range, rod=self._rod)

    def _build_rod(self, rod_config: Any, solver_cfg: Any, positions: np.ndarray) -> Any:
        """Build the rod solver, with a deformable vessel when the twin has one."""
        from catheter_vasculature_solver import CathRodSolver

        spec = self._spec
        self._vessel = self._build_vessel()
        contact_iterations = int(os.environ.get(CONTACT_ITERATIONS_ENV_VAR, spec.contact_coupling_iterations))
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
            track_free_distal_edges=self._track_free_distal_edges(),
            proximal_feed_span=self._proximal_feed_span(),
            track_path=self._track_path(),
            track_stage=spec.track_stage,
            track_stiffness=float(spec.track_stiffness),
            centerline_runtime=self._vessel,
            centerline_containment_stage=spec.containment_stage,
            containment_cleanup_iterations=int(spec.containment_cleanup_iterations),
            containment_cleanup_rounds=int(cleanup_rounds_override() or spec.containment_cleanup_rounds),
            containment_cleanup_relaxation=float(spec.containment_cleanup_relaxation),
            contact_coupling_iterations=contact_iterations,
        )
        print(
            f"[catheter solver] contact iterations={contact_iterations} "
            f"tip={int(solver_cfg.tip_num_edges)} edges "
            f"({int(solver_cfg.tip_num_edges) * spec.segment_length_m * 1000.0:.2f} mm)",
            flush=True,
        )
        from .catheter_initialization import initialize_rod_state

        initialize_rod_state(solver, positions)
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
        # Resolved here rather than on the spec so both embodiments get the
        # sweep from one place.
        scale = seeded_rest_curvature_scale(
            from_path=bool(spec.rest_curvature_from_path),
            spec_scale=float(spec.rest_curvature_scale),
            override=rest_curvature_override(),
        )
        if scale is not None and spec.initial_path_world_m is not None:
            _seed_shaft_rest_darboux(
                solver,
                num_tip_edges=int(solver_cfg.tip_num_edges),
                segment_length_m=float(spec.segment_length_m),
                positions_world_m=positions,
                scale=scale,
            )
            # Printed because the sweep is only readable after the fact if the
            # log says which scale produced it. Matches the probe's stream so
            # both land in the same run log.
            solver.capture_tip_bend_baseline()
            print(f"[catheter rest curvature] seeded at scale {scale:.2f}", flush=True)
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

    def _proximal_feed_span(self) -> int:
        """Nodes spanned by the insertion direction; ``1`` keeps the first segment.

        Rounded up, so a span shorter than one segment still reaches past the
        adjacent node rather than silently collapsing back onto it.
        """
        spec = self._spec
        if spec.proximal_feed_span_m is None:
            return 1
        span = math.ceil(spec.proximal_feed_span_m / spec.segment_length_m)
        return max(1, min(spec.num_segments, int(span)))

    def _track_free_distal_edges(self) -> int | None:
        """Distal edges guidance leaves free, or ``None`` to keep the tip default.

        Rounded down, so the free window never comes out shorter than asked and
        guidance never reaches further toward the tip than intended.
        """
        spec = self._spec
        if spec.track_free_distal_length_m is None:
            return None
        return min(spec.num_segments, int(spec.track_free_distal_length_m / spec.segment_length_m))

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

        response = self._spec.vessel_response
        linear_damping = self._spec.vessel_linear_damping
        angular_damping = self._spec.vessel_angular_damping
        compliance = vessel_compliance_override()
        if compliance is not None:
            response, linear_damping, angular_damping = compliance

        vessel = centerline_vessel_from_twin(
            PatientTwin.load(self._spec.patient_twin_manifest),
            device=self._spec.device,
            num_envs=self._spec.num_envs,
            catheter_radius_m=self._spec.radius_m,
            interior_deadband=float(deadband),
            interior_stiffness=float(stiffness),
            endpoints_locked=bool(self._spec.vessel_endpoints_locked),
            vessel_response=float(response),
            linear_damping=float(linear_damping),
            angular_damping=float(angular_damping),
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
    "GRAVITY_NEUTRAL_BUOYANCY",
    "GRAVITY_WORLD_Z_UP",
    "VESSEL_COMPLIANCE_ENV_VAR",
    "CatheterRodHandle",
    "CatheterRodSpec",
    "active_handle",
    "coupled_solver_cfg",
    "newton_physics_cfg",
    "newton_solver_cfg",
    "require_active_handle",
    "rod_solver_cfg",
    "vessel_compliance_override",
]
