# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deformable vessel construction from a patient twin manifest.

The twin stores its centerline graph in patient millimetres, while the solver
expects lumen geometry in Isaac world metres. Doing that conversion in one place
keeps the renderer, the collision solver, and the USD anatomy from each picking
their own units, which is the failure the manifest's explicit transforms exist
to prevent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from i4h_arena.medical.patient_twin import PatientTwin

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from catheter_vasculature_solver.vessel_deformation import CenterlineVesselRuntime

# Radii are lengths rather than points, so they scale by the transform's linear
# part instead of going through the full affine.
_MM_TO_M = 0.001

# Wall settings matched to the reference endoluminal scene, omniendo's
# xcath/scenes/s0065.yaml, which drives this same centerline Cosserat vessel
# against this same two-way containment. These four are dimensionless, so they
# carry over despite that scene running at 20x length scale. Its per-iteration
# delta clamps work out to the 5 mm this module already uses.
#: Share of a contact correction the wall absorbs; the catheter takes the rest.
VESSEL_RESPONSE = 0.5
#: Anchor the distal end as well as the root. Held at the root alone, a wall
#: whose bend stiffness is 1.0 does not so much bend under contact as swing
#: about that one anchor, carrying the lumen away from the anatomy with it.
VESSEL_ENDPOINTS_LOCKED = True
#: Wall damping. Undamped, the energy a contact puts into the wall stays there.
VESSEL_LINEAR_DAMPING = 0.01
VESSEL_ANGULAR_DAMPING = 0.01


def vessel_dynamics_fields(
    *,
    endpoints_locked: bool = VESSEL_ENDPOINTS_LOCKED,
    linear_damping: float = VESSEL_LINEAR_DAMPING,
    angular_damping: float = VESSEL_ANGULAR_DAMPING,
) -> dict[str, Any]:
    """Return ``CenterlineDynamicsParams`` fields for the vessel wall.

    Kept apart from the runtime so the boundary conditions and damping can be
    checked without a Warp device, since they are what decide whether the wall
    stays on the anatomy and neither is observable from the rod's own state.

    Args:
        endpoints_locked: Anchor the distal end too, not just the root.
        linear_damping: Wall translational damping, in ``[0, 1]``.
        angular_damping: Wall rotational damping, in ``[0, 1]``.

    Raises:
        ValueError: If either damping falls outside ``[0, 1]``.
    """
    if not 0.0 <= float(linear_damping) <= 1.0:
        raise ValueError(f"linear_damping must be in [0, 1], got {linear_damping}")
    if not 0.0 <= float(angular_damping) <= 1.0:
        raise ValueError(f"angular_damping must be in [0, 1], got {angular_damping}")
    return {
        # The root keeps the vessel registered to the anatomy instead of
        # drifting away under contact, since nothing else constrains it.
        "root_locked": True,
        "endpoints_locked": bool(endpoints_locked),
        "linear_damping": float(linear_damping),
        "angular_damping": float(angular_damping),
    }


def length_scale_from_affine(world_from_patient_m: np.ndarray) -> float:
    """Return the uniform length scale of a patient-to-world transform.

    Args:
        world_from_patient_m: 4x4 affine placing patient metres in the world.

    Raises:
        ValueError: If the transform scales axes unequally. A radius is a single
            scalar per centerline node, so it has no way to express an
            anisotropic lumen, and silently picking one axis would deform the
            vessel differently from the anatomy it is meant to line up with.
    """
    linear = np.asarray(world_from_patient_m, dtype=np.float64)[:3, :3]
    axis_scales = np.linalg.norm(linear, axis=0)
    if not np.allclose(axis_scales, axis_scales[0], rtol=1.0e-5, atol=1.0e-8):
        raise ValueError(
            "world_from_patient_m scales axes unequally "
            f"({axis_scales.tolist()}), so a scalar lumen radius cannot be converted"
        )
    return float(axis_scales[0])


def centerline_data_from_twin(twin: PatientTwin) -> Any | None:
    """Build solver ``CenterlineData`` in world metres, or ``None`` if absent.

    Returns ``None`` when the twin carries no centerline graph, which is the
    normal case for phantom and synthetic scenes; callers then run without a
    deformable vessel rather than failing.
    """
    from catheter_vasculature_solver.vessel_deformation import CenterlineData

    points_path = twin.artifacts.get("centerline_points")
    edges_path = twin.artifacts.get("centerline_edges")
    if points_path is None or edges_path is None:
        return None

    points_world_m = np.asarray(twin.patient_mm_to_world(np.load(points_path)), dtype=np.float32)
    edges = np.asarray(np.load(edges_path), dtype=np.int64).reshape(-1, 2)
    if edges.size and (edges.min() < 0 or edges.max() >= len(points_world_m)):
        raise ValueError(f"centerline_edges index outside centerline_points (0..{len(points_world_m) - 1})")

    radii_path = twin.artifacts.get("centerline_radii")
    if radii_path is None:
        raise ValueError(
            "the deformable vessel needs artifacts.centerline_radii; without lumen radii "
            "there is no wall for the catheter to contact"
        )
    radius_scale = _MM_TO_M * length_scale_from_affine(twin.world_from_patient_m)
    radii_m = np.asarray(np.load(radii_path), dtype=np.float32).reshape(-1) * radius_scale
    if len(radii_m) != len(points_world_m):
        raise ValueError(
            f"centerline_radii has {len(radii_m)} entries but centerline_points has "
            f"{len(points_world_m)}; one radius per node is required"
        )
    if not np.all(radii_m > 0.0):
        raise ValueError("centerline_radii must be strictly positive")

    start_radii = radii_m[edges[:, 0]]
    end_radii = radii_m[edges[:, 1]]
    return CenterlineData(
        starts=points_world_m[edges[:, 0]],
        ends=points_world_m[edges[:, 1]],
        # The tree builder recovers branching from welded node connectivity, so
        # a single branch id keeps every edge eligible to be joined up.
        branch_ids=np.zeros(len(edges), dtype=np.int32),
        start_radius_min=start_radii,
        end_radius_min=end_radii,
        start_radius_max=start_radii,
        end_radius_max=end_radii,
    )


def centerline_vessel_from_twin(
    twin: PatientTwin,
    *,
    device: str,
    num_envs: int = 1,
    catheter_radius_m: float,
    two_way: bool = True,
    vessel_response: float = VESSEL_RESPONSE,
    interior_deadband: float = 1.0,
    interior_stiffness: float = 0.0,
    max_distance_m: float = 0.05,
    catheter_max_delta_m: float = 0.005,
    vessel_max_delta_m: float = 0.005,
    endpoints_locked: bool = VESSEL_ENDPOINTS_LOCKED,
    linear_damping: float = VESSEL_LINEAR_DAMPING,
    angular_damping: float = VESSEL_ANGULAR_DAMPING,
    params: Any = None,
) -> "CenterlineVesselRuntime | None":
    """Build a per-environment deformable vessel from the twin's centerline.

    Every environment gets its own vessel state, so two-way contact in one
    environment cannot move another environment's wall.

    Args:
        twin: Loaded patient twin manifest.
        device: Warp device string; must match the solver's device.
        num_envs: Independent vessel replicas, one per environment.
        catheter_radius_m: Catheter radius, used to inset the contact surface.
        two_way: Let the catheter push the wall back, not just be constrained.
        vessel_response: Scale on the wall's share of a contact correction.
        interior_deadband: Fraction of the free radius the wire may occupy before
            containment starts pulling it back toward the axis. ``1.0`` reaches
            the wall, which is the one-sided behaviour.
        interior_stiffness: How hard that interior pull acts, in ``[0, 1]``.
            ``0.0`` disables it, leaving containment one-sided.
        max_distance_m: Containment search radius.
        catheter_max_delta_m: Per-iteration clamp on catheter corrections.
        vessel_max_delta_m: Per-iteration clamp on wall corrections.
        endpoints_locked: Anchor the wall's distal end as well as its root.
        linear_damping: Wall translational damping, in ``[0, 1]``.
        angular_damping: Wall rotational damping, in ``[0, 1]``.
        params: Optional ``CenterlineDynamicsParams`` override, which replaces
            the boundary conditions and damping above rather than merging.

    Returns:
        The runtime, or ``None`` when the twin carries no centerline graph.
    """
    from catheter_vasculature_solver.vessel_deformation import (
        CenterlineDynamicsParams,
        CenterlineVesselRuntime,
        build_centerline_tree,
    )

    data = centerline_data_from_twin(twin)
    if data is None:
        return None

    tree = build_centerline_tree(data)
    return CenterlineVesselRuntime.from_tree(
        tree,
        device=device,
        num_envs=int(num_envs),
        params=params
        or CenterlineDynamicsParams(
            **vessel_dynamics_fields(
                endpoints_locked=endpoints_locked,
                linear_damping=linear_damping,
                angular_damping=angular_damping,
            )
        ),
        catheter_radius=float(catheter_radius_m),
        max_distance=float(max_distance_m),
        two_way=bool(two_way),
        vessel_response=float(vessel_response),
        interior_deadband=float(interior_deadband),
        interior_stiffness=float(interior_stiffness),
        catheter_max_delta=float(catheter_max_delta_m),
        vessel_max_delta=float(vessel_max_delta_m),
    )


__all__ = [
    "VESSEL_ANGULAR_DAMPING",
    "VESSEL_ENDPOINTS_LOCKED",
    "VESSEL_LINEAR_DAMPING",
    "VESSEL_RESPONSE",
    "centerline_data_from_twin",
    "centerline_vessel_from_twin",
    "length_scale_from_affine",
    "vessel_dynamics_fields",
]
