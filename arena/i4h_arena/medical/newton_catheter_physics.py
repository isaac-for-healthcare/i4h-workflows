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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Isaac's world is Z-up, while the standalone rod solver's config defaults to
# Y-down. Naming the world value here keeps the scene from inheriting a gravity
# vector pointing sideways.
GRAVITY_WORLD_Z_UP = (0.0, 0.0, -9.81)


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
        solver_overrides: Extra fields forwarded to ``XPBDRodSolverCfg``.
    """

    num_envs: int = 1
    device: str = "cuda:0"
    origin_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    track_direction_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    length_m: float = 0.4
    num_segments: int = 40
    radius_m: float = 0.0005
    patient_twin_manifest: str | None = None
    vessel_enabled: bool = True
    gravity_world: tuple[float, float, float] = GRAVITY_WORLD_Z_UP
    initial_path_world_m: tuple[tuple[float, float, float], ...] | None = None
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
    fields.update(spec.solver_overrides)
    return XPBDRodSolverCfg(**fields)


def newton_physics_cfg(spec: CatheterRodSpec) -> Any:
    """Build the ``NewtonCfg`` for a catheter scene.

    ``class_type`` is deliberately not set: ``NewtonCfg`` derives it from
    ``solver_cfg.class_type`` and rejects a manual value.

    CUDA graph capture is disabled whenever a deformable vessel is in play. The
    vessel's containment allocates and resizes contact scratch as the catheter
    advances, which a captured graph cannot express.
    """
    from isaaclab_newton.physics import NewtonCfg

    return NewtonCfg(
        solver_cfg=rod_solver_cfg(spec),
        use_cuda_graph=not spec.wants_vessel,
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
        return CathRodSolver(
            rod_config,
            num_envs=spec.num_envs,
            collision_mesh=None,
            track_start=np.asarray(spec.origin_world_m, dtype=np.float32),
            track_dir=np.asarray(spec.track_direction_world, dtype=np.float32),
            track_length=float(spec.length_m),
            tip_num_edges=int(solver_cfg.tip_num_edges),
            particle_radius=float(spec.radius_m),
            segment_length=spec.segment_length_m,
            # Static-mesh collision and track guidance stay off; the deformable
            # centerline supplies containment instead.
            collision_enabled=False,
            track_enabled=False,
            centerline_runtime=self._vessel,
        )

    def _build_vessel(self) -> Any:
        if not self._spec.wants_vessel:
            return None
        from i4h_arena.medical.patient_twin import PatientTwin
        from i4h_arena.medical.vessel_deformation import centerline_vessel_from_twin

        vessel = centerline_vessel_from_twin(
            PatientTwin.load(self._spec.patient_twin_manifest),
            device=self._spec.device,
            num_envs=self._spec.num_envs,
            catheter_radius_m=self._spec.radius_m,
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
    "newton_physics_cfg",
    "require_active_handle",
    "rod_solver_cfg",
]
