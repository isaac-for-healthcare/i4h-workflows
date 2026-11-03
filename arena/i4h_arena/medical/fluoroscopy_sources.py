# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind navigation geometry before the fluoroscopy sensor's first read."""

from __future__ import annotations

import logging
from typing import Any

from .carm import (
    PerPatientCArmStateProvider,
    ReferenceProjectionCArmStateProvider,
    SceneCArmStateProvider,
    follow_tip_enabled,
)

logger = logging.getLogger("i4h_arena.scene")


def bind_fluoroscopy_sources(env: Any) -> None:
    """Connect the live catheter and C-arm once per sensor instance.

    Isaac Lab constructs the action manager before the observation manager,
    which reads every observation to infer its shape. Bind at that first read
    so patient-backed rendering also works before a Workflow view exists, as
    in an RL environment. The view calls this too for scenes without image
    observations. Repeated calls preserve the provider's accumulated pan.
    """
    fluoroscopy = env.scene["fluoroscopy"]
    if fluoroscopy.state_providers_bound:
        return

    catheter = env.scene["catheter"]
    detector_size_m = (0.6144, 0.6144)
    if fluoroscopy.patient_twin is not None:
        from .patient_volume import PatientVolume

        def reference(twin: Any) -> ReferenceProjectionCArmStateProvider:
            return ReferenceProjectionCArmStateProvider(
                PatientVolume.load(twin),
                env.action_manager.get_term("carm_orbit"),
                detector_size_m=detector_size_m,
                tip_source=catheter if follow_tip_enabled() else None,
            )

        groups = getattr(fluoroscopy, "patient_groups", ())
        if groups:
            carm_provider = PerPatientCArmStateProvider(
                [reference(twin) for twin, _ in groups],
                [env_ids for _, env_ids in groups],
            )
        else:
            carm_provider = reference(fluoroscopy.patient_twin)
    else:
        carm_provider = _scene_data_carm_provider(env, detector_size_m) or SceneCArmStateProvider(
            env.scene["xray_source"],
            env.scene["detector"],
            detector_size_m=detector_size_m,
        )
    fluoroscopy.bind_catheter_provider(catheter)
    fluoroscopy.bind_carm_provider(carm_provider)
    _bind_vessel_motion(env, fluoroscopy, catheter)


def _bind_vessel_motion(env: Any, fluoroscopy: Any, catheter: Any) -> None:
    """Connect the deformable wall and the contrast run to the renderer, when the scene has them.

    Skipped rather than failed in every case that is simply an absent piece: a phantom scene
    with no centerline, a run without the deformation solver, or a renderer not carving the
    lumen. The frame those produce is the one they produced before.
    """
    if not getattr(fluoroscopy.cfg, "vessel_contrast", False) or fluoroscopy.patient_twin is None:
        return

    from .newton_catheter_physics import active_handle

    handle = active_handle()
    vessel = None if handle is None else handle.vessel
    if vessel is None:
        logger.info("fluoroscopy vessel_contrast is on but the scene has no deformable wall; rendering it at rest")
        return

    from .patient_volume import PatientVolume
    from .vessel_motion import CenterlineVesselMotion, ContrastRun

    try:
        contrast = ContrastRun(PatientVolume.load(fluoroscopy.patient_twin), dt_s=_sensor_dt_s(env, fluoroscopy))
        fluoroscopy.bind_vessel_provider(
            CenterlineVesselMotion(fluoroscopy.patient_twin, vessel, contrast=contrast, tip_source=catheter)
        )
    except ValueError:
        # A mismatch here means the solver and the renderer are holding different centerlines,
        # and compositing one onto the other would put the vessel in the wrong place. Rendering
        # it at rest is wrong too, but visibly and consistently so.
        logger.warning("could not match the deformable wall to the rendered centerline", exc_info=True)


def _sensor_dt_s(env: Any, fluoroscopy: Any) -> float:
    """Simulated time between two reads of the fluoroscopy sensor.

    The bolus is advected in seconds, so reading it at the wrong rate makes contrast arrive at
    the wrong time rather than failing visibly. The sensor's own update period governs when
    frames are produced; a period of zero means every step, which is the environment's own.
    """
    step_dt = float(env.step_dt)
    period = float(getattr(fluoroscopy.cfg, "update_period", 0.0) or 0.0)
    return max(period, step_dt)


def _scene_data_carm_provider(env: Any, detector_size_m: tuple[float, float]) -> Any | None:
    """Prefer mapped scene-data poses, falling back to the two visual assets."""
    try:
        from isaaclab.sim import SimulationContext

        from .newton_providers import SceneDataCArmStateProvider

        provider = SimulationContext.instance().get_scene_data_provider()
        if provider is None:
            return None
        num_envs = int(env.num_envs)
        root = env.scene.env_prim_paths
        return SceneDataCArmStateProvider(
            provider,
            source_paths=[f"{root[index]}/CArm/Orbit/Source" for index in range(num_envs)],
            detector_paths=[f"{root[index]}/CArm/Orbit/Detector" for index in range(num_envs)],
            detector_size_m=detector_size_m,
        )
    except Exception:  # noqa: BLE001 - every construction failure is a fall back
        # Some backends cannot map these visual-only prims. Probe during
        # binding so the first image uses the same fallback as later frames,
        # and preserve the reason in the log when this path is unavailable.
        logger.warning(
            "C-arm SceneDataProvider unavailable; reading the source and detector prims directly",
            exc_info=True,
        )
        return None
