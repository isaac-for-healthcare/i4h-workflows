# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind navigation geometry before the fluoroscopy sensor's first read."""

from __future__ import annotations

import logging
from typing import Any

from .carm import ReferenceProjectionCArmStateProvider, SceneCArmStateProvider, follow_tip_enabled

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

        carm_provider = ReferenceProjectionCArmStateProvider(
            PatientVolume.load(fluoroscopy.patient_twin),
            env.action_manager.get_term("carm_orbit"),
            detector_size_m=detector_size_m,
            tip_source=catheter if follow_tip_enabled() else None,
        )
    else:
        carm_provider = _scene_data_carm_provider(env, detector_size_m) or SceneCArmStateProvider(
            env.scene["xray_source"],
            env.scene["detector"],
            detector_size_m=detector_size_m,
        )
    fluoroscopy.bind_catheter_provider(catheter)
    fluoroscopy.bind_carm_provider(carm_provider)


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
