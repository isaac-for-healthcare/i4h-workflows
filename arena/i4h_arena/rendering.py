# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac RTX settings for i4h scenes.

Isaac Lab 3.0 GA removed ``SimulationCfg.render``. RTX settings are process-global
carb settings, applied through Isaac Lab's ``IsaacRtxRendererGlobalSettingsCfg``
either directly or by each camera's renderer when it starts.

Isaac Lab 3.0 GA also changed its rendering experience file: it now enables ambient
occlusion, reflections, indirect diffuse, translucency, cached ray tracing and the
former quality preset by default, which up to v3.0.0-beta2 were off. IsaacLab-Arena
0.3 additionally sets the scene ambient light to zero every time it builds an
environment, so that USD lights alone light its kitchens. i4h scenes were authored,
and their policies trained, under the earlier defaults, so every i4h scene starts
from :data:`SCENE_DEFAULTS` and layers its own settings on top.
A Scene hands the merged settings to every camera renderer, which applies them
as it starts after Arena's override, and adds a ``startup`` event term that applies
them again once the environment exists, for the viewport and for scenes without
cameras such as the RSL-RL path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: What every i4h scene renders with unless it says otherwise: Isaac Lab
#: v3.0.0-beta2's ``apps/isaaclab.python.rendering.kit`` over Isaac Sim's defaults,
#: for every setting that Isaac Lab 3.0 GA's experience file changed, with Arena's
#: zero ambient light undone.
SCENE_DEFAULTS: Mapping[str, Any] = {
    "ambient_light_intensity": 1.0,
    "enable_translucency": False,
    "enable_reflections": False,
    "enable_global_illumination": False,
    "enable_ambient_occlusion": False,
    "enable_cached_raytracing": False,
    "subpixel_mode": 0,
    "dlss_mode": 0,
    "carb_settings": {
        "/rtx/rtpt/cached/enabled": True,
        "/rtx/rtpt/lightcache/cached/enabled": True,
        "/rtx/rtpt/translucency/virtualMotion/enabled": True,
        "/rtx/rtpt/adaptiveSampling/disocclusion/enabled": False,
    },
}

#: Isaac Lab's former ``rendering_mode="quality"``. Isaac Lab up to v3.0.0-beta2
#: loaded ``apps/rendering_modes/quality.kit`` for it; v3.0.0 removed the modes, so
#: the preset's settings are spelled out here.
QUALITY: Mapping[str, Any] = {
    "max_bounces": 3,
    "split_rough_reflection": True,
    "ambient_light_intensity": 1.0,
    "enable_shadows": True,
    "enable_ambient_occlusion": True,
    "ambient_occlusion_denoiser_mode": 0,
    "subpixel_mode": 1,
    "enable_cached_raytracing": True,
    "enable_dlssg": False,
    "dlss_mode": 2,
    "max_samples_per_launch": 1000000,
    "view_tile_limit": 1000000,
    "carb_settings": {
        "/rtx/rtpt/cached/enabled": False,
        "/rtx/rtpt/lightcache/cached/enabled": False,
        "/rtx/rtpt/translucency/virtualMotion/enabled": False,
        "/rtx/rtpt/adaptiveSampling/disocclusion/enabled": True,
        "/rtx/rtpt/adaptiveSampling/disocclusion/spp": 4,
    },
}


#: Isaac Lab's former ``rendering_mode="performance"`` (``apps/rendering_modes/performance.kit``).
PERFORMANCE: Mapping[str, Any] = {
    "max_bounces": 2,
    "split_glass": False,
    "split_clearcoat": False,
    "split_rough_reflection": True,
    "ambient_light_intensity": 1.0,
    "enable_shadows": True,
    "dome_light_upper_lower_strategy": 3,
    "enable_ambient_occlusion": False,
    "ambient_occlusion_denoiser_mode": 1,
    "subpixel_mode": 0,
    "enable_cached_raytracing": False,
    "enable_dlssg": False,
    "enable_dl_denoiser": False,
    "dlss_mode": 0,
    "max_samples_per_launch": 1000000,
    "view_tile_limit": 1000000,
    "carb_settings": {
        "/rtx/rtpt/cached/enabled": False,
        "/rtx/rtpt/lightcache/cached/enabled": False,
        "/rtx/rtpt/translucency/virtualMotion/enabled": False,
        "/rtx/rtpt/useAmbientOcclusionForAmbientLight": False,
    },
}


def merge_rtx_settings(*layers: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge settings layers in order; later layers win, and ``carb_settings`` merge key by key."""
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            if key == "carb_settings":
                merged["carb_settings"] = {**merged.get("carb_settings", {}), **value}
            else:
                merged[key] = value
    return merged


def global_settings_cfg(settings: Mapping[str, Any]) -> Any:
    """Isaac Lab's ``IsaacRtxRendererGlobalSettingsCfg`` for ``settings``."""
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg  # noqa: PLC0415

    return IsaacRtxRendererGlobalSettingsCfg(**settings)


def apply_to_cameras(env_cfg: Any, settings: Mapping[str, Any]) -> None:
    """Give every camera renderer in the scene cfg these global settings, applied when it starts."""
    from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings  # noqa: PLC0415

    for value in vars(env_cfg.scene).values():
        renderer_cfg = getattr(value, "renderer_cfg", None)
        if renderer_cfg is not None and hasattr(renderer_cfg, "global_settings"):
            set_isaac_rtx_global_settings(renderer_cfg, **settings)


def apply_rtx_settings(env: Any, env_ids: Any, settings: Mapping[str, Any]) -> None:
    """Event term: apply ``settings`` to the running app."""
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings  # noqa: PLC0415

    apply_isaac_rtx_global_settings(global_settings_cfg(settings))


def add_startup_event(env_cfg: Any, settings: Mapping[str, Any]) -> None:
    """Re-apply these settings at env startup, after Arena zeroed the ambient light."""
    from isaaclab.managers import EventTermCfg  # noqa: PLC0415

    env_cfg.events.i4h_rtx_settings = EventTermCfg(
        func=apply_rtx_settings, mode="startup", params={"settings": dict(settings)}
    )


def apply_to_env_cfg(env_cfg: Any, settings: Mapping[str, Any]) -> None:
    """Hand these settings to every camera renderer and re-apply them at env startup."""
    apply_to_cameras(env_cfg, settings)
    add_startup_event(env_cfg, settings)
