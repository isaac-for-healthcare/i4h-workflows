# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render settings and Kit args for Blackwell GPUs (e.g. RTX 5090) on Isaac Sim 5.1."""

from __future__ import annotations

import os
import sys
from typing import Any

# Kit flags applied before SimulationApp starts. NGX/DLSS and heavy RTX features can
# segfault on RTX 50-series (Blackwell) with Isaac Sim 5.1 + Vulkan shader compilation.
_BLACKWELL_KIT_FLAGS: tuple[str, ...] = (
    "--/ngx/enabled=false",
    "--/rtx-transient/dldenoiser/enabled=false",
    "--/rtx-transient/dlssg/enabled=false",
    "--/rtx/reflections/enabled=false",
    "--/rtx/indirectDiffuse/enabled=false",
    "--/rtx/translucency/enabled=false",
    "--/rtx/ambientOcclusion/enabled=false",
    "--/rtx/raytracing/cached/enabled=false",
    "--/rtx/directLighting/sampledLighting/enabled=false",
    "--/renderer/multiGpu/maxGpuCount=1",
)

BLACKWELL_SAFE_KIT_ARGS = " ".join(_BLACKWELL_KIT_FLAGS)


def blackwell_render_patch_enabled() -> bool:
    """Return False when the user opts out via RHEO_DISABLE_BLACKWELL_RENDER_PATCH."""
    return os.environ.get("RHEO_DISABLE_BLACKWELL_RENDER_PATCH", "").lower() not in ("1", "true", "yes")


def _merge_kit_arg_values(existing: str) -> str:
    merged = (existing or "").strip()
    for flag in _BLACKWELL_KIT_FLAGS:
        if flag not in merged:
            merged = f"{merged} {flag}".strip()
    return merged


def ensure_blackwell_safe_kit_args(argv: list[str] | None = None) -> None:
    """Inject Kit args to stabilize rendering unless the user already set --kit_args."""
    if not blackwell_render_patch_enabled():
        return
    target = sys.argv if argv is None else argv

    for index, arg in enumerate(target):
        if arg == "--kit_args":
            if index + 1 < len(target):
                target[index + 1] = _merge_kit_arg_values(target[index + 1])
            else:
                target.insert(index + 1, BLACKWELL_SAFE_KIT_ARGS)
            return
        if arg.startswith("--kit_args="):
            prefix, value = arg.split("=", 1)
            target[index] = f"{prefix}={_merge_kit_arg_values(value)}"
            return

    # Insert before env subcommands (e.g. observe_object); trailing kit_args are not seen by the parent parser.
    target.insert(1, "--kit_args")
    target.insert(2, BLACKWELL_SAFE_KIT_ARGS)


def apply_blackwell_safe_render_cfg(render_cfg: Any) -> None:
    """Tune Isaac Lab RenderCfg for stability on Blackwell GPUs."""
    if not blackwell_render_patch_enabled():
        return

    render_cfg.rendering_mode = "performance"
    render_cfg.antialiasing_mode = "Off"
    render_cfg.enable_translucency = False
    render_cfg.enable_dlssg = False
    render_cfg.enable_dl_denoiser = False
    render_cfg.enable_reflections = False
    render_cfg.enable_global_illumination = False
    render_cfg.enable_direct_lighting = False
    render_cfg.enable_shadows = False
    render_cfg.enable_ambient_occlusion = False
    render_cfg.samples_per_pixel = 1

    carb_settings = dict(render_cfg.carb_settings or {})
    carb_settings.pop("rtx.raytracing.fractionalCutoutOpacity", None)
    render_cfg.carb_settings = carb_settings or None
