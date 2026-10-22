# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from i4h_common.manifest import load_scene_spec

WORKFLOWS_ROOT = Path(__file__).parents[2]


def test_scene_contract() -> None:
    """The probe-reach scene runs as an Arena environment; its RL profile and policy read this manifest."""
    manifest = WORKFLOWS_ROOT / "arena" / "i4h_arena" / "scenes" / "manifest" / "ultrasound_probe_reach.yaml"
    spec = load_scene_spec(manifest)
    assert spec.impl == "i4h_arena.scenes.ultrasound_probe_reach:UltrasoundProbeReachScene"
    assert spec.max_steps > 0
    assert spec.control_hz > 0.0
    assert spec.embodiment == "panda"
    assert spec.action_space == "ee_pose"
    assert spec.dof == 6
    assert spec.robots == ("robot",)
    assert spec.objects == ("table", "organs", "target")
    assert spec.cameras == ("room", "wrist")
