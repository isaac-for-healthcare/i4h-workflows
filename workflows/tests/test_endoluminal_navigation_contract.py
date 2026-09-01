# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from i4h_common.manifest import load_scene_spec
from i4h_engine.loader import load_workflow_module, resolve_workflow


def test_workflow_contract() -> None:
    source = load_workflow_module("endoluminal_navigation")
    resolved = resolve_workflow("endoluminal_navigation", "idle")
    assert source.scene == "endoluminal_navigation"
    assert "idle" in source.modes
    assert "teleop" in source.modes
    assert "demo" in source.modes
    assert "validate_fluoroscopy" in source.modes
    assert resolved.scene == "endoluminal_navigation"
    assert resolved.mode == "idle"

    manifest = Path(__file__).parents[2] / "arena" / "i4h_arena" / "scenes" / "manifest" / "endoluminal_navigation.yaml"
    spec = load_scene_spec(manifest)
    assert spec.impl == "i4h_arena.scenes.endoluminal_navigation:EndoluminalNavigationScene"
    assert spec.max_steps > 0
    assert spec.control_hz > 0.0
    assert spec.embodiment == "catheter"
    assert spec.action_space == "catheter_carm_velocity"
    assert spec.dof == 3
    assert spec.robots == ("robot",)
