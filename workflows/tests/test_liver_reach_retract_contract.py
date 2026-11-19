# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from i4h_common.manifest import load_scene_spec
from i4h_engine.loader import load_workflow_module, resolve_workflow


def test_workflow_contract() -> None:
    source = load_workflow_module("liver_reach_retract")
    resolved = resolve_workflow("liver_reach_retract", "idle")
    assert source.scene == "liver_reach_retract"
    assert "idle" in source.modes
    assert resolved.scene == "liver_reach_retract"
    assert resolved.mode == "idle"

    manifest = Path(__file__).parents[2] / "arena" / "i4h_arena" / "scenes" / "manifest" / "liver_reach_retract.yaml"
    spec = load_scene_spec(manifest)
    assert spec.impl == "i4h_arena.scenes.liver_reach_retract:LiverReachRetractScene"
    assert spec.max_steps > 0
    assert spec.control_hz > 0.0
    assert spec.dof >= 0
    if spec.embodiment == "none":
        assert spec.dof == 0
        assert spec.robots == ()
    else:
        assert spec.dof > 0
        assert spec.robots
