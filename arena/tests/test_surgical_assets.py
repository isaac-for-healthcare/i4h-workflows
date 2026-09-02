# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path


def test_default_surgical_table_uses_static_usd_spawner() -> None:
    source_path = Path(__file__).parents[1] / "i4h_arena" / "assets" / "_surgical.py"
    module = ast.parse(source_path.read_text())
    scene_cfg = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SurgicalSceneCfg")
    table_assignment = next(
        node
        for node in scene_cfg.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "table" for target in node.targets)
    )

    assert isinstance(table_assignment.value, ast.Call)
    assert isinstance(table_assignment.value.func, ast.Name)
    assert table_assignment.value.func.id == "AssetBaseCfg"
    spawn_keyword = next(keyword for keyword in table_assignment.value.keywords if keyword.arg == "spawn")
    assert isinstance(spawn_keyword.value, ast.Call)
    func_keyword = next(keyword for keyword in spawn_keyword.value.keywords if keyword.arg == "func")
    assert isinstance(func_keyword.value, ast.Name)
    assert func_keyword.value.id == "_spawn_static_usd"


def test_organs_needle_uses_collision_ready_asset() -> None:
    source_path = Path(__file__).parents[1] / "i4h_arena" / "assets" / "_surgical.py"
    module = ast.parse(source_path.read_text())
    scene_cfg = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SurgicalSceneCfg")
    needle_assignment = next(
        node
        for node in scene_cfg.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "organs_needle_object" for target in node.targets)
    )
    spawn_keyword = next(keyword for keyword in needle_assignment.value.keywords if keyword.arg == "spawn")
    usd_keyword = next(keyword for keyword in spawn_keyword.value.keywords if keyword.arg == "usd_path")

    assert isinstance(usd_keyword.value, ast.Name)
    assert usd_keyword.value.id == "NEEDLE_SDF_USD"
