# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Arena's registration contract without importing the simulator."""

import ast
from pathlib import Path


def test_registered_assets_declare_their_own_registry_key():
    root = Path(__file__).resolve().parents[1] / "arena/i4h_arena"
    checked = 0
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(isinstance(item, ast.Name) and item.id == "register_asset" for item in node.decorator_list):
                continue
            names = {
                target.id
                for item in node.body
                if isinstance(item, ast.Assign | ast.AnnAssign)
                for target in (item.targets if isinstance(item, ast.Assign) else [item.target])
                if isinstance(target, ast.Name)
            }
            assert "registry_name" in names, f"{path}:{node.lineno} {node.name} inherits or omits its registry key"
            checked += 1
    assert checked > 0
