# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ARENA = Path(__file__).parents[1] / "i4h_arena"
SOURCE = ARENA / "embodiments" / "kuka_lbr.py"
MANIFEST = ARENA / "embodiments" / "manifest" / "kuka_lbr14.yaml"
EXPECTED_JOINT_NAMES = [f"axis{index}" for index in range(1, 8)]


def test_kuka_embodiment_preserves_isaac_sim_6_contract() -> None:
    module = ast.parse(SOURCE.read_text())
    names = {node.name for node in module.body if isinstance(node, ast.ClassDef | ast.FunctionDef)}

    assert "KukaLbr14MedEmbodiment" in names
    assert "make_kuka_lbr14_med_cfg" in names
    assert "make_kuka_lbr14_med_ik_action_cfg" in names
    assert "LBR14Med.usd" in SOURCE.read_text()


def test_kuka_manifest_replaces_legacy_robot_yaml() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())

    assert manifest == {
        "name": "kuka_lbr14",
        "joint_names": EXPECTED_JOINT_NAMES,
    }
