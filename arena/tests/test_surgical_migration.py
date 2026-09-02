# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static regression coverage for behavior migrated from robotic_surgery.

IsaacLab env cfg modules require AppLauncher to be initialized before import,
so these tests inspect the authored contract without starting Kit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ARENA = Path(__file__).parents[1] / "i4h_arena"
ENVCFG_SOURCE = ARENA / "envcfg" / "_surgical.py"
SCENE_SOURCE = ARENA / "scenes" / "_surgical.py"
STAR_SOURCE = ARENA / "embodiments" / "star.py"
MANIFESTS = ARENA / "scenes" / "manifest"


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    text = path.read_text()
    cls = _class(ast.parse(text), class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return ast.get_source_segment(text, method) or ""


def test_reach_dispatches_all_source_reset_randomizers() -> None:
    source = _method_source(ENVCFG_SOURCE, "SurgicalReachEnvCfg", "get_events_cfg")

    assert "_PsmReachEventsCfg()" in source
    assert "_DualPsmReachEventsCfg()" in source
    assert "_StarReachEventsCfg()" in source


def test_star_reach_resets_inside_joint_limits_for_stable_ik() -> None:
    text = ENVCFG_SOURCE.read_text()
    star_events = _class(ast.parse(text), "_StarReachEventsCfg")
    source = ast.get_source_segment(text, star_events) or ""

    assert '"position_range": (0.75, 0.9)' in source


def test_star_restores_source_actuator_velocity_limits() -> None:
    text = STAR_SOURCE.read_text()

    assert "velocity_limit_sim=2.175" in text
    assert "velocity_limit_sim=0.2" in text


def test_star_reach_samples_the_validated_ik_workspace() -> None:
    text = ENVCFG_SOURCE.read_text()
    function = next(
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == "_star_reach_pose_command_cfg"
    )
    source = ast.get_source_segment(text, function) or ""

    assert "pos_x=(0.45, 0.52)" in source
    assert "pos_z=(0.26, 0.4)" in source


def test_reach_target_is_stable_for_the_whole_episode() -> None:
    text = ENVCFG_SOURCE.read_text()
    module = ast.parse(text)
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_REACH_COMMAND_HOLD_S" for target in node.targets)
    )

    assert ast.literal_eval(assignment.value) == 6.0
    commands_source = _method_source(ENVCFG_SOURCE, "SurgicalReachEnvCfg", "get_commands_cfg")
    assert commands_source.count("resampling_time_range=hold") == 4


@pytest.mark.parametrize(
    ("manifest", "control_hz", "dof", "gripper"),
    [
        ("psm_reach.yaml", 30.0, 7, False),
        ("dual_psm_reach.yaml", 30.0, 14, False),
        ("star_reach.yaml", 30.0, 7, False),
        ("psm_block.yaml", 50.0, 8, True),
        ("psm_needle.yaml", 50.0, 8, True),
        ("psm_needle_organs.yaml", 50.0, 8, True),
    ],
)
def test_surgical_manifest_matches_migrated_control_contract(
    manifest: str,
    control_hz: float,
    dof: int,
    gripper: bool,
) -> None:
    data = yaml.safe_load((MANIFESTS / manifest).read_text())

    assert data["control_hz"] == control_hz
    assert data["dof"] == dof
    assert data.get("gripper", True) is gripper


def test_lift_scene_restores_source_physics_rate() -> None:
    text = SCENE_SOURCE.read_text()
    lift_class = _class(ast.parse(text), "SurgicalLiftScene")
    assignments = {
        target.id: ast.unparse(node.value)
        for node in lift_class.body
        if isinstance(node, ast.AnnAssign) and isinstance((target := node.target), ast.Name)
    }

    assert assignments["sim_dt"] == "1.0 / 200.0"
    assert assignments["sim_decimation"] == "4"
    assert assignments["render_interval"] == "4"


def test_psm_embodiment_restores_reach_and_block_action_differences() -> None:
    source = _method_source(SCENE_SOURCE, "SurgicalScene", "_embodiment")

    assert 'shared["include_gripper_action"] = self.asset_mode != "reach_psm"' in source
    assert 'shared["gripper_close"] = 0.1 if self.asset_mode == "lift_block" else 0.09' in source
