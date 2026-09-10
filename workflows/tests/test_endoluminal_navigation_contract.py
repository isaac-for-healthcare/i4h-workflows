# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from i4h_common.manifest import load_scene_spec
from i4h_engine.loader import load_workflow_module, resolve_workflow
from i4h_engine.registry import default_registry

#: Both endoluminal workflows drive the same catheter through the same goal.
WORKFLOWS = ("endoluminal_navigation", "endoluminal_navigation_arm")

MANIFESTS = Path(__file__).parents[2] / "arena" / "i4h_arena" / "scenes" / "manifest"

#: The remote task a finetuned N1.7 checkpoint is served as.
N17_TASK = "gr00t_n17/catheter_navigation"


def scene_spec(name: str):
    return load_scene_spec(MANIFESTS / f"{name}.yaml")


def authored(name: str):
    """The exported ``WORKFLOW`` value, whose ``modes`` are the builders."""
    return load_workflow_module(name).workflow


def only_node(workflow: str, mode: str):
    """The single node of a one-task mode graph."""
    (node,) = authored(workflow).modes[mode]().nodes
    return node


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

    spec = scene_spec("endoluminal_navigation")
    assert spec.impl == "i4h_arena.scenes.endoluminal_navigation:EndoluminalNavigationScene"
    assert spec.max_steps > 0
    assert spec.control_hz > 0.0
    assert spec.embodiment == "catheter"
    assert spec.action_space == "catheter_carm_velocity"
    assert spec.dof == 4
    assert spec.robots == ("robot",)


def test_teleop_stops_on_arrival() -> None:
    """A demonstration has to end at the goal to be worth training on.

    Without ``until`` the graph only ends at the step cap, so every episode
    looks the same to the recorder and none of them carries a success label.
    """
    for name in WORKFLOWS:
        drive = only_node(name, "teleop")
        assert drive.task_id == "teleop/drive"
        assert drive.params["until"] is authored(name).success


def test_the_n17_mode_rolls_out_the_catheter_task() -> None:
    rollout = only_node("endoluminal_navigation", "policy_n17")
    assert rollout.task_id == N17_TASK
    assert rollout.params["until"] is authored("endoluminal_navigation").success


def test_the_arm_workflow_exposes_no_policy_mode() -> None:
    """Recording happens with the arm; rollout does not.

    The pairing is refused rather than chosen: see the embodiment test below
    for the reason. Asserting the absence here keeps someone from adding the
    mode back without also reconciling the two embodiment names.
    """
    modes = load_workflow_module("endoluminal_navigation_arm").modes
    assert "policy_n17" not in modes
    assert "policy" not in modes


def test_the_n17_task_embodiment_admits_only_the_armless_scene() -> None:
    """Why the arm workflow has no policy mode.

    ``merged_requires`` folds a remote task's ``embodiment`` into the
    capabilities lint matches against a Scene, so the two names have to agree.
    """
    requires = default_registry().task(N17_TASK).requires
    assert requires["embodiment"] == "catheter"
    assert scene_spec("endoluminal_navigation").embodiment == "catheter"
    assert scene_spec("endoluminal_navigation_arm").embodiment == "franka_catheter"


def test_both_scenes_speak_the_action_space_the_n17_task_wants() -> None:
    """The arm changes who holds the drive unit, not what is commanded.

    This is what makes a recording made with the arm trainable for a task the
    arm scene cannot itself roll out.
    """
    wanted = default_registry().task(N17_TASK).requires["action_space"]
    assert wanted == "catheter_carm_velocity"
    for name in WORKFLOWS:
        spec = scene_spec(name)
        assert spec.action_space == wanted
        assert spec.dof == 4
