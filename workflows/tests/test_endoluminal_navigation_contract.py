# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from i4h_common.manifest import load_scene_spec
from i4h_common.training import task_spec
from i4h_engine.loader import load_workflow_module, resolve_workflow
from i4h_engine.registry import default_registry

#: Both endoluminal workflows drive the same catheter through the same goal.
WORKFLOWS = ("endoluminal_navigation", "endoluminal_navigation_arm")

MANIFESTS = Path(__file__).parents[2] / "arena" / "i4h_arena" / "scenes" / "manifest"

#: The remote task a finetuned N1.7 checkpoint is served as.
N17_TASK = "gr00t_n17/catheter_navigation"

#: The same checkpoint served into the arm-borne scene. A separate declaration
#: only because a manifest names one embodiment and lint matches it by name.
N17_ARM_TASK = "gr00t_n17/catheter_navigation_arm"

#: Which task each workflow's ``policy_n17`` mode rolls out.
N17_TASK_FOR_WORKFLOW = {
    "endoluminal_navigation": N17_TASK,
    "endoluminal_navigation_arm": N17_ARM_TASK,
}


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


def test_each_workflow_rolls_out_the_task_that_names_its_embodiment() -> None:
    """Both scenes roll out, each through the task declaring its embodiment.

    ``merged_requires`` folds a remote task's ``embodiment`` into the
    capabilities lint matches against a Scene, and it is matched by name, so a
    manifest admits exactly one embodiment. That is the only reason there are
    two declarations of one task.
    """
    for workflow, task_id in N17_TASK_FOR_WORKFLOW.items():
        rollout = only_node(workflow, "policy_n17")
        assert rollout.task_id == task_id
        assert rollout.params["until"] is authored(workflow).success
        assert default_registry().task(task_id).requires["embodiment"] == scene_spec(workflow).embodiment


def test_the_two_scenes_name_different_embodiments() -> None:
    """The premise of the split. Were these equal, one manifest would do."""
    assert scene_spec("endoluminal_navigation").embodiment == "catheter"
    assert scene_spec("endoluminal_navigation_arm").embodiment == "franka_catheter"


def test_the_arm_task_differs_from_the_armless_one_only_in_embodiment() -> None:
    """One checkpoint serves both, so everything it reads has to agree.

    The arm carries the drive unit rather than being commanded through it, so a
    difference in any of these would mean the two declarations had drifted into
    describing different models rather than one model in two scenes.
    """
    armless = task_spec(N17_TASK)
    arm = task_spec(N17_ARM_TASK)
    assert arm.cameras == armless.cameras
    assert arm.requires == armless.requires
    assert arm.model == armless.model
    assert arm.observation["image_size"] == armless.observation["image_size"]


def test_only_the_armless_task_is_finetunable() -> None:
    """A recording from either scene carries the same catheter/carm columns, so
    there is one thing to train and a second ``train:`` block would only be a
    copy to keep in step."""
    assert task_spec(N17_TASK).trainable
    assert not task_spec(N17_ARM_TASK).trainable


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
