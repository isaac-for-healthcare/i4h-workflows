# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One GR00T N1.7 catheter checkpoint serves both endoluminal environments."""

from __future__ import annotations

from pathlib import Path

from i4h_common.config import get_robot_config
from i4h_common.manifest import load_scene_spec
from i4h_common.training import task_spec

#: Both endoluminal scenes drive the same catheter through the same goal.
SCENES = ("endoluminal_navigation", "endoluminal_navigation_arm")

MANIFESTS = Path(__file__).parents[1] / "arena" / "i4h_arena" / "scenes" / "manifest"

#: The task a finetuned N1.7 checkpoint is served as, in either scene.
N17_TASK = "gr00t_n17/catheter_navigation"


def scene_spec(name: str):
    return load_scene_spec(MANIFESTS / f"{name}.yaml")


def test_the_two_scenes_name_different_embodiments() -> None:
    """The premise of the split. Were these equal, one manifest would do."""
    assert scene_spec("endoluminal_navigation").embodiment == "catheter"
    assert scene_spec("endoluminal_navigation_arm").embodiment == "franka_catheter"


def test_both_scenes_speak_the_action_space_the_n17_task_wants() -> None:
    """The arm changes who holds the drive unit, not what is commanded."""
    wanted = task_spec(N17_TASK).requires["action_space"]
    assert wanted == "catheter_carm_velocity"
    for name in SCENES:
        spec = scene_spec(name)
        assert spec.action_space == wanted
        assert spec.dof == 4


def test_both_embodiments_split_into_the_groups_the_checkpoint_reads() -> None:
    """A recording from either scene trains the same task, so the converted
    datasets must carry the same catheter and C-arm groups."""
    armless = get_robot_config("catheter")
    arm = get_robot_config("franka_catheter")
    assert [group for group, *_ in arm.action_split] == [group for group, *_ in armless.action_split]
