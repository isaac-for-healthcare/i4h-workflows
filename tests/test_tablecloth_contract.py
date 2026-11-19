# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from i4h_common.config import get_robot_config
from i4h_common.manifest import load_scene_spec

MANIFESTS = Path(__file__).parents[1] / "arena" / "i4h_arena" / "scenes" / "manifest"


@pytest.mark.parametrize("robot,width,state_width", [("g1", 38, 53), ("h2", 58, 75)])
def test_recordings_and_their_descriptors_share_the_controller_contract(robot, width, state_width):
    spec = load_scene_spec(MANIFESTS / f"{robot}_tablecloth.yaml")
    assert spec.dof == width and spec.action_space == "bimanual_pose"
    embodiment = get_robot_config(spec.embodiment)
    assert len(embodiment.action_names) == width
    assert len(embodiment.joint_names) == len(embodiment.state_names) == state_width
    assert len(set(embodiment.joint_names)) == state_width
    assert embodiment.action_names[:7] == tuple(
        f"left_wrist.{axis}" for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")
    )
