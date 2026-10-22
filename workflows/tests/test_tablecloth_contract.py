# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from i4h_common.config import get_robot_config
from i4h_engine.registry import default_registry


@pytest.mark.parametrize("robot,width,state_width", [("g1", 38, 53), ("h2", 58, 75)])
def test_recordings_and_their_descriptors_share_the_controller_contract(robot, width, state_width):
    spec = default_registry().scene(f"{robot}_tablecloth")
    assert spec.dof == width and spec.action_space == "bimanual_pose"
    embodiment = get_robot_config(spec.embodiment)
    assert len(embodiment.action_names) == width
    assert len(embodiment.joint_names) == len(embodiment.state_names) == state_width
    assert len(set(embodiment.joint_names)) == state_width
    assert embodiment.action_names[:7] == tuple(
        f"left_wrist.{axis}" for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")
    )
