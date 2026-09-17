# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from i4h_common.config import get_robot_config
from i4h_engine.lint import lint_workflow
from i4h_engine.loader import load_workflow_module, resolve_workflow
from i4h_engine.registry import default_registry


@pytest.mark.parametrize("robot,width,state_width", [("g1", 38, 53), ("h2", 58, 75)])
def test_record_and_replay_share_controller_contract(robot, width, state_width):
    registry = default_registry()
    teleop = resolve_workflow(f"spread_tablecloth_{robot}", "teleop")
    replay = resolve_workflow(f"spread_tablecloth_{robot}", "replay", dataset="unused.hdf5")
    assert teleop.scene == replay.scene
    assert lint_workflow(teleop, registry).ok
    spec = registry.scene(teleop.scene)
    assert spec.dof == width and spec.action_space == "bimanual_pose"
    embodiment = get_robot_config(spec.embodiment)
    assert len(embodiment.action_names) == width
    assert len(embodiment.joint_names) == len(embodiment.state_names) == state_width
    assert len(set(embodiment.joint_names)) == state_width
    assert embodiment.action_names[:7] == tuple(
        f"left_wrist.{axis}" for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")
    )


def test_tablecloth_rejects_unrelated_teleop_device():
    with pytest.raises(ValueError, match="requires the xr device"):
        resolve_workflow("spread_tablecloth_g1", "teleop", device="keyboard")


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_tablecloth_exposes_only_recovered_modes(robot):
    source = load_workflow_module(f"spread_tablecloth_{robot}")
    assert set(source.modes) == {"idle", "teleop", "replay"}
    assert source.default_mode == "idle"
