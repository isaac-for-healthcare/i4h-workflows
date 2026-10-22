# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the GR00T N1.7 remote policy's requests and chunk replay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from i4h_arena.policies import gr00t_n17
from i4h_common.config import get_robot_config
from i4h_common.joint_utils import isaaclab_rad_to_lerobot


class _Server:
    """Answers like GR00T 1.7's server: the home pose, in LeRobot units, for 16 steps."""

    def __init__(self, home_lerobot: np.ndarray) -> None:
        self.home = home_lerobot
        self.requests: list[dict] = []

    def __call__(self, host: str, port: int) -> _Server:
        return self

    def get_modality_config(self) -> dict:
        return {"language": SimpleNamespace(modality_keys=["annotation.human.task_description"])}

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        self.requests.append(observation)
        arm = np.broadcast_to(self.home[:5], (1, 16, 5)).copy()
        return {"single_arm": arm, "gripper": np.full((1, 16, 1), self.home[5])}, {}


def test_so101_requests_are_nested_in_lerobot_units_and_actions_come_back_in_radians(monkeypatch) -> None:
    robot = get_robot_config("so101")
    home = np.asarray(robot.home_joint_pos_rad)
    server = _Server(
        isaaclab_rad_to_lerobot(home, robot.isaaclab_joint_pos_limit_range, robot.lerobot_joint_pos_limit_range)
    )
    monkeypatch.setattr(gr00t_n17, "PolicyClient", server)
    policy = gr00t_n17.Gr00tN17RemotePolicy(gr00t_n17.Gr00tN17RemotePolicyCfg())
    policy.set_task_description("Grip the scissors and put it into the tray")
    observation = {
        "policy": {
            "joint_pos": torch.from_numpy(home).float().unsqueeze(0),
            "room": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
            "wrist": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
        }
    }
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(num_envs=1, action_manager=SimpleNamespace(total_action_dim=6), device="cpu")
    )

    actions = [policy.get_action(env, observation) for _ in range(17)]

    first = server.requests[0]
    assert first["video"]["room"].shape == (1, 1, 480, 640, 3)
    assert first["state"]["single_arm"].shape == (1, 1, 5)
    assert first["state"]["gripper"].dtype == np.float32
    assert first["language"] == {"annotation.human.task_description": [["Grip the scissors and put it into the tray"]]}
    # One request per 16 executed actions, each back in Isaac Lab radians.
    assert len(server.requests) == 2
    assert torch.allclose(actions[0][0], torch.from_numpy(home).float(), atol=1e-5)
