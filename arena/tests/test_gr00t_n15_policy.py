# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the GR00T N1.5 remote policy's requests and chunk replay."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from i4h_arena.policies.gr00t_n15 import Gr00tN15RemotePolicy, Gr00tN15RemotePolicyCfg


def _trocar_env(num_envs: int) -> SimpleNamespace:
    unwrapped = SimpleNamespace(num_envs=num_envs, action_manager=SimpleNamespace(total_action_dim=43), device="cpu")
    return SimpleNamespace(unwrapped=unwrapped)


def _trocar_observation(num_envs: int) -> dict:
    group = {"assemble_trocar_joint_pos": torch.arange(43.0).repeat(num_envs, 1)}
    for camera in ("front_camera_rgb", "left_wrist_camera_rgb", "right_wrist_camera_rgb"):
        group[camera] = torch.full((num_envs, 480, 640, 3), 7, dtype=torch.uint8)
    return {"policy": group}


def _policy(monkeypatch, horizon: int) -> tuple[Gr00tN15RemotePolicy, list[dict]]:
    policy = Gr00tN15RemotePolicy(Gr00tN15RemotePolicyCfg(open_loop_horizon=horizon))
    policy.set_task_description("install trocar from box")
    requests: list[dict] = []

    def reply(request: dict) -> dict:
        requests.append(request)
        step = torch.arange(16.0).reshape(1, 16, 1)
        return {
            "action.left_arm": step.expand(1, 16, 7) + 100 * len(requests),
            "action.right_arm": step.expand(1, 16, 7),
            "action.left_hand": step.expand(1, 16, 7),
            "action.right_hand": step.expand(1, 16, 7),
        }

    monkeypatch.setattr(policy, "_call", reply)
    return policy, requests


def test_requests_carry_the_checkpoint_state_slices_and_uint8_views(monkeypatch) -> None:
    policy, requests = _policy(monkeypatch, horizon=1)
    action = policy.get_action(_trocar_env(1), _trocar_observation(1))
    policy.close()

    request = requests[0]
    assert request["state.left_arm"].flatten().tolist() == list(range(15, 22))
    assert request["state.right_hand"].flatten().tolist() == list(range(36, 43))
    assert request["video.room_view"].shape == (1, 1, 480, 640, 3)
    assert request["video.room_view"].dtype == torch.uint8
    assert request["annotation.human.task_description"] == ["install trocar from box"]
    # The 15 leg and waist targets stay at zero; the 28 arm and hand values follow them.
    assert action.shape == (1, 43)
    assert torch.equal(action[0, :15], torch.zeros(15))
    assert action[0, 15].item() == 100.0


def test_each_env_replays_its_chunk_before_requesting_another(monkeypatch) -> None:
    policy, requests = _policy(monkeypatch, horizon=2)
    env, observation = _trocar_env(2), _trocar_observation(2)
    first = policy.get_action(env, observation)
    second = policy.get_action(env, observation)
    assert len(requests) == 2  # one per env
    assert second[0, 22].item() == first[0, 22].item() + 1  # the chunk's next step

    policy.reset(torch.tensor([1]))
    policy.get_action(env, observation)
    assert len(requests) == 4  # env 0 finished its chunk, env 1 was reset
    policy.close()


def test_so101_requests_cross_in_lerobot_units_and_actions_come_back_in_radians(monkeypatch) -> None:
    from i4h_common.config import get_robot_config
    from i4h_common.joint_utils import isaaclab_rad_to_lerobot

    policy = Gr00tN15RemotePolicy(Gr00tN15RemotePolicyCfg(embodiment="so101_scissors"))
    policy.set_task_description("Grip the scissors and put it into the tray")
    robot = get_robot_config("so101")
    home = torch.tensor(robot.home_joint_pos_rad)
    in_lerobot = isaaclab_rad_to_lerobot(
        home.double().numpy(), robot.isaaclab_joint_pos_limit_range, robot.lerobot_joint_pos_limit_range
    )
    requests: list[dict] = []

    def reply(request: dict) -> dict:
        requests.append(request)
        # The model answers in LeRobot units; send the home pose back for all 16 steps.
        arm = torch.from_numpy(in_lerobot[:5]).expand(16, 5)
        return {"action.single_arm": arm, "action.gripper": torch.from_numpy(in_lerobot[5:]).expand(16)}

    monkeypatch.setattr(policy, "_call", reply)
    observation = {
        "policy": {
            "joint_pos": home.unsqueeze(0),
            "room": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
            "wrist": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
        }
    }
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(num_envs=1, action_manager=SimpleNamespace(total_action_dim=6), device="cpu")
    )

    actions = [policy.get_action(env, observation) for _ in range(17)]
    policy.close()

    (first, second) = requests
    assert first["state.single_arm"].shape == (1, 5)
    assert first["state.single_arm"].dtype == torch.float64
    assert torch.allclose(first["state.gripper"][0], torch.from_numpy(in_lerobot[5:]))
    assert first["video.room"].shape == (1, 480, 640, 3)
    assert first["annotation.human.task_description"] == "Grip the scissors and put it into the tray"
    # One request per 16 executed actions, each back in Isaac Lab radians.
    assert second is not first
    assert torch.allclose(actions[0][0], home, atol=1e-5)
