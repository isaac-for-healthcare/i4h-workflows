# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the liver-scan policies: the rule-based sweep and the PI0 request."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from i4h_arena.policies import scan
from i4h_arena.policies.pi0 import FrankaProbeAdapter

IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


class _ProbeEnv:
    """A one-env phantom scene at 50 Hz whose probe follows each relative command exactly."""

    num_envs = 1
    device = "cpu"
    step_dt = 0.02

    def __init__(self) -> None:
        self.unwrapped = self
        self.tcp_pos = torch.tensor([[0.33, -0.01, 0.30]])
        self.tcp_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        root = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(1, 3), root_quat_w=IDENTITY))
        phantom = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=torch.tensor([[0.45, 0.05, 0.02]]), root_quat_w=IDENTITY)
        )
        self.scene = {"robot": root, "organs": phantom}
        self.action_manager = SimpleNamespace(total_action_dim=6)

    def apply(self, action: torch.Tensor) -> None:
        self.tcp_pos = self.tcp_pos + action[:, :3]
        angle = action[:, 3:].norm(dim=-1, keepdim=True)
        axis = action[:, 3:] / angle.clamp_min(1e-12)
        delta = torch.cat([axis * torch.sin(angle / 2), torch.cos(angle / 2)], -1)
        self.tcp_quat = scan.quat_mul(delta, self.tcp_quat)


def test_sweep_moves_through_the_phantom_waypoints_on_the_task_graph_schedule(monkeypatch) -> None:
    env = _ProbeEnv()
    monkeypatch.setattr(scan, "tcp_pose_in_root", lambda _env, _frame, _robot: (env.tcp_pos, env.tcp_quat))
    policy = scan.ProbeSweepPolicy(scan.ProbeSweepPolicyCfg())

    positions = []
    for _ in range(200):
        action = policy.get_action(env, None)
        env.apply(action)
        positions.append(env.tcp_pos[0].clone())

    # locate 0, approach 1-30, contact 31-60, sweep_0 61-110, sweep_1 111-160, then hold
    phantom = torch.tensor([0.45, 0.05, 0.02])
    for step, waypoint in ((30, 0), (60, 1), (110, 2), (160, 3)):
        # The phantom frame is the identity, so each waypoint is an offset from its origin.
        assert torch.allclose(positions[step], phantom + torch.tensor(scan.WAYPOINTS[waypoint]), atol=1e-5)
    assert torch.allclose(positions[199], positions[160], atol=1e-6)
    # The probe points down: its z axis along the phantom's -z.
    assert torch.allclose(env.tcp_quat.abs(), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-5)

    policy.reset(torch.tensor([0]))
    assert torch.equal(policy.get_action(env, None), torch.zeros(1, 6))


def test_a_move_waits_for_the_probe_to_arrive(monkeypatch) -> None:
    env = _ProbeEnv()
    monkeypatch.setattr(scan, "tcp_pose_in_root", lambda _env, _frame, _robot: (env.tcp_pos, env.tcp_quat))
    policy = scan.ProbeSweepPolicy(scan.ProbeSweepPolicyCfg())

    for _ in range(100):
        policy.get_action(env, None)  # the probe never moves

    assert int(policy._phase[0]) == scan.FIRST_MOVE


@pytest.mark.parametrize(
    ("quat", "vector"),
    [
        ((0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)), (0.0, 0.0, math.pi / 2)),
        # A negative w keeps the long way round, as the relative command was given it.
        ((0.0, 0.0, -math.sin(math.pi / 4), -math.cos(math.pi / 4)), (0.0, 0.0, -3 * math.pi / 2)),
        ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
    ],
)
def test_rotation_vectors_keep_the_quaternions_arc(quat, vector) -> None:
    result = scan._rotation_vector(torch.tensor([quat]))
    assert torch.allclose(result, torch.tensor([vector]), atol=1e-6)


def test_pi0_requests_carry_the_arm_joints_and_both_views_at_the_trained_size() -> None:
    observation = {
        "policy": {
            "joint_pos": torch.arange(9.0).reshape(1, 9),
            "room": torch.full((1, 480, 640, 3), 200, dtype=torch.uint8),
            "wrist": torch.full((1, 240, 320, 3), 100, dtype=torch.uint8),
        }
    }
    adapter = FrankaProbeAdapter()

    request = adapter.pack_request(adapter.extract(observation, 0), "Perform a liver ultrasound.")

    assert request["observation/state"].tolist() == list(range(7))
    assert request["observation/state"].dtype == np.float64
    assert request["observation/image"].shape == (224, 224, 3)
    assert request["observation/wrist_image"].dtype == np.uint8
    assert request["prompt"] == "Perform a liver ultrasound."
