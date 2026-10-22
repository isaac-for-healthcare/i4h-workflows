# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the i4h Arena environments, surgical success terms and rule-based policies."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from i4h_arena import environments
from i4h_arena.envcfg._surgical import object_above_root, tcps_reached_commands
from i4h_arena.policies import _cartesian, lift
from i4h_arena.policies._cartesian import Arm, ease_pose, nlerp, smoothstep

IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


def _flags(environment: type) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    environment.add_cli_args(parser)
    return parser.parse_args([])


def test_every_scene_is_an_arena_environment_named_after_it() -> None:
    assert environments.PsmReach.name == "psm_reach"
    assert environments.DualPsmReach.name == "dual_psm_reach"
    assert environments.UltrasoundProbeReach.name == "ultrasound_probe_reach"


def test_scene_flags_reach_the_arena_builder_by_name() -> None:
    surgical = _flags(environments.PsmNeedle)
    assert surgical.action_device == "ik_abs"
    assert surgical.recorder_dataset_export_dir_path is None
    assert surgical.recorder_dataset_filename is None
    assert not hasattr(_flags(environments.SoarmScissors), "action_device")
    assert _flags(environments.UltrasoundProbeReach).rl_observations is False


def test_eased_motion_ends_exactly_on_the_goal() -> None:
    assert smoothstep(torch.tensor([0.0, 0.5, 1.0, 2.0])).tolist() == [0.0, 0.5, 1.0, 1.0]
    goal_quat = torch.tensor([[0.0, 0.0, -0.7071068, -0.7071068]])
    pos, quat = ease_pose(torch.zeros(1, 3), IDENTITY, torch.ones(1, 3), goal_quat, torch.tensor([1.0]))
    assert torch.equal(pos, torch.ones(1, 3))
    assert torch.equal(quat, goal_quat)


def test_quaternion_interpolation_takes_the_short_way() -> None:
    halfway = nlerp(IDENTITY, -IDENTITY, torch.tensor([[0.5]]))
    assert torch.allclose(halfway.abs(), IDENTITY)


def _scene_entity(**data: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(**data))


def _surgical_env(tcp_z: float, object_z: float, command_z: float) -> SimpleNamespace:
    robot = _scene_entity(root_state_w=torch.tensor([[0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0, *([0.0] * 6)]]))
    frame = _scene_entity(
        target_pos_w=torch.tensor([[[0.0, 0.0, 0.15 + tcp_z]]]), target_quat_w=IDENTITY.reshape(1, 1, 4)
    )
    obj = _scene_entity(root_pos_w=torch.tensor([[0.0, 0.0, 0.15 + object_z]]))
    command = torch.tensor([[0.0, 0.0, command_z, 0.0, 0.0, 0.0, 1.0]])
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"robot": robot, "ee_frame": frame, "object": obj},
        command_manager=SimpleNamespace(get_command=lambda _name: command),
    )


def test_reach_succeeds_within_the_radius_of_the_command_in_the_root_frame() -> None:
    arms = (("ee_pose", "ee_frame", "robot"),)
    assert tcps_reached_commands(_surgical_env(-0.105, 0.0, -0.1), arms, 0.01).all()
    assert not tcps_reached_commands(_surgical_env(-0.115, 0.0, -0.1), arms, 0.01).any()


def test_lift_succeeds_above_the_height_in_the_root_frame() -> None:
    assert object_above_root(_surgical_env(0.0, -0.08, 0.0), -0.085).all()
    assert not object_above_root(_surgical_env(0.0, -0.09, 0.0), -0.085).any()


class _TrackingEnv:
    """A one-env scene at 50 Hz whose tool tip lands exactly on each commanded pose."""

    num_envs = 1
    device = "cpu"
    step_dt = 0.02

    def __init__(self) -> None:
        self.unwrapped = self
        self.tcp = torch.tensor([[0.0, 0.0, -0.06, 0.0, 0.0, 0.0, 1.0]])
        root = _scene_entity(root_pos_w=torch.tensor([[0.0, 0.0, 0.15]]), root_quat_w=IDENTITY)
        obj = _scene_entity(
            root_pos_w=torch.tensor([[0.01, -0.01, 0.15 - 0.13]]),
            root_quat_w=IDENTITY,
            root_lin_vel_w=torch.zeros(1, 3),
        )
        self.scene = {"robot": root, "object": obj}
        self.action_manager = SimpleNamespace(total_action_dim=8)


def test_lift_phases_last_as_long_as_the_task_graph_nodes_did(monkeypatch) -> None:
    env = _TrackingEnv()
    monkeypatch.setattr(lift, "arms_of", lambda _env: (Arm("robot", "ee_frame", slice(0, 7), jaw=7),))
    monkeypatch.setattr(lift, "tcp_pose_in_root", lambda _env, _frame, _robot: (env.tcp[:, :3], env.tcp[:, 3:]))
    policy = lift.CartesianLiftPolicy(lift.CartesianLiftPolicyCfg())

    actions = []
    for _ in range(180):
        action = policy.get_action(env, None)
        actions.append(action.clone())
        env.tcp = action[:, :7].clone()
    jaw = torch.cat([action[:, 7] for action in actions])
    z = torch.cat([action[:, 2] for action in actions])

    # rest 0-24, locate 25, approach 26-75, descend 76-105, grasp 106-125, lift 126-175
    assert int(torch.nonzero(jaw < 0)[0]) == 106
    assert torch.isclose(z[75], torch.tensor(-0.13 + 0.05))
    assert torch.isclose(z[105], torch.tensor(-0.13))
    assert torch.equal(z[106:126], z[105].expand(20))
    assert torch.isclose(z[175], torch.tensor(-0.13 + 0.10))

    policy.reset(torch.tensor([0]))
    assert policy.get_action(env, None)[0, 7] == 1.0


@pytest.mark.parametrize("width", [7, 14])
def test_arms_pair_ik_terms_with_their_frames(monkeypatch, width: int) -> None:
    class Ik:
        def __init__(self, asset: str) -> None:
            self.action_dim = 7
            self.cfg = SimpleNamespace(asset_name=asset, controller=SimpleNamespace(use_relative_mode=False))

    monkeypatch.setattr(_cartesian, "DifferentialInverseKinematicsAction", Ik)
    terms = {"arm_1": Ik("robot_1"), "arm_2": Ik("robot_2")} if width == 14 else {"arm": Ik("robot")}
    manager = SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__)
    arms = _cartesian.arms_of(SimpleNamespace(action_manager=manager))
    assert [arm.frame for arm in arms] == (["ee_frame", "ee_2_frame"] if width == 14 else ["ee_frame"])
    assert arms[-1].pose == slice(width - 7, width)
