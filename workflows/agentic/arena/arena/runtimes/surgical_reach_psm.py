# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PSM reach runtime helpers."""

from __future__ import annotations

import numpy as np
import torch
from arena.runtimes.core.base import PolicyIO, run_policy_episode
from common.config import get_robot_config, get_zenoh_config

_PSM_CONFIG = get_robot_config("psm")
_ARM_NAMES = _PSM_CONFIG.body_joint_names
_ACTION_DIM = _PSM_CONFIG.action_dim
_PHYSICAL_GRIPPER_NAMES = ("psm_tool_gripper1_joint", "psm_tool_gripper2_joint")


class SurgicalReachPsmPolicyIO(PolicyIO):
    def __init__(self, *, env_id: str) -> None:
        zenoh = get_zenoh_config(env_id)
        super().__init__(
            camera_keys=zenoh.camera_keys,
            state_key=zenoh.robot_state_key,
            command_key=zenoh.robot_command_key,
            action_dim=_ACTION_DIM,
        )


def publish_obs(ctx) -> None:
    scene = ctx.env.unwrapped.scene
    room = scene["room"].data.output["rgb"][0, ..., :3].cpu().numpy()
    ctx.io.publish_camera("room", room)
    ctx.io.publish_state(_logical_joint_positions(scene["robot"]).flatten())


def run_policy_based_episode(ctx, *, max_timesteps: int) -> str:
    return run_policy_episode(
        ctx,
        max_timesteps=max_timesteps,
        publish_obs=publish_obs,
        policy_action=_policy_action,
        stop_on_env_done=True,
    )


@torch.no_grad()
def sync_robot_joints(env, joint_positions: torch.Tensor) -> None:
    robot = env.unwrapped.scene["robot"]
    target = joint_positions.detach().to(device=env.device, dtype=robot.data.joint_pos.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if target.shape[0] == 1 and env.unwrapped.num_envs > 1:
        target = target.repeat(env.unwrapped.num_envs, 1)
    physical = _logical_to_physical_tensor(target)
    joint_names = (*_ARM_NAMES, *_PHYSICAL_GRIPPER_NAMES)
    joint_ids = [_joint_index(robot, name, index) for index, name in enumerate(joint_names)]
    zeros = torch.zeros_like(physical)
    robot.write_joint_state_to_sim(physical, zeros, joint_ids=joint_ids)
    robot.set_joint_position_target(physical, joint_ids=joint_ids)
    robot.set_joint_velocity_target(zeros, joint_ids=joint_ids)
    robot.write_data_to_sim()


def _logical_joint_positions(robot) -> np.ndarray:
    full = robot.data.joint_pos[0:1].detach().cpu().numpy()
    joint_ids = [_joint_index(robot, name, index) for index, name in enumerate((*_ARM_NAMES, *_PHYSICAL_GRIPPER_NAMES))]
    physical = full[:, joint_ids]
    aperture = (physical[:, -1:] - physical[:, -2:-1]) * 0.5
    return np.concatenate([physical[:, : len(_ARM_NAMES)], aperture], axis=-1)


def _logical_to_physical_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != _ACTION_DIM:
        raise ValueError(f"expected {_ACTION_DIM} logical PSM values, got {values.shape[-1]}")
    arm = values[:, : len(_ARM_NAMES)]
    aperture = values[:, -1:]
    return torch.cat([arm, -aperture, aperture], dim=-1)


def _policy_action(ctx) -> torch.Tensor | None:
    row = ctx.io.pop_action()
    if row is None:
        return None
    return torch.as_tensor(row, device=ctx.env.device, dtype=torch.float32)


def _joint_index(robot, joint_name: str, fallback: int) -> int:
    if hasattr(robot, "find_joints"):
        found = robot.find_joints(joint_name, preserve_order=True)
        for value in (found[0], found[1]):
            if hasattr(value, "numel") and value.numel() > 0:
                return int(value[0])
            if len(value) > 0 and isinstance(value[0], int):
                return int(value[0])
    return fallback
