# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""STAR reach runtime helpers."""

from __future__ import annotations

import numpy as np
import torch
from arena.runtimes.core.base import PolicyIO, run_policy_episode
from common.config import get_robot_config, get_zenoh_config

_STAR_CONFIG = get_robot_config("star")
_JOINT_NAMES = (*_STAR_CONFIG.body_joint_names, *_STAR_CONFIG.hand_joint_names)
_ACTION_DIM = _STAR_CONFIG.action_dim


class SurgicalReachStarPolicyIO(PolicyIO):
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
    ctx.io.publish_state(_joint_positions(scene["robot"]).flatten())


def run_policy_based_episode(ctx, *, max_timesteps: int) -> str:
    return run_policy_episode(
        ctx,
        max_timesteps=max_timesteps,
        publish_obs=publish_obs,
        policy_action=_policy_action,
        stop_on_env_done=True,
    )


def _joint_positions(robot) -> np.ndarray:
    full = robot.data.joint_pos[0:1].detach().cpu().numpy()
    joint_ids = [_joint_index(robot, name, index) for index, name in enumerate(_JOINT_NAMES)]
    return full[:, joint_ids]


def _joint_index(robot, joint_name: str, fallback: int) -> int:
    if hasattr(robot, "find_joints"):
        found = robot.find_joints(joint_name, preserve_order=True)
        for value in (found[0], found[1]):
            if hasattr(value, "numel") and value.numel() > 0:
                return int(value[0])
            if len(value) > 0 and isinstance(value[0], int):
                return int(value[0])
    return fallback


def _policy_action(ctx) -> torch.Tensor | None:
    row = ctx.io.pop_action()
    if row is None:
        return None
    return torch.as_tensor(row, device=ctx.env.device, dtype=torch.float32)
