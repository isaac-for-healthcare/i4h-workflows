# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the remote GR00T clients share: per-env chunk replay and LeRobot joint units."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from i4h_common.config import get_robot_config
from i4h_common.joint_utils import isaaclab_rad_to_lerobot, lerobot_to_isaaclab_rad


class ChunkReplay:
    """Infer an action chunk for an env when its last one is spent, then step through it.

    One request per env, as Arena's remote policies make them. Subclasses set
    ``horizon`` and implement ``_infer``.
    """

    horizon: int
    _chunk: torch.Tensor | None = None
    _index: torch.Tensor | None = None

    def _infer(self, observation: dict, env_id: int, action_dim: int) -> torch.Tensor:
        """The next ``(steps, action_dim)`` chunk for one env."""
        raise NotImplementedError

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        env = env.unwrapped
        n, action_dim = env.num_envs, env.action_manager.total_action_dim
        if self._chunk is None:
            self._chunk = torch.zeros(n, self.horizon, action_dim)
            self._index = torch.full((n,), self.horizon, dtype=torch.long)
        for env_id in range(n):
            if self._index[env_id] >= self.horizon:
                self._chunk[env_id] = self._infer(observation, env_id, action_dim)[: self.horizon]
                self._index[env_id] = 0
        action = self._chunk[torch.arange(n), self._index]
        self._index += 1
        return action.to(env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._index is not None:
            self._index[slice(None) if env_ids is None else env_ids.cpu()] = self.horizon


def to_lerobot(joints: np.ndarray, robot: str) -> np.ndarray:
    """Isaac Lab radians to the robot's calibrated LeRobot joint coordinates."""
    config = get_robot_config(robot)
    return isaaclab_rad_to_lerobot(joints, config.isaaclab_joint_pos_limit_range, config.lerobot_joint_pos_limit_range)


def from_lerobot(joints: np.ndarray, robot: str) -> np.ndarray:
    """The robot's calibrated LeRobot joint coordinates to Isaac Lab radians."""
    config = get_robot_config(robot)
    return lerobot_to_isaaclab_rad(joints, config.lerobot_joint_pos_limit_range, config.isaaclab_joint_pos_limit_range)


def uint8_image(image: torch.Tensor) -> torch.Tensor:
    """A camera image as ``uint8``, scaling a ``[0, 1]`` float image."""
    if image.is_floating_point():
        image = (image * 255.0 if image.max() <= 1.0 else image).clamp(0, 255)
    return image.to(torch.uint8)
