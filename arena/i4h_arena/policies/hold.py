# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hold every robot where the env reset it, for looking at a scene.

Arena's ``zero_action`` policy holds a robot only when zero means "stay": relative
actions, or joint positions offset by the default pose. The i4h arms also take absolute
joint positions and absolute Cartesian poses, where zero drives them away. This policy
commands each action term's value for the reset state instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class HoldPolicyCfg(PolicyCfg):
    """Configure a policy that holds the reset state."""


@register_policy
class HoldPolicy(PolicyBase[HoldPolicyCfg]):
    """Joint targets at the default pose, Cartesian targets at the reset tool pose, jaws open, G1 standing,
    Pink IK wrists and fingers where they are."""

    name = "i4h_hold"

    def __init__(self, config: HoldPolicyCfg):
        super().__init__(config)
        self._action: torch.Tensor | None = None
        self._stale: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._stale is None:
            return
        if env_ids is None:
            self._stale[:] = True
        else:
            self._stale[env_ids] = True

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        env = env.unwrapped
        if self._action is None:
            self._action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            self._stale = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if self._stale.any():
            fresh = _reset_state_action(env)
            self._action[self._stale] = fresh[self._stale]
            self._stale[:] = False
        return self._action


def _reset_state_action(env) -> torch.Tensor:
    columns = []
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        value = torch.zeros(env.num_envs, term.action_dim, device=env.device)
        if isinstance(term, JointPositionAction):
            default = term._asset.data.default_joint_pos.torch[:, term._joint_ids]
            value = (default - term._offset) / term._scale
        elif isinstance(term, DifferentialInverseKinematicsAction) and not term.cfg.controller.use_relative_mode:
            pos, quat = term._compute_frame_pose()
            pose = torch.cat([pos, quat], dim=-1)
            value = (pose - term._offset) / term._scale.clamp_min(1e-9)
        elif isinstance(term, BinaryJointAction):
            value[:] = 1.0
        elif hasattr(term, "_ik_controllers") and hasattr(term, "_hand_joint_ids"):
            # Isaac Lab's Pink IK: each frame task's link pose in the env frame (xyz, xyzw), then
            # the hand joints. Hold the measured wrists and fingers.
            data = term._asset.data
            poses = data.body_link_pose_w.torch.clone()
            poses[:, :, :3] -= env.scene.env_origins.unsqueeze(1)
            frames = [task.frame for task in term._ik_controllers[0].cfg.variable_input_tasks if hasattr(task, "frame")]
            links = [poses[:, data.body_names.index(frame)] for frame in frames]
            value = torch.cat([*links, data.joint_pos.torch[:, term._hand_joint_ids]], dim=-1)
        elif hasattr(term, "get_base_height_cmd_from_actions"):
            # Arena's G1 whole-body controller: joint targets, then navigate (3), base height (1)
            # and torso roll-pitch-yaw (3) commands. Stand still at the default pose and height.
            joints = term.num_joints
            value[:, :joints] = term._asset.data.default_joint_pos.torch[:, :joints]
            value[:, joints + term.navigate_cmd_dim] = float(getattr(term.cfg, "default_base_height", 0.75))
        columns.append(value)
    return torch.cat(columns, dim=-1)
