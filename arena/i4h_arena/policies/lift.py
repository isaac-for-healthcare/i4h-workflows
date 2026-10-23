# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule-based approach, grasp and lift for the dVRK PSM lift scenes.

Per env: hold the measured tool tip, wait for the object to settle and snapshot its
pose, approach a standoff above it, descend onto it, close the jaw, then lift. The
scene's success term ends the episode once the object is high enough; its time-out or
drop term ends it otherwise. A phase that has not converged within its settle budget
keeps commanding its goal until the episode ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from i4h_arena.envcfg._surgical import tcp_pose_in_root
from i4h_arena.policies._cartesian import arms_of, ease_pose
from i4h_arena.tensor_utils import to_torch

REST, LOCATE, APPROACH, DESCEND, GRASP, LIFT = range(6)


@dataclass
class CartesianLiftPolicyCfg(PolicyCfg):
    rest_s: float = 0.5
    """Time the arm holds its measured pose while the object settles [s]."""

    settle_timeout_s: float = 2.0
    """Longest wait for the object to settle (linear speed below 1 cm/s) before snapshotting it [s]."""

    standoff_m: float = 0.05
    """Height of the approach pose above the object [m]."""

    approach_s: float = 1.0
    descend_s: float = 0.6
    grasp_s: float = 0.4

    jaw_close: float = -0.16
    """Jaw action while grasping; negative closes the binary jaw."""

    lift_m: float = 0.10
    lift_s: float = 1.0

    position_tolerance: float = 0.04
    """Distance at which an approach or descent counts as arrived [m]. The organ bed uses 0.07."""


@register_policy
class CartesianLiftPolicy(PolicyBase[CartesianLiftPolicyCfg]):
    """Approach, descend, grasp and lift the scene's ``object`` with the first IK arm."""

    name = "i4h_cartesian_lift"

    def __init__(self, config: CartesianLiftPolicyCfg):
        super().__init__(config)
        self._phase: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._phase is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        self._phase[ids] = REST
        self._tick[ids] = 0

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        env = env.unwrapped
        cfg = self.config
        arm = arms_of(env)[0]
        n = env.num_envs
        if self._phase is None:
            self._phase = torch.full((n,), REST, dtype=torch.long, device=env.device)
            self._tick = torch.zeros(n, dtype=torch.long, device=env.device)
            self._command = torch.zeros(n, 7, device=env.device)
            self._object = torch.zeros(n, 7, device=env.device)
            self._start = torch.zeros(n, 7, device=env.device)
            self._goal = torch.zeros(n, 7, device=env.device)

        dt = env.step_dt
        tcp = torch.cat(tcp_pose_in_root(env, arm.frame, arm.robot), dim=-1)
        self._tick += 1
        done = torch.zeros(n, dtype=torch.bool, device=env.device)

        rest = self._phase == REST
        self._command[rest] = tcp[rest]
        done |= rest & (self._tick * dt >= cfg.rest_s)

        locate = self._phase == LOCATE
        if locate.any():
            obj = env.scene["object"].data
            settled = to_torch(obj.root_lin_vel_w).norm(dim=-1) < 0.01
            timed_out = self._tick * dt >= cfg.settle_timeout_s
            located = locate & (settled | timed_out)
            root = env.scene[arm.robot].data
            pos, quat = subtract_frame_transforms(
                to_torch(root.root_pos_w),
                to_torch(root.root_quat_w),
                to_torch(obj.root_pos_w),
                to_torch(obj.root_quat_w),
            )
            self._object[located] = torch.cat([pos, quat], -1)[located]
            done |= located

        moving = (self._phase == APPROACH) | (self._phase == DESCEND) | (self._phase == LIFT)
        entering = moving & (self._tick == 1)
        if entering.any():
            standoff = torch.zeros(n, 7, device=env.device)
            standoff[:, 2] = cfg.standoff_m
            lift = torch.zeros(n, 7, device=env.device)
            lift[:, 2] = cfg.lift_m
            goal = torch.where(
                (self._phase == APPROACH).unsqueeze(-1),
                self._object + standoff,
                torch.where((self._phase == DESCEND).unsqueeze(-1), self._object, tcp + lift),
            )
            goal[:, 3:] = torch.where((self._phase == LIFT).unsqueeze(-1), tcp[:, 3:], self._object[:, 3:])
            self._start[entering] = tcp[entering]
            self._goal[entering] = goal[entering]
        if moving.any():
            duration = torch.full((n,), cfg.lift_s, device=env.device)
            duration[self._phase == APPROACH] = cfg.approach_s
            duration[self._phase == DESCEND] = cfg.descend_s
            steps = (duration / dt).round().clamp_min(1.0)
            alpha = self._tick.float() / steps
            pos, quat = ease_pose(self._start[:, :3], self._start[:, 3:], self._goal[:, :3], self._goal[:, 3:], alpha)
            self._command[moving] = torch.cat([pos, quat], -1)[moving]
            arrived = (tcp[:, :3] - self._goal[:, :3]).norm(dim=-1) < cfg.position_tolerance
            # Lift is the last phase: the success term, not arrival, ends it.
            done |= moving & (self._phase != LIFT) & (alpha >= 1.0) & arrived

        grasp = self._phase == GRASP
        done |= grasp & (self._tick >= max(1, round(cfg.grasp_s / dt)))

        action = torch.zeros(n, env.action_manager.total_action_dim, device=env.device)
        action[:, arm.pose] = self._command
        if arm.jaw is not None:
            action[:, arm.jaw] = torch.where(self._phase >= GRASP, cfg.jaw_close, 1.0)

        self._phase[done] += 1
        self._tick[done] = 0
        return action
