# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule-based ultrasound sweep for the Franka probe over the abdominal phantom.

Per env: note where the phantom is, approach a standoff above it with the probe
pointing down, touch down, sweep through two waypoints across it, then hold. Every
target is in the phantom's frame. The arm takes relative pose commands, so each step
commands the delta from the measured probe pose to the eased target. The scene's
scan-success term ends the episode; its time-out ends it otherwise. A move that has
not arrived within its settle budget keeps commanding its goal until the episode ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, subtract_frame_transforms
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from i4h_arena.envcfg._surgical import tcp_pose_in_root
from i4h_arena.policies._cartesian import ease_pose
from i4h_arena.tensor_utils import to_torch

#: Phantom-frame probe targets in order: approach, contact, then the sweep.
WAYPOINTS = (
    (0.0030, 0.0500, 0.2000),
    (0.0030, 0.0500, 0.0863),
    (-0.0310, -0.01075, 0.0800),
    (-0.0650, -0.0715, 0.0737),
)
#: Probe-down orientation in the phantom frame, ``(x, y, z, w)``.
PROBE_DOWN = (1.0, 0.0, 0.0, 0.0)

LOCATE, FIRST_MOVE = 0, 1
HOLD = FIRST_MOVE + len(WAYPOINTS)


@dataclass
class ProbeSweepPolicyCfg(PolicyCfg):
    approach_s: float = 0.6
    """Time to ease onto the approach and contact poses [s]."""

    sweep_s: float = 1.0
    """Time to ease to each sweep waypoint [s]."""

    position_tolerance: float = 0.05
    """Distance at which a move counts as arrived [m]; the loaded probe converges to about 4 cm."""


@register_policy
class ProbeSweepPolicy(PolicyBase[ProbeSweepPolicyCfg]):
    """Approach, touch down on and sweep across the scene's ``organs`` phantom."""

    name = "i4h_probe_sweep"

    def __init__(self, config: ProbeSweepPolicyCfg):
        super().__init__(config)
        self._phase: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._phase is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        self._phase[ids] = LOCATE
        self._tick[ids] = 0

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        env = env.unwrapped
        cfg = self.config
        n, device = env.num_envs, env.device
        if self._phase is None:
            self._phase = torch.full((n,), LOCATE, dtype=torch.long, device=device)
            self._tick = torch.zeros(n, dtype=torch.long, device=device)
            self._phantom = torch.zeros(n, 7, device=device)
            self._start = torch.zeros(n, 7, device=device)
            self._goal = torch.zeros(n, 7, device=device)

        dt = env.step_dt
        tcp_pos, tcp_quat = tcp_pose_in_root(env, "ee_frame", "robot")
        self._tick += 1
        done = torch.zeros(n, dtype=torch.bool, device=device)
        action = torch.zeros(n, env.action_manager.total_action_dim, device=device)

        locate = self._phase == LOCATE
        if locate.any():
            root, phantom = env.scene["robot"].data, env.scene["organs"].data
            pos, quat = subtract_frame_transforms(
                to_torch(root.root_pos_w),
                to_torch(root.root_quat_w),
                to_torch(phantom.root_pos_w),
                to_torch(phantom.root_quat_w),
            )
            self._phantom[locate] = torch.cat([pos, quat], -1)[locate]
            done |= locate

        moving = (self._phase >= FIRST_MOVE) & (self._phase < HOLD)
        entering = moving & (self._tick == 1)
        if entering.any():
            index = (self._phase - FIRST_MOVE).clamp(0, len(WAYPOINTS) - 1)
            offset = torch.tensor(WAYPOINTS, device=device)[index]
            phantom_quat = self._phantom[:, 3:]
            goal_pos = self._phantom[:, :3] + quat_apply(phantom_quat, offset)
            goal_quat = quat_mul(phantom_quat, torch.tensor(PROBE_DOWN, device=device).expand(n, 4))
            self._start[entering] = torch.cat([tcp_pos, tcp_quat], -1)[entering]
            self._goal[entering] = torch.cat([goal_pos, goal_quat], -1)[entering]
        if moving.any():
            duration = torch.where(self._phase < FIRST_MOVE + 2, cfg.approach_s, cfg.sweep_s)
            steps = (duration / dt).round().clamp_min(1.0)
            alpha = self._tick.float() / steps
            pos, quat = ease_pose(self._start[:, :3], self._start[:, 3:], self._goal[:, :3], self._goal[:, 3:], alpha)
            delta = torch.cat([pos - tcp_pos, _rotation_vector(quat_mul(quat, quat_conjugate(tcp_quat)))], -1)
            action[moving] = delta[moving]
            arrived = (tcp_pos - self._goal[:, :3]).norm(dim=-1) < cfg.position_tolerance
            done |= moving & (alpha >= 1.0) & arrived

        # Holding: a zero delta keeps the measured pose until the scan-success term ends the episode.
        self._phase[done] += 1
        self._tick[done] = 0
        return action


def _rotation_vector(quat: torch.Tensor) -> torch.Tensor:
    """The rotation vector of ``(x, y, z, w)`` quaternions, without folding to the short arc.

    The relative IK command takes the rotation as given, so this keeps the angle the
    quaternion's sign implies rather than Isaac Lab's ``axis_angle_from_quat`` short arc.
    """
    vector = quat[:, :3]
    norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, quat[:, 3:4].clamp(-1.0, 1.0))
    return torch.where(norm > 1e-8, vector * angle / norm.clamp_min(1e-8), torch.zeros_like(vector))
