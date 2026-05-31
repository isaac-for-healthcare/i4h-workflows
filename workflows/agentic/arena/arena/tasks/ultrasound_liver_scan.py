# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ultrasound liver-scan task.

Single-file env recipe — MDP helpers (events, observations, rewards), small
configclasses (Terminations, Rewards, Events, Observations), and the
:class:`UltrasoundLiverScanTask` recipe consumed by the env.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.tasks.task_base import TaskBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

_ULTRASOUND_TARGET_TOLERANCE_M = 0.20
_ULTRASOUND_ALIGNMENT_THRESHOLD = 0.80
_ULTRASOUND_MIN_SCAN_STEPS = 245
_ULTRASOUND_CONTACT_Z_MAX_M = 0.22
_ULTRASOUND_TWIST_ALIGNMENT_THRESHOLD = 0.95
_ULTRASOUND_SCAN_DISTANCE_M = 0.10


# ---------- Reset events ------------------------------------------------------


def reset_panda_joints_by_fraction_of_limits(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fraction: float = 0.1,
) -> None:
    """Reset Panda joints with offsets sampled from a fraction of the joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_limits = asset.data.default_joint_limits[env_ids].clone()
    joint_sample_ranges = joint_limits * fraction

    lower = joint_sample_ranges[:, :, 0]
    upper = joint_sample_ranges[:, :, 1]
    joint_pos_delta = torch.rand(joint_pos.shape, device=joint_pos.device) * (upper - lower) + lower
    joint_pos += joint_pos_delta
    joint_pos = torch.clamp(joint_pos, joint_limits[:, :, 0], joint_limits[:, :, 1])

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_ultrasound_success_state(env: "ManagerBasedEnv", env_ids: torch.Tensor) -> None:
    attrs = (
        "_ultrasound_success_steps",
        "_ultrasound_contact_pos",
        "_ultrasound_twist_pos",
        "_ultrasound_contact_seen",
        "_ultrasound_twist_seen",
        "_ultrasound_max_scan_distance",
    )
    for attr in attrs:
        if hasattr(env, attr):
            getattr(env, attr)[env_ids] = 0
    if hasattr(env, "_ultrasound_success_last"):
        delattr(env, "_ultrasound_success_last")


# ---------- Observations ------------------------------------------------------


def object_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("organs"),
) -> torch.Tensor:
    """Organ XYZ position expressed in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], obj.data.root_pos_w[:, :3]
    )
    return object_pos_b


# ---------- Rewards -----------------------------------------------------------


def object_ee_distance(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("organs"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """Inverse-square reward for approaching the organ scan target above the phantom."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos_w = obj.data.root_pos_w + torch.tensor([0.0, -0.25, 1.0], device=obj.data.root_pos_w.device)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos_w - ee_w, dim=1)

    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def align_ee_handle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for aligning the EE z+x axes with the organ goal frame."""
    ee_frame_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    goal_frame_quat = env.scene["goal_frame"].data.target_quat_w[..., 0, :]
    ee_rot = matrix_from_quat(ee_frame_quat)
    goal_rot = matrix_from_quat(goal_frame_quat)
    goal_x, goal_z = goal_rot[..., 0], goal_rot[..., 2]
    ee_x, ee_z = ee_rot[..., 0], ee_rot[..., 2]
    align_z = torch.bmm(ee_z.unsqueeze(1), goal_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), goal_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


# ---------- Terminations ------------------------------------------------------


def ultrasound_scan_success(
    env: "ManagerBasedRLEnv",
    target_tolerance_m: float = _ULTRASOUND_TARGET_TOLERANCE_M,
    alignment_threshold: float = _ULTRASOUND_ALIGNMENT_THRESHOLD,
    min_scan_steps: int = _ULTRASOUND_MIN_SCAN_STEPS,
    contact_z_max_m: float = _ULTRASOUND_CONTACT_Z_MAX_M,
    twist_alignment_threshold: float = _ULTRASOUND_TWIST_ALIGNMENT_THRESHOLD,
    scan_distance_m: float = _ULTRASOUND_SCAN_DISTANCE_M,
) -> torch.Tensor:
    """Success after either direct target alignment or a completed scan trajectory."""
    target_pos, ee_pos, alignment = _ultrasound_success_metrics(env)
    distance = torch.linalg.norm(target_pos - ee_pos, dim=-1)
    near_target = (distance <= target_tolerance_m) & (alignment >= alignment_threshold)

    if not hasattr(env, "_ultrasound_success_steps"):
        env._ultrasound_success_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._ultrasound_contact_pos = torch.zeros(env.num_envs, 3, dtype=ee_pos.dtype, device=env.device)
        env._ultrasound_twist_pos = torch.zeros(env.num_envs, 3, dtype=ee_pos.dtype, device=env.device)
        env._ultrasound_contact_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ultrasound_twist_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ultrasound_max_scan_distance = torch.zeros(env.num_envs, dtype=ee_pos.dtype, device=env.device)

    env._ultrasound_success_steps += 1
    contact_now = ee_pos[:, 2] <= contact_z_max_m
    new_contact = contact_now & ~env._ultrasound_contact_seen
    env._ultrasound_contact_pos = torch.where(new_contact.unsqueeze(-1), ee_pos, env._ultrasound_contact_pos)
    env._ultrasound_contact_seen = env._ultrasound_contact_seen | contact_now

    twist_now = env._ultrasound_contact_seen & (alignment >= twist_alignment_threshold)
    new_twist = twist_now & ~env._ultrasound_twist_seen
    env._ultrasound_twist_pos = torch.where(new_twist.unsqueeze(-1), ee_pos, env._ultrasound_twist_pos)
    env._ultrasound_twist_seen = env._ultrasound_twist_seen | twist_now

    scan_distance = torch.linalg.norm(ee_pos[:, :2] - env._ultrasound_twist_pos[:, :2], dim=-1)
    env._ultrasound_max_scan_distance = torch.where(
        env._ultrasound_twist_seen,
        torch.maximum(env._ultrasound_max_scan_distance, scan_distance),
        env._ultrasound_max_scan_distance,
    )
    scan_complete = (
        (env._ultrasound_success_steps >= min_scan_steps)
        & env._ultrasound_contact_seen
        & env._ultrasound_twist_seen
        & (alignment >= alignment_threshold)
        & (env._ultrasound_max_scan_distance >= scan_distance_m)
    )
    success = near_target | scan_complete
    env._ultrasound_success_last = {
        "target_pos": target_pos.detach().clone(),
        "ee_pos": ee_pos.detach().clone(),
        "distance": distance.detach().clone(),
        "alignment": alignment.detach().clone(),
        "contact_seen": env._ultrasound_contact_seen.detach().clone(),
        "twist_seen": env._ultrasound_twist_seen.detach().clone(),
        "scan_distance": env._ultrasound_max_scan_distance.detach().clone(),
        "steps": env._ultrasound_success_steps.detach().clone(),
        "near_target": near_target.detach().clone(),
        "scan_complete": scan_complete.detach().clone(),
        "success": success.detach().clone(),
        "target_tolerance_m": target_tolerance_m,
        "alignment_threshold": alignment_threshold,
        "scan_distance_threshold_m": scan_distance_m,
    }
    return success


def _ultrasound_success_metrics(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scene = env.scene
    target_pos = scene["goal_frame"].data.target_pos_w[:, 0, :]
    ee_pos = scene["ee_frame"].data.target_pos_w[:, 0, :]
    ee_rot = matrix_from_quat(scene["ee_frame"].data.target_quat_w[:, 0, :])
    goal_rot = matrix_from_quat(scene["goal_frame"].data.target_quat_w[:, 0, :])
    align_z = torch.sum(ee_rot[..., 2] * goal_rot[..., 2], dim=-1)
    align_x = torch.sum(ee_rot[..., 0] * goal_rot[..., 0], dim=-1)
    alignment = torch.minimum(align_z, align_x)
    return target_pos, ee_pos, alignment


# ---------- Env-cfg recipe ----------------------------------------------------


@configclass
class _EventsCfg:
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_success_state = EventTermCfg(func=reset_ultrasound_success_state, mode="reset")
    reset_object_position = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.0, -0.0),
                "yaw": (-math.pi / 2, math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("organs"),
        },
    )
    reset_joint_position = EventTermCfg(
        func=reset_panda_joints_by_fraction_of_limits,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]), "fraction": 0.01},
    )


@configclass
class _RewardsCfg:
    reaching_object = RewardTermCfg(func=object_ee_distance, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewardTermCfg(func=align_ee_handle, weight=2.5)
    alive = RewardTermCfg(func=base_mdp.is_alive, weight=0.1)
    action_rate_l2 = RewardTermCfg(func=base_mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewardTermCfg(func=base_mdp.joint_vel_l2, weight=-0.0001)


@configclass
class _TerminationsCfg:
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    success = TerminationTermCfg(func=ultrasound_scan_success, time_out=False)


class UltrasoundLiverScanTask(TaskBase):
    """Arena task wrapper for the Franka liver-scan ultrasound objective."""

    def __init__(self, episode_length_s: float = 5.0):
        super().__init__(episode_length_s=episode_length_s, task_description="Perform a liver ultrasound.")

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return _TerminationsCfg()

    def get_events_cfg(self):
        return _EventsCfg()

    def get_rewards_cfg(self):
        return _RewardsCfg()

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self):
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.5, 1.3, 1.0), lookat=(0.0, 0.0, 0.0))

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg
