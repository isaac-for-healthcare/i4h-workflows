# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-dimensional PPO objective for ultrasound probe target reaching."""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from i4h_arena.tensor_utils import to_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

POSITION_TOLERANCE_M = 0.02
ORIENTATION_TOLERANCE_RAD = math.radians(10.0)
SUCCESS_HOLD_STEPS = 20

# Probe goals sampled directly from the upper-surface vertices of the shipped
# abdominal phantom. Each z value includes a 5 mm offset for the marker center,
# keeping the intended TCP pose just above the skin instead of on the table or
# floating over a rectangular approximation of the curved torso.
TORSO_TARGET_POSES = (
    (0.586714, -0.079570, 0.196037),
    (0.599361, -0.064474, 0.196182),
    (0.599774, -0.081807, 0.196065),
    (0.600238, -0.063867, 0.196150),
    (0.600454, -0.073684, 0.195948),
    (0.610764, -0.085254, 0.195921),
    (0.612643, -0.077850, 0.195937),
    (0.613564, -0.073620, 0.195862),
)


def _poses(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probe: FrameTransformer = env.scene["ee_frame"]
    target: RigidObject = env.scene["target"]
    probe_pos = to_torch(probe.data.target_pos_w)[:, 0, :]
    probe_quat = to_torch(probe.data.target_quat_w)[:, 0, :]
    target_pos = to_torch(target.data.root_pos_w)
    target_quat = to_torch(target.data.root_quat_w)
    return probe_pos, probe_quat, target_pos, target_quat


def probe_pose_in_env(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Probe position relative to its cloned env origin, followed by quaternion."""
    probe_pos, probe_quat, _, _ = _poses(env)
    return torch.cat((probe_pos - env.scene.env_origins, probe_quat), dim=-1)


def target_pose_in_env(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target position relative to its cloned env origin, followed by quaternion."""
    _, _, target_pos, target_quat = _poses(env)
    return torch.cat((target_pos - env.scene.env_origins, target_quat), dim=-1)


def position_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    probe_pos, _, target_pos, _ = _poses(env)
    return torch.linalg.norm(probe_pos - target_pos, dim=-1)


def orientation_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    _, probe_quat, _, target_quat = _poses(env)
    return quat_error_magnitude(probe_quat, target_quat)


def position_reward(env: ManagerBasedRLEnv, scale_m: float) -> torch.Tensor:
    return torch.exp(-position_error(env) / scale_m)


def orientation_reward(env: ManagerBasedRLEnv, scale_rad: float) -> torch.Tensor:
    scaled = orientation_error(env) / scale_rad
    return 1.0 / (1.0 + scaled * scaled)


def within_tolerance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (position_error(env) <= POSITION_TOLERANCE_M) & (orientation_error(env) <= ORIENTATION_TOLERANCE_RAD)


def reset_success_counter(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    if not hasattr(env, "_probe_reach_success_steps"):
        env._probe_reach_success_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._probe_reach_success_steps[env_ids] = 0


def reset_target_on_torso(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Place each target on a measured upper-torso surface point."""
    if len(env_ids) == 0:
        return
    target: RigidObject = env.scene["target"]
    root_state = to_torch(target.data.default_root_state)[env_ids].clone()
    surface_poses = torch.tensor(TORSO_TARGET_POSES, dtype=root_state.dtype, device=env.device)
    sample_ids = torch.randint(surface_poses.shape[0], (len(env_ids),), device=env.device)
    root_state[:, :3] = surface_poses[sample_ids] + env.scene.env_origins[env_ids]
    target.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
    target.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((len(env_ids), 6), dtype=root_state.dtype, device=env.device),
        env_ids=env_ids,
    )


def probe_reach_success(env: ManagerBasedRLEnv, *, rl_training: bool) -> torch.Tensor:
    """Require the pose tolerance to remain satisfied for consecutive steps."""
    if not hasattr(env, "_probe_reach_success_steps"):
        env._probe_reach_success_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    qualified = within_tolerance(env)
    env._probe_reach_success_steps = torch.where(
        qualified,
        env._probe_reach_success_steps + 1,
        torch.zeros_like(env._probe_reach_success_steps),
    )
    success = env._probe_reach_success_steps >= SUCCESS_HOLD_STEPS
    return torch.zeros_like(success) if rl_training else success


@configclass
class _TaskObservationsCfg:
    @configclass
    class TaskObsCfg(ObservationGroupCfg):
        probe_pose = ObservationTermCfg(func=probe_pose_in_env)
        target_pose = ObservationTermCfg(func=target_pose_in_env)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    task_obs: TaskObsCfg = TaskObsCfg()


@configclass
class _EventsCfg:
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_target = EventTermCfg(func=reset_target_on_torso, mode="reset")
    reset_robot = EventTermCfg(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.98, 1.02),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )
    reset_success = EventTermCfg(func=reset_success_counter, mode="reset")


@configclass
class _RewardsCfg:
    position = RewardTermCfg(func=position_reward, weight=3.0, params={"scale_m": 0.10})
    position_fine = RewardTermCfg(func=position_reward, weight=4.0, params={"scale_m": 0.025})
    orientation = RewardTermCfg(func=orientation_reward, weight=2.0, params={"scale_rad": 0.35})
    success = RewardTermCfg(func=within_tolerance, weight=5.0)
    action_smoothness = RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewardTermCfg(func=base_mdp.joint_vel_l2, weight=-0.0001)


@configclass
class _TerminationsCfg:
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    success: TerminationTermCfg = MISSING


class UltrasoundProbeReachTask(TaskBase):
    """Reach and align the probe with a randomized point on the phantom."""

    def __init__(self, *, rl_training_mode: bool, episode_length_s: float = 5.0):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Reach and align the probe with the marked point on the torso.",
        )
        self._rl_training_mode = rl_training_mode

    def get_scene_cfg(self):
        return None

    def get_observation_cfg(self):
        return _TaskObservationsCfg()

    def get_events_cfg(self):
        return _EventsCfg()

    def get_rewards_cfg(self):
        return _RewardsCfg()

    def get_termination_cfg(self):
        return _TerminationsCfg(
            success=TerminationTermCfg(
                func=probe_reach_success,
                time_out=False,
                params={"rl_training": self._rl_training_mode},
            )
        )

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.7, 1.35, 1.05), lookat=(0.48, -0.02, 0.26))

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg
