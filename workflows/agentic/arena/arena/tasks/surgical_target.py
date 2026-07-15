# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic surgical tool-tip target and old reach-command tasks."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import torch
from arena.assets.surgical_scenes import REACH_TARGET_POS
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.tasks.task_base import TaskBase

_SUCCESS_TOLERANCE_M = 0.025


def _root_pose_w(robot) -> tuple[torch.Tensor, torch.Tensor]:
    root_state_w = getattr(robot.data, "root_state_w", None)
    if root_state_w is not None:
        return root_state_w[:, :3], root_state_w[:, 3:7]
    return robot.data.root_pos_w, robot.data.root_quat_w


def _body_pose_w(robot, body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    body_state_w = getattr(robot.data, "body_state_w", None)
    if body_state_w is not None:
        return body_state_w[:, body_id, :3], body_state_w[:, body_id, 3:7]
    return robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id]


def _ee_position(env) -> torch.Tensor:
    return env.scene["ee_frame"].data.target_pos_w[:, 0, :]


def _target_position(env) -> torch.Tensor:
    target = torch.tensor(REACH_TARGET_POS, dtype=torch.float32, device=env.device).unsqueeze(0)
    return target + env.scene.env_origins


def target_distance(env) -> torch.Tensor:
    return torch.linalg.norm(_ee_position(env) - _target_position(env), dim=-1)


def target_success(env, tolerance_m: float = _SUCCESS_TOLERANCE_M) -> torch.Tensor:
    distance = target_distance(env)
    success = distance <= tolerance_m
    env._surgical_target_success_last = {
        "ee_pos": _ee_position(env).detach().clone(),
        "target_pos": _target_position(env).detach().clone(),
        "distance_m": distance.detach().clone(),
        "tolerance_m": tolerance_m,
        "success": success.detach().clone(),
    }
    return success


def target_reward(env) -> torch.Tensor:
    return 1.0 / (1.0 + target_distance(env).square())


def object_ee_distance(env, std: float) -> torch.Tensor:
    object_pos_w = env.scene["object"].data.root_pos_w
    ee_pos_w = _ee_position(env)
    return 1.0 - torch.tanh(torch.linalg.norm(object_pos_w - ee_pos_w, dim=1) / std)


def object_is_lifted(env, minimal_height: float) -> torch.Tensor:
    return torch.where(
        env.scene["object"].data.root_pos_w[:, 2] > minimal_height,
        torch.ones(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )


def object_goal_distance(env, std: float, minimal_height: float, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    root_pos_w, root_quat_w = _root_pose_w(robot)
    goal_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, command[:, :3], command[:, 3:7])
    object_pos_w = env.scene["object"].data.root_pos_w
    distance = torch.linalg.norm(goal_pos_w - object_pos_w, dim=1)
    return torch.where(
        object_pos_w[:, 2] > minimal_height,
        1.0 - torch.tanh(distance / std),
        torch.zeros(env.num_envs, device=env.device),
    )


def command_position_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    root_pos_w, root_quat_w = _root_pose_w(robot)
    des_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, command[:, :3], command[:, 3:7])
    curr_pos_w, _ = _body_pose_w(robot, asset_cfg.body_ids[0])
    return torch.linalg.norm(curr_pos_w - des_pos_w, dim=1)


def command_orientation_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, root_quat_w = _root_pose_w(robot)
    des_quat_w = quat_mul(root_quat_w, command[:, 3:7])
    _, curr_quat_w = _body_pose_w(robot, asset_cfg.body_ids[0])
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def _pose_command_cfg(
    *,
    asset_name: str,
    body_name: str,
    pos_x: tuple[float, float],
    pos_y: tuple[float, float],
    pos_z: tuple[float, float],
    roll: tuple[float, float] = (0.0, 0.0),
    pitch: tuple[float, float] = (0.0, 0.0),
    yaw: tuple[float, float] = (0.0, 0.0),
    resampling_time_range: tuple[float, float] = (4.0, 4.0),
    debug_vis: bool = False,
    marker_scale: tuple[float, float, float] = (0.01, 0.01, 0.01),
) -> mdp.UniformPoseCommandCfg:
    cfg = mdp.UniformPoseCommandCfg(
        asset_name=asset_name,
        body_name=body_name,
        resampling_time_range=resampling_time_range,
        debug_vis=debug_vis,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        ),
    )
    cfg.goal_pose_visualizer_cfg.markers["frame"].scale = marker_scale
    cfg.current_pose_visualizer_cfg.markers["frame"].scale = marker_scale
    return cfg


def _psm_reach_pose_command_cfg(asset_name: str = "robot", debug_vis: bool = True) -> mdp.UniformPoseCommandCfg:
    return _pose_command_cfg(
        asset_name=asset_name,
        body_name="psm_tool_tip_link",
        pos_x=(-0.07, 0.07),
        pos_y=(-0.07, 0.07),
        pos_z=(-0.12, -0.08),
        debug_vis=debug_vis,
    )


def _star_reach_pose_command_cfg() -> mdp.UniformPoseCommandCfg:
    return _pose_command_cfg(
        asset_name="robot",
        body_name="endo360_needle",
        pos_x=(0.45, 0.55),
        pos_y=(0.0, 0.3),
        pos_z=(0.2, 0.4),
        pitch=((2 / 3) * math.pi, (2 / 3) * math.pi),
        debug_vis=True,
        marker_scale=(0.1, 0.1, 0.1),
    )


def _lift_object_pose_command_cfg() -> mdp.UniformPoseCommandCfg:
    return _pose_command_cfg(
        asset_name="robot",
        body_name="psm_tool_tip_link",
        pos_x=(-0.05, 0.05),
        pos_y=(-0.05, 0.05),
        pos_z=(-0.12, -0.12),
        resampling_time_range=(1.0, 1.0),
        debug_vis=False,
    )


@configclass
class _CommandsCfg:
    ee_pose: mdp.UniformPoseCommandCfg = MISSING


@configclass
class _DualCommandsCfg:
    ee_1_pose: mdp.UniformPoseCommandCfg = MISSING
    ee_2_pose: mdp.UniformPoseCommandCfg = MISSING


@configclass
class _LiftCommandsCfg:
    object_pose: mdp.UniformPoseCommandCfg = MISSING


@configclass
class _EventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class _PsmReachEventsCfg:
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.01, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class _StarReachEventsCfg:
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class _DualPsmReachEventsCfg:
    reset_robot_1_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1"),
            "position_range": (0.01, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_robot_2_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_2"),
            "position_range": (0.01, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class _TargetRewardsCfg:
    reaching = RewardTermCfg(func=target_reward, weight=1.0)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class _CommandRewardsCfg:
    end_effector_position_tracking = RewardTermCfg(
        func=command_position_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["psm_tool_tip_link"]), "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewardTermCfg(
        func=command_orientation_error,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["psm_tool_tip_link"]), "command_name": "ee_pose"},
    )
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class _DualCommandRewardsCfg:
    end_effector_1_position_tracking = RewardTermCfg(
        func=command_position_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot_1", body_names=["psm_tool_tip_link"]), "command_name": "ee_1_pose"},
    )
    end_effector_1_orientation_tracking = RewardTermCfg(
        func=command_orientation_error,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot_1", body_names=["psm_tool_tip_link"]), "command_name": "ee_1_pose"},
    )
    end_effector_2_position_tracking = RewardTermCfg(
        func=command_position_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot_2", body_names=["psm_tool_tip_link"]), "command_name": "ee_2_pose"},
    )
    end_effector_2_orientation_tracking = RewardTermCfg(
        func=command_orientation_error,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot_2", body_names=["psm_tool_tip_link"]), "command_name": "ee_2_pose"},
    )
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-4)
    joint_1_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot_1")})
    joint_2_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot_2")})


@configclass
class _LiftRewardsCfg:
    reaching_object = RewardTermCfg(func=object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewardTermCfg(func=object_is_lifted, params={"minimal_height": 0.02}, weight=15.0)
    object_goal_tracking = RewardTermCfg(
        func=object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewardTermCfg(
        func=object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=5.0,
    )
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-3)
    joint_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class _TargetTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success = TerminationTermCfg(func=target_success, time_out=False)


@configclass
class _TimeoutTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class _LiftTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    object_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class _LiftEventsCfg:
    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


class SurgicalTargetTask(TaskBase):
    """Surgical tool-tip target task used for native env smoke and validation."""

    def __init__(
        self,
        task_description: str,
        episode_length_s: float = 5.0,
        env_spacing: float = 2.5,
        use_old_reach_command: bool = False,
        terminate_on_success: bool = True,
        viewer_eye: tuple[float, float, float] | None = None,
        viewer_lookat: tuple[float, float, float] | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.env_spacing = env_spacing
        self.use_old_reach_command = use_old_reach_command
        self.terminate_on_success = terminate_on_success
        self.viewer_eye = viewer_eye
        self.viewer_lookat = viewer_lookat

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        if self.terminate_on_success:
            return _TargetTerminationsCfg()
        return _TimeoutTerminationsCfg()

    def get_events_cfg(self):
        if self.mode == "dual_psm":
            return _DualPsmReachEventsCfg()
        if self.mode == "star":
            return _StarReachEventsCfg()
        return _PsmReachEventsCfg()

    def get_rewards_cfg(self):
        if self.use_old_reach_command:
            return _CommandRewardsCfg()
        return _TargetRewardsCfg()

    def get_commands_cfg(self):
        if not self.use_old_reach_command:
            return None
        cfg = _CommandsCfg()
        cfg.ee_pose = _psm_reach_pose_command_cfg()
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self):
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(
            eye=self.viewer_eye or (0.35, 0.35, 0.25),
            lookat=self.viewer_lookat or (0.0, 0.0, 0.02),
        )

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg.scene.env_spacing = self.env_spacing
        return env_cfg


class SurgicalReachCommandTask(TaskBase):
    """Original robotic_surgery reach command task."""

    def __init__(
        self,
        *,
        task_description: str,
        mode: str,
        episode_length_s: float = 5.0,
        env_spacing: float = 2.5,
        viewer_eye: tuple[float, float, float],
        viewer_lookat: tuple[float, float, float] | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.mode = mode
        self.env_spacing = env_spacing
        self.viewer_eye = viewer_eye
        self.viewer_lookat = viewer_lookat

    @classmethod
    def psm(cls) -> "SurgicalReachCommandTask":
        return cls(
            task_description="Move the PSM tool tip to sampled reach poses.",
            mode="psm",
            viewer_eye=(0.2, 0.2, 0.1),
            viewer_lookat=(0.0, 0.0, 0.04),
        )

    @classmethod
    def dual_psm(cls) -> "SurgicalReachCommandTask":
        return cls(
            task_description="Move both dVRK PSM tool tips through coordinated sampled reach poses.",
            mode="dual_psm",
            viewer_eye=(0.0, 0.5, 0.2),
            viewer_lookat=(0.0, 0.0, 0.05),
        )

    @classmethod
    def star(cls) -> "SurgicalReachCommandTask":
        return cls(
            task_description="Move the STAR tool tip to sampled reach poses.",
            mode="star",
            viewer_eye=(2.0, 2.0, 1.0),
            viewer_lookat=(0.0, 0.0, 0.0),
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return _TimeoutTerminationsCfg()

    def get_events_cfg(self):
        return _EventsCfg()

    def get_rewards_cfg(self):
        if self.mode == "dual_psm":
            return _DualCommandRewardsCfg()
        rewards = _CommandRewardsCfg()
        if self.mode == "star":
            rewards.end_effector_position_tracking.params["asset_cfg"] = SceneEntityCfg(
                "robot", body_names=["endo360_needle"]
            )
            rewards.end_effector_orientation_tracking.params["asset_cfg"] = SceneEntityCfg(
                "robot", body_names=["endo360_needle"]
            )
        return rewards

    def get_commands_cfg(self):
        if self.mode == "dual_psm":
            cfg = _DualCommandsCfg()
            cfg.ee_1_pose = _psm_reach_pose_command_cfg(asset_name="robot_1", debug_vis=False)
            cfg.ee_2_pose = _psm_reach_pose_command_cfg(asset_name="robot_2", debug_vis=False)
            return cfg
        cfg = _CommandsCfg()
        cfg.ee_pose = _star_reach_pose_command_cfg() if self.mode == "star" else _psm_reach_pose_command_cfg()
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self):
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=self.viewer_eye, lookat=self.viewer_lookat or (0.0, 0.0, 0.0))

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg.scene.env_spacing = self.env_spacing
        if self.mode == "dual_psm":
            env_cfg.scene.replicate_physics = False
        return env_cfg


class SurgicalLiftTask(TaskBase):
    """Original robotic_surgery pick-and-lift command task."""

    def __init__(
        self,
        *,
        task_description: str,
        organs: bool = False,
        episode_length_s: float = 5.0,
        env_spacing: float = 2.5,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.organs = organs
        self.env_spacing = env_spacing

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return _LiftTerminationsCfg()

    def get_events_cfg(self):
        cfg = _LiftEventsCfg()
        if self.organs:
            cfg.reset_object_position.params["pose_range"] = {"x": (-0.03, 0.02), "y": (-0.01, 0.01), "z": (0.0, 0.0)}
        return cfg

    def get_rewards_cfg(self):
        return _LiftRewardsCfg()

    def get_commands_cfg(self):
        cfg = _LiftCommandsCfg()
        cfg.object_pose = _lift_object_pose_command_cfg()
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self):
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.organs:
            return ViewerCfg(eye=(-0.32, 0.12, 0.12), lookat=(0.0, 0.0, 0.04))
        return ViewerCfg(eye=(0.2, 0.2, 0.1), lookat=(0.0, 0.0, 0.04))

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg.scene.env_spacing = self.env_spacing
        return env_cfg
