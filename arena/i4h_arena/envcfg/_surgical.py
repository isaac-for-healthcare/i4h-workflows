# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic surgical tool-tip target and legacy reach-command tasks."""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, subtract_frame_transforms
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from i4h_arena.tensor_utils import to_torch

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

# Workflow reach rollouts last five seconds and Locate intentionally snapshots
# the sampled command. Keep that command stable until the episode resets so
# control and success never refer to different targets.
_REACH_COMMAND_HOLD_S = 6.0


def _root_pose_w(robot) -> tuple[torch.Tensor, torch.Tensor]:
    root_state_w = to_torch(robot.data.root_state_w)
    return root_state_w[:, :3], root_state_w[:, 3:7]


def _body_pose_w(robot, body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    body_state_w = to_torch(robot.data.body_state_w)
    return body_state_w[:, body_id, :3], body_state_w[:, body_id, 3:7]


def _ee_position(env) -> torch.Tensor:
    return to_torch(env.scene["ee_frame"].data.target_pos_w)[:, 0, :]


def tcp_pose_in_root(env, frame_name: str = "ee_frame", robot_name: str = "robot") -> tuple[torch.Tensor, torch.Tensor]:
    """Position and ``xyzw`` orientation of ``frame_name``'s tool tip in ``robot_name``'s root frame."""
    data = env.scene[frame_name].data
    root_pos_w, root_quat_w = _root_pose_w(env.scene[robot_name])
    return subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        to_torch(data.target_pos_w)[:, 0, :],
        to_torch(data.target_quat_w)[:, 0, :],
    )


def tcps_reached_commands(env, arms: tuple[tuple[str, str, str], ...], threshold: float) -> torch.Tensor:
    """Whether every ``(command, frame, robot)`` arm's tool tip is within ``threshold`` of its command [m]."""
    reached = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for command_name, frame_name, robot_name in arms:
        tcp_pos, _ = tcp_pose_in_root(env, frame_name, robot_name)
        target_pos = to_torch(env.command_manager.get_command(command_name))[:, :3]
        reached &= torch.linalg.norm(tcp_pos - target_pos, dim=-1) < threshold
    return reached


def object_above_root(env, height: float, object_name: str = "object", robot_name: str = "robot") -> torch.Tensor:
    """Whether ``object_name``'s root is above ``height`` [m] in ``robot_name``'s root frame."""
    root_pos_w, root_quat_w = _root_pose_w(env.scene[robot_name])
    object_pos, _ = subtract_frame_transforms(root_pos_w, root_quat_w, to_torch(env.scene[object_name].data.root_pos_w))
    return object_pos[:, 2] > height


def object_ee_distance(env, std: float) -> torch.Tensor:
    object_pos_w = to_torch(env.scene["object"].data.root_pos_w)
    ee_pos_w = _ee_position(env)
    return 1.0 - torch.tanh(torch.linalg.norm(object_pos_w - ee_pos_w, dim=1) / std)


def object_is_lifted(env, minimal_height: float) -> torch.Tensor:
    return torch.where(
        to_torch(env.scene["object"].data.root_pos_w)[:, 2] > minimal_height,
        torch.ones(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )


def object_goal_distance(env, std: float, minimal_height: float, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    root_pos_w, root_quat_w = _root_pose_w(robot)
    goal_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, command[:, :3], command[:, 3:7])
    object_pos_w = to_torch(env.scene["object"].data.root_pos_w)
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


def _psm_reach_pose_command_cfg(
    asset_name: str = "robot",
    debug_vis: bool = True,
    resampling_time_range: tuple[float, float] = (4.0, 4.0),
) -> mdp.UniformPoseCommandCfg:
    return _pose_command_cfg(
        asset_name=asset_name,
        body_name="psm_tool_tip_link",
        pos_x=(-0.07, 0.07),
        pos_y=(-0.07, 0.07),
        pos_z=(-0.12, -0.08),
        resampling_time_range=resampling_time_range,
        debug_vis=debug_vis,
    )


def _star_reach_pose_command_cfg(
    resampling_time_range: tuple[float, float] = (4.0, 4.0),
) -> mdp.UniformPoseCommandCfg:
    return _pose_command_cfg(
        asset_name="robot",
        body_name="endo360_needle",
        # Independent box sampling in the source includes a far-and-low
        # corner outside STAR's fixed-orientation IK workspace.
        pos_x=(0.45, 0.52),
        pos_y=(0.0, 0.3),
        pos_z=(0.26, 0.4),
        pitch=((2 / 3) * math.pi, (2 / 3) * math.pi),
        resampling_time_range=resampling_time_range,
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
            # The source's 1.5x upper bound clips joints 4 and 6 against
            # their hard limits before DLS begins. Keep reset diversity in
            # the stable interior basin seen in the source trajectories.
            "position_range": (0.75, 0.9),
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


#: Per reach mode: the ``(command, frame, robot)`` of each arm and the success radius [m].
_REACH_SUCCESS: dict[str, tuple[tuple[tuple[str, str, str], ...], float]] = {
    "psm": ((("ee_pose", "ee_frame", "robot"),), 0.01),
    "star": ((("ee_pose", "ee_frame", "robot"),), 0.03),
    "dual_psm": ((("ee_1_pose", "ee_frame", "robot_1"), ("ee_2_pose", "ee_2_frame", "robot_2")), 0.01),
}


class SurgicalReachEnvCfg(TaskBase):
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
    def psm(cls) -> SurgicalReachEnvCfg:
        return cls(
            task_description="Move the PSM tool tip to sampled reach poses.",
            mode="psm",
            viewer_eye=(1.20, 1.20, 0.90),
            viewer_lookat=(0.0, 0.0, 0.06),
        )

    @classmethod
    def dual_psm(cls) -> SurgicalReachEnvCfg:
        return cls(
            task_description="Move both dVRK PSM tool tips through coordinated sampled reach poses.",
            mode="dual_psm",
            viewer_eye=(0.0, 1.50, 0.90),
            viewer_lookat=(0.0, 0.0, 0.07),
        )

    @classmethod
    def star(cls) -> SurgicalReachEnvCfg:
        return cls(
            task_description="Move the STAR tool tip to sampled reach poses.",
            mode="star",
            viewer_eye=(2.0, 2.0, 1.0),
            viewer_lookat=(0.0, 0.0, 0.0),
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Succeed once every tool tip is within the mode's radius of its sampled target."""
        arms, threshold = _REACH_SUCCESS[self.mode]
        reached = TerminationTermCfg(func=tcps_reached_commands, params={"arms": arms, "threshold": threshold})
        return TaskTerminationCfg(
            timeout_s=self.get_episode_length_s(),
            success=[CompletionCriteria(name="reach", predicate_sequence=[reached])],
        )

    def get_events_cfg(self):
        if self.mode == "dual_psm":
            return _DualPsmReachEventsCfg()
        if self.mode == "star":
            return _StarReachEventsCfg()
        return _PsmReachEventsCfg()

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
            hold = (_REACH_COMMAND_HOLD_S, _REACH_COMMAND_HOLD_S)
            cfg.ee_1_pose = _psm_reach_pose_command_cfg(
                asset_name="robot_1",
                debug_vis=False,
                resampling_time_range=hold,
            )
            cfg.ee_2_pose = _psm_reach_pose_command_cfg(
                asset_name="robot_2",
                debug_vis=False,
                resampling_time_range=hold,
            )
            return cfg
        cfg = _CommandsCfg()
        hold = (_REACH_COMMAND_HOLD_S, _REACH_COMMAND_HOLD_S)
        cfg.ee_pose = (
            _star_reach_pose_command_cfg(resampling_time_range=hold)
            if self.mode == "star"
            else _psm_reach_pose_command_cfg(resampling_time_range=hold)
        )
        return cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=self.viewer_eye, lookat=self.viewer_lookat or (0.0, 0.0, 0.0))

    def tune_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg.scene.env_spacing = self.env_spacing
        return env_cfg


class SurgicalLiftEnvCfg(TaskBase):
    """Original robotic_surgery pick-and-lift command task."""

    def __init__(
        self,
        *,
        task_description: str,
        success_height: float,
        organs: bool = False,
        episode_length_s: float = 5.0,
        env_spacing: float = 2.5,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.success_height = success_height
        self.organs = organs
        self.env_spacing = env_spacing

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Succeed once the object is above ``success_height`` in the robot root frame; fail if it falls."""
        lifted = TerminationTermCfg(func=object_above_root, params={"height": self.success_height})
        return TaskTerminationCfg(
            timeout_s=self.get_episode_length_s(),
            success=[CompletionCriteria(name="lift", predicate_sequence=[lifted])],
            failures={
                "object_dropping": TerminationTermCfg(
                    func=mdp.root_height_below_minimum,
                    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
                )
            },
        )

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
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.organs:
            return ViewerCfg(eye=(-1.10, 0.80, 0.80), lookat=(0.0, 0.0, 0.06))
        return ViewerCfg(eye=(0.65, 0.65, 0.50), lookat=(0.0, 0.0, 0.05))

    def tune_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg.scene.env_spacing = self.env_spacing
        return env_cfg
