# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble-trocar task.

Single-file env recipe for the G1 dex-hand assemble-trocar task. The file is
organised top-down:

1. Joint mappings (shared with ``arena.embodiments.g1``).
2. Observations.
3. Reset events.
4. Task-stage tracking + reward shaping.
5. Termination predicates.
6. Configclasses + the :class:`AssembleTrocarTask` recipe consumed by the env.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObject
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger("arena")


# ---------- Joint mappings ----------------------------------------------------


ASSEMBLE_TROCAR_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

ASSEMBLE_TROCAR_ACTION_OFFSETS = {
    "left_elbow_joint": -0.3,
    "right_elbow_joint": -0.3,
}
ASSEMBLE_TROCAR_HAND_JOINT_NAMES = ASSEMBLE_TROCAR_JOINT_NAMES[-14:]
_ASSEMBLE_TROCAR_HAND_OPEN_TOLERANCE_RAD = 0.25


# ---------- Observations ------------------------------------------------------


def get_assemble_trocar_joint_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids = _joint_ids(robot.data.joint_names, ASSEMBLE_TROCAR_JOINT_NAMES, env.device)
    return robot.data.joint_pos[:, joint_ids]


def get_robot_body_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    idx = torch.as_tensor(
        [
            0,
            3,
            6,
            9,
            13,
            17,
            1,
            4,
            7,
            10,
            14,
            18,
            2,
            5,
            8,
            11,
            15,
            19,
            21,
            23,
            25,
            27,
            12,
            16,
            20,
            22,
            24,
            26,
            28,
        ],
        dtype=torch.long,
        device=env.device,
    )
    return torch.cat((joint_pos[:, idx], joint_vel[:, idx], joint_torque[:, idx]), dim=1)


def get_robot_dex3_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    joint_pos = env.scene["robot"].data.joint_pos
    idx = torch.as_tensor(
        [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38],
        dtype=torch.long,
        device=env.device,
    )
    return joint_pos[:, idx]


# ---------- Reset events ------------------------------------------------------


def reset_task_stage(env: ManagerBasedRLEnv, env_ids: torch.Tensor, print_log: bool = False) -> None:
    if not hasattr(env, "_task_stage"):
        env._task_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._task_stage[env_ids] = 0
    for attr in ("_prev_stage_lift", "_prev_stage_tip", "_prev_stage_insert", "_prev_stage_place"):
        if hasattr(env, attr):
            getattr(env, attr)[env_ids] = 0
    if print_log:
        print(f"Reset assemble-trocar stage for {len(env_ids)} environment(s)")


def reset_tray_with_random_rotation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    tray_cfg: SceneEntityCfg,
    trocar_1_cfg: SceneEntityCfg,
    trocar_2_cfg: SceneEntityCfg,
    rotation_range: tuple[float, float] | list[float] | float = (0.0, 10.0),
) -> None:
    if len(env_ids) == 0:
        return
    if isinstance(rotation_range, (tuple, list)):
        min_angle_deg, max_angle_deg = float(rotation_range[0]), float(rotation_range[1])
    else:
        min_angle_deg, max_angle_deg = -float(rotation_range), float(rotation_range)

    tray = env.scene[tray_cfg.name]
    trocar_1 = env.scene[trocar_1_cfg.name]
    trocar_2 = env.scene[trocar_2_cfg.name]

    env_origins = env.scene.env_origins[env_ids]
    tray_state = tray.data.default_root_state[env_ids].clone()
    trocar_1_state = trocar_1.data.default_root_state[env_ids].clone()
    trocar_2_state = trocar_2.data.default_root_state[env_ids].clone()
    tray_state[:, :3] += env_origins
    trocar_1_state[:, :3] += env_origins
    trocar_2_state[:, :3] += env_origins

    random_yaw = torch.rand(len(env_ids), device=env.device)
    random_yaw = random_yaw * ((max_angle_deg - min_angle_deg) * math.pi / 180.0) + min_angle_deg * math.pi / 180.0
    half_angle = random_yaw / 2.0
    delta_quat = torch.zeros(len(env_ids), 4, device=env.device)
    delta_quat[:, 0] = torch.cos(half_angle)
    delta_quat[:, 3] = torch.sin(half_angle)

    tray_center = tray_state[:, :3]
    trocar_1_state[:, :3] = tray_center + _quat_rotate_vector(delta_quat, trocar_1_state[:, :3] - tray_center)
    trocar_2_state[:, :3] = tray_center + _quat_rotate_vector(delta_quat, trocar_2_state[:, :3] - tray_center)
    tray_state[:, 3:7] = _quat_multiply(delta_quat, tray_state[:, 3:7])
    trocar_1_state[:, 3:7] = _quat_multiply(delta_quat, trocar_1_state[:, 3:7])
    trocar_2_state[:, 3:7] = _quat_multiply(delta_quat, trocar_2_state[:, 3:7])

    zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)
    tray.write_root_pose_to_sim(tray_state[:, :7], env_ids=env_ids)
    trocar_1.write_root_pose_to_sim(trocar_1_state[:, :7], env_ids=env_ids)
    trocar_2.write_root_pose_to_sim(trocar_2_state[:, :7], env_ids=env_ids)
    tray.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    trocar_1.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    trocar_2.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


# ---------- Task-stage tracking + reward shaping ------------------------------


def get_task_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_task_stage"):
        env._task_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._task_stage


def update_task_stage(
    env: ManagerBasedRLEnv,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    table_height: float = 0.85483,
    lift_threshold: float = 0.15,
    tip_align_threshold: float = 0.015,
    insertion_dist_threshold: float = 0.05,
    insertion_angle_threshold: float = 0.15,
    placement_x_min: float = -1.8,
    placement_x_max: float = -1.4,
    placement_y_min: float = 1.5,
    placement_y_max: float = 1.8,
    placement_z_min: float = 0.9,
    print_log: bool = False,
) -> torch.Tensor:
    stage = get_task_stage(env)
    old_stage = stage.clone()
    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]
    pos1 = obj1.data.root_pos_w
    pos2 = obj2.data.root_pos_w
    quat1 = obj1.data.root_quat_w
    quat2 = obj2.data.root_quat_w

    lifted = (pos1[:, 2] > table_height + lift_threshold) & (pos2[:, 2] > table_height + lift_threshold)
    stage = torch.where((stage == 0) & lifted, torch.ones_like(stage), stage)

    tip_distance = torch.linalg.norm(
        _trocar_tip_position(env, asset_cfg1) - _trocar_tip_position(env, asset_cfg2), dim=-1
    )
    stage = torch.where((stage == 1) & (tip_distance < tip_align_threshold), torch.full_like(stage, 2), stage)

    target_axis = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    axis1 = quat_apply(quat1, target_axis)
    axis2 = quat_apply(quat2, target_axis)
    angle = torch.acos(torch.clamp(torch.abs(torch.sum(axis1 * axis2, dim=-1)), max=1.0))
    center_distance = torch.linalg.norm(pos1 - pos2, dim=-1)
    center_close = center_distance < insertion_dist_threshold
    stage = torch.where(
        (stage == 2) & center_close & (angle < insertion_angle_threshold),
        torch.full_like(stage, 3),
        stage,
    )

    origins = env.scene.env_origins
    x_min = origins[:, 0] + min(placement_x_min, placement_x_max)
    x_max = origins[:, 0] + max(placement_x_min, placement_x_max)
    y_min = origins[:, 1] + min(placement_y_min, placement_y_max)
    y_max = origins[:, 1] + max(placement_y_min, placement_y_max)
    in_zone_1 = (pos1[:, 0] >= x_min) & (pos1[:, 0] <= x_max) & (pos1[:, 1] >= y_min) & (pos1[:, 1] <= y_max)
    in_zone_2 = (pos2[:, 0] >= x_min) & (pos2[:, 0] <= x_max) & (pos2[:, 1] >= y_min) & (pos2[:, 1] <= y_max)
    # Placement and release are distinct phases: table contact alone is not
    # success until the dex hands return near their open/reset pose.
    still_assembled = center_close & (angle < insertion_angle_threshold)
    placed = in_zone_1 & in_zone_2 & (pos1[:, 2] < placement_z_min) & (pos2[:, 2] < placement_z_min)
    stage = torch.where((stage == 3) & placed & still_assembled, torch.full_like(stage, 4), stage)
    hands_open = _robot_hands_near_open(env)
    released = placed & still_assembled & hands_open
    stage = torch.where((old_stage >= 4) & released, torch.full_like(stage, 5), stage)

    env._task_stage = stage
    _log_stage_transitions(
        env,
        old_stage,
        stage,
        pos1,
        pos2,
        tip_distance,
        center_distance,
        angle,
        lift_z_min=table_height + lift_threshold,
        tip_align_threshold=tip_align_threshold,
        insertion_dist_threshold=insertion_dist_threshold,
        insertion_angle_threshold=insertion_angle_threshold,
        placement_x_min=placement_x_min,
        placement_x_max=placement_x_max,
        placement_y_min=placement_y_min,
        placement_y_max=placement_y_max,
        placement_z_min=placement_z_min,
    )
    if print_log and (stage != old_stage).any():
        for env_id in range(env.num_envs):
            if stage[env_id] != old_stage[env_id]:
                print(f"Env {env_id}: assemble-trocar stage {old_stage[env_id].item()} -> {stage[env_id].item()}")
    return stage


def lift_trocars_reward(
    env: ManagerBasedRLEnv,
    table_height: float = 0.85483,
    lift_threshold: float = 0.15,
    tip_align_threshold: float = 0.015,
    insertion_dist_threshold: float = 0.05,
    insertion_angle_threshold: float = 0.15,
    placement_x_min: float = -1.8,
    placement_x_max: float = -1.4,
    placement_y_min: float = 1.5,
    placement_y_max: float = 1.8,
    placement_z_min: float = 0.9,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    print_log: bool = False,
) -> torch.Tensor:
    stage = update_task_stage(
        env,
        asset_cfg1=asset_cfg1,
        asset_cfg2=asset_cfg2,
        table_height=table_height,
        lift_threshold=lift_threshold,
        tip_align_threshold=tip_align_threshold,
        insertion_dist_threshold=insertion_dist_threshold,
        insertion_angle_threshold=insertion_angle_threshold,
        placement_x_min=placement_x_min,
        placement_x_max=placement_x_max,
        placement_y_min=placement_y_min,
        placement_y_max=placement_y_max,
        placement_z_min=placement_z_min,
        print_log=print_log,
    )
    return _transition_reward(env, stage, "_prev_stage_lift", 0, 1)


def trocar_tip_alignment_reward(
    env: ManagerBasedRLEnv,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    print_log: bool = False,
) -> torch.Tensor:
    stage = update_task_stage(env, asset_cfg1=asset_cfg1, asset_cfg2=asset_cfg2, print_log=print_log)
    return _transition_reward(env, stage, "_prev_stage_tip", 1, 2)


def trocar_insertion_reward(
    env: ManagerBasedRLEnv,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    print_log: bool = False,
) -> torch.Tensor:
    stage = update_task_stage(env, asset_cfg1=asset_cfg1, asset_cfg2=asset_cfg2, print_log=print_log)
    return _transition_reward(env, stage, "_prev_stage_insert", 2, 3)


def trocar_placement_reward(
    env: ManagerBasedRLEnv,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    print_log: bool = False,
) -> torch.Tensor:
    stage = update_task_stage(env, asset_cfg1=asset_cfg1, asset_cfg2=asset_cfg2, print_log=print_log)
    return _transition_reward(env, stage, "_prev_stage_place", 3, 4)


# ---------- Termination predicates --------------------------------------------


def task_success_termination(env: ManagerBasedRLEnv, success_stage: int = 5, print_log: bool = False) -> torch.Tensor:
    complete = get_task_stage(env) >= success_stage
    _store_assemble_trocar_state(env, "success", complete)
    if print_log and complete.any():
        print(f"Assemble Trocar completed in {complete.sum().item()} environment(s)")
    return complete


def object_drop_termination(
    env: ManagerBasedRLEnv,
    drop_height_threshold: float = 0.5,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    print_log: bool = False,
) -> torch.Tensor:
    dropped = (env.scene[asset_cfg1.name].data.root_pos_w[:, 2] < drop_height_threshold) | (
        env.scene[asset_cfg2.name].data.root_pos_w[:, 2] < drop_height_threshold
    )
    _store_assemble_trocar_state(env, "object_drop", dropped, asset_cfg1=asset_cfg1, asset_cfg2=asset_cfg2)
    if print_log and dropped.any():
        print(f"Assemble Trocar drop termination triggered for {dropped.sum().item()} environment(s)")
    return dropped


# ---------- Private helpers ---------------------------------------------------


def _transition_reward(
    env: ManagerBasedRLEnv,
    stage: torch.Tensor,
    attr: str,
    from_stage: int,
    to_stage: int,
) -> torch.Tensor:
    if not hasattr(env, attr):
        setattr(env, attr, stage.clone())
    prev = getattr(env, attr)
    reward = torch.where(
        (prev == from_stage) & (stage >= to_stage),
        torch.ones(env.num_envs, device=env.device) / env.step_dt,
        torch.zeros(env.num_envs, device=env.device),
    )
    setattr(env, attr, stage.clone())
    return reward


def _robot_hands_near_open(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids = _joint_ids(robot.data.joint_names, ASSEMBLE_TROCAR_HAND_JOINT_NAMES, env.device)
    current = robot.data.joint_pos[:, joint_ids]
    target = robot.data.default_joint_pos[:, joint_ids]
    max_error = torch.max(torch.abs(current - target), dim=1).values
    return max_error < _ASSEMBLE_TROCAR_HAND_OPEN_TOLERANCE_RAD


def _store_assemble_trocar_state(
    env: ManagerBasedRLEnv,
    status: str,
    mask: torch.Tensor,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
) -> None:
    if not bool(mask.any().item()):
        return
    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]
    pos1 = obj1.data.root_pos_w.detach().clone()
    pos2 = obj2.data.root_pos_w.detach().clone()
    tip1 = _trocar_tip_position(env, asset_cfg1)
    tip2 = _trocar_tip_position(env, asset_cfg2)
    stage = get_task_stage(env).detach().clone()
    angle = _trocar_axis_angle(obj1.data.root_quat_w, obj2.data.root_quat_w)
    env._assemble_trocar_last = {
        "status": status,
        "mask": mask.detach().clone(),
        "stage": stage,
        "trocar_1": pos1,
        "trocar_2": pos2,
        "tip_distance": torch.linalg.norm(tip1 - tip2, dim=-1).detach().clone(),
        "center_distance": torch.linalg.norm(pos1 - pos2, dim=-1).detach().clone(),
        "axis_angle": angle.detach().clone(),
    }


def _log_stage_transitions(
    env: ManagerBasedRLEnv,
    old_stage: torch.Tensor,
    stage: torch.Tensor,
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    tip_distance: torch.Tensor,
    center_distance: torch.Tensor,
    angle: torch.Tensor,
    *,
    lift_z_min: float,
    tip_align_threshold: float,
    insertion_dist_threshold: float,
    insertion_angle_threshold: float,
    placement_x_min: float,
    placement_x_max: float,
    placement_y_min: float,
    placement_y_max: float,
    placement_z_min: float,
) -> None:
    changed = torch.nonzero(stage != old_stage, as_tuple=False).flatten()
    if changed.numel() == 0:
        return
    names = {
        1: "lifted both trocars",
        2: "aligned trocar tips",
        3: "inserted/assembled trocars",
        4: "placed assembled trocars",
        5: "released assembled trocars",
    }
    for env_id_tensor in changed.detach().cpu().tolist():
        env_id = int(env_id_tensor)
        new_stage = int(stage[env_id].item())
        logger.info(
            "Assemble Trocar milestone: step=%s env=%s stage=%s %s "
            "(trocar_1=%s trocar_2=%s lift_z_min_m=%.4f tip_distance_m=%.4f tip_threshold_m=%.4f "
            "center_distance_m=%.4f center_threshold_m=%.4f angle_rad=%.4f angle_threshold_rad=%.4f "
            "placement_bounds=x[%.3f,%.3f] y[%.3f,%.3f] z<%.3f)",
            int(getattr(env, "common_step_counter", 0)),
            env_id,
            new_stage,
            names.get(new_stage, "advanced"),
            _format_vec3(pos1[env_id]),
            _format_vec3(pos2[env_id]),
            lift_z_min,
            float(tip_distance[env_id].item()),
            tip_align_threshold,
            float(center_distance[env_id].item()),
            insertion_dist_threshold,
            float(angle[env_id].item()),
            insertion_angle_threshold,
            min(placement_x_min, placement_x_max),
            max(placement_x_min, placement_x_max),
            min(placement_y_min, placement_y_max),
            max(placement_y_min, placement_y_max),
            placement_z_min,
        )


def _format_vec3(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values.detach().cpu().flatten()[:3].tolist()) + "]"


def _trocar_tip_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Tip position of a trocar in world frame.

    Marker offsets come from the original Rheo MDP — the cannula (``trocar_1``)
    tip sits +6.4 cm along the local z, while the puncture device's tip
    (``trocar_2``) sits -9.6 cm along the local z.
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    if asset_cfg.name == "trocar_1":
        local_tip = torch.tensor([0.0, 0.0, 0.064], device=env.device)
    else:
        local_tip = torch.tensor([0.0, 0.0, -0.096387], device=env.device)
    return obj.data.root_pos_w + quat_apply(obj.data.root_quat_w, local_tip.repeat(env.num_envs, 1))


def _trocar_axis_angle(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    axis1 = quat_apply(quat1, torch.tensor([0.0, 0.0, 1.0], device=quat1.device).repeat(quat1.shape[0], 1))
    axis2 = quat_apply(quat2, torch.tensor([0.0, 0.0, 1.0], device=quat2.device).repeat(quat2.shape[0], 1))
    cosine = torch.clamp(torch.sum(axis1 * axis2, dim=-1), -1.0, 1.0)
    return torch.acos(cosine)


def _joint_ids(joint_names: Sequence[str], selected_names: Sequence[str], device: str) -> torch.Tensor:
    ids = [joint_names.index(name) for name in selected_names]
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[:, 1:]
    q_w = q[:, :1]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q_w * uv + uuv)


# ---------- Env-cfg recipe ----------------------------------------------------


@configclass
class SceneLightsCfg:
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    success: TerminationTermCfg = TerminationTermCfg(
        func=task_success_termination,
        time_out=False,
        params={"success_stage": 5, "print_log": False},
    )
    object_drop: TerminationTermCfg = TerminationTermCfg(
        func=object_drop_termination,
        time_out=True,
        params={
            "drop_height_threshold": 0.5,
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "print_log": False,
        },
    )


def _stage_params() -> dict:
    return {
        "table_height": 0.85483,
        "lift_threshold": 0.15,
        "tip_align_threshold": 0.015,
        "insertion_dist_threshold": 0.05,
        "insertion_angle_threshold": 0.15,
        "placement_x_min": -1.8,
        "placement_x_max": -1.4,
        "placement_y_min": 1.5,
        "placement_y_max": 1.8,
        "placement_z_min": 0.9,
        "asset_cfg1": SceneEntityCfg("trocar_1"),
        "asset_cfg2": SceneEntityCfg("trocar_2"),
        "print_log": False,
    }


@configclass
class RewardsCfg:
    lift_trocars: RewardTermCfg = RewardTermCfg(func=lift_trocars_reward, weight=1.0, params=_stage_params())
    tip_alignment: RewardTermCfg = RewardTermCfg(
        func=trocar_tip_alignment_reward,
        weight=1.0,
        params={"asset_cfg1": SceneEntityCfg("trocar_1"), "asset_cfg2": SceneEntityCfg("trocar_2"), "print_log": False},
    )
    insert_trocars: RewardTermCfg = RewardTermCfg(
        func=trocar_insertion_reward,
        weight=1.0,
        params={"asset_cfg1": SceneEntityCfg("trocar_1"), "asset_cfg2": SceneEntityCfg("trocar_2"), "print_log": False},
    )
    placement_trocars: RewardTermCfg = RewardTermCfg(
        func=trocar_placement_reward,
        weight=1.0,
        params={"asset_cfg1": SceneEntityCfg("trocar_1"), "asset_cfg2": SceneEntityCfg("trocar_2"), "print_log": False},
    )


@configclass
class EventsCfg:
    reset_scene: EventTermCfg = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_task_stage: EventTermCfg = EventTermCfg(func=reset_task_stage, mode="reset")
    reset_tray_random_rotation: EventTermCfg = EventTermCfg(
        func=reset_tray_with_random_rotation,
        mode="reset",
        params={
            "tray_cfg": SceneEntityCfg("tray"),
            "trocar_1_cfg": SceneEntityCfg("trocar_1"),
            "trocar_2_cfg": SceneEntityCfg("trocar_2"),
            "rotation_range": [0.0, 10.0],
        },
    )


class AssembleTrocarTask(TaskBase):
    """Arena task wrapper for the G1 dex-hand assemble-trocar objective."""

    def __init__(self, episode_length_s: float = 20.0):
        super().__init__(episode_length_s=episode_length_s, task_description="install trocar from box")

    def get_scene_cfg(self):
        return SceneLightsCfg()

    def get_termination_cfg(self):
        return TerminationsCfg()

    def get_events_cfg(self):
        return EventsCfg()

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_rewards_cfg(self):
        return RewardsCfg()

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(-0.5, 2.4, 1.6), lookat=(-5.4, 0.2, -1.2), cam_prim_path="/OmniverseKit_Persp")

    def modify_env_cfg(self, env_cfg):
        env_cfg.decimation = 4
        env_cfg.episode_length_s = 20.0
        env_cfg.scene.env_spacing = 6.0
        env_cfg.scene.replicate_physics = True
        env_cfg.sim.dt = 1 / 200
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.sim.physx.bounce_threshold_velocity = 0.01
        env_cfg.sim.render.enable_translucency = True
        if env_cfg.sim.render.carb_settings is None:
            env_cfg.sim.render.carb_settings = {}
        env_cfg.sim.render.carb_settings["rtx.raytracing.fractionalCutoutOpacity"] = True
        env_cfg.sim.render.rendering_mode = "quality"
        env_cfg.sim.render.antialiasing_mode = "DLAA"
        return env_cfg
