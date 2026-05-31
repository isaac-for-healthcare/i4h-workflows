# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import torch
from common.config import get_robot_config
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.tasks.task_base import TaskBase


def reset_xform_root_pose_uniform(env, env_ids, pose_range, velocity_range, asset_cfg: SceneEntityCfg):
    xform = env.scene[asset_cfg.name]
    positions, orientations = xform.get_world_poses()

    if hasattr(env_ids, "to"):
        idx_tensor = env_ids.to(dtype=torch.long, device=positions.device)
    else:
        idx_tensor = torch.tensor(env_ids, dtype=torch.long, device=positions.device)

    pos_sel = positions.index_select(0, idx_tensor).clone()
    ori_sel = orientations.index_select(0, idx_tensor).clone()
    offs = torch.zeros((idx_tensor.shape[0], 3), device=positions.device, dtype=positions.dtype)
    for axis, key in enumerate(("x", "y", "z")):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        if lo != 0.0 or hi != 0.0:
            offs[:, axis] = torch.empty(idx_tensor.shape[0], device=positions.device, dtype=positions.dtype).uniform_(
                lo, hi
            )

    base_pos = pos_sel[:, :3].clone()
    target_pos = base_pos + offs
    for axis, key in enumerate(("x", "y", "z")):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        if lo != 0.0 or hi != 0.0:
            target_pos[:, axis] = torch.minimum(
                torch.maximum(target_pos[:, axis], base_pos[:, axis] + lo),
                base_pos[:, axis] + hi,
            )

    pos_sel[:, :3] = target_pos
    xform.set_world_poses(pos_sel, ori_sel, indices=idx_tensor.detach().cpu().tolist())


_TRAY_SUCCESS_HALF_EXTENTS_XY = (0.08, 0.08)
_TRAY_SUCCESS_Z_RANGE = (-0.02, 0.09)
_SCISSORS_SETTLE_MOVE_M = 0.004
_SCISSORS_HOME_TOLERANCE_M = 0.03
_ROBOT_EE_BODY_NAME = "gripper"
_SO101_CONFIG = get_robot_config("so101")
_ROBOT_ARM_JOINT_COUNT = _SO101_CONFIG.arm_joint_count or _SO101_CONFIG.body_joint_count
_ROBOT_JOINT_NAMES = _SO101_CONFIG.joint_names


def success_scissors_in_tray(env) -> torch.Tensor:
    scissors_pos = env.scene["scissors"].data.root_pos_w
    return _points_inside_tray_bounds(env, scissors_pos)


def success_scissors_placed(
    env,
    settle_move_m: float = _SCISSORS_SETTLE_MOVE_M,
    home_tolerance_m: float = _SCISSORS_HOME_TOLERANCE_M,
    home_joint_pos_rad: Sequence[float] | None = None,
) -> torch.Tensor:
    """Success once scissors are settled in the tray and the arm is back home."""
    in_tray = success_scissors_in_tray(env)
    scissors_pos = env.scene["scissors"].data.root_pos_w.detach()
    if not hasattr(env, "_scissor_success_prev_pos"):
        env._scissor_success_prev_pos = scissors_pos.clone()
        env._scissor_success_settled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._scissor_success_home_ee_pos = _robot_ee_position(env).detach().clone()

    move = torch.linalg.norm(scissors_pos - env._scissor_success_prev_pos, dim=-1)
    settled = in_tray & (move <= settle_move_m)
    env._scissor_success_settled = torch.where(
        in_tray, env._scissor_success_settled | settled, torch.zeros_like(settled)
    )
    env._scissor_success_prev_pos = torch.where(
        in_tray.unsqueeze(-1), scissors_pos.clone(), env._scissor_success_prev_pos
    )
    arm_home = _robot_ee_near_home(env, home_tolerance_m)
    success = env._scissor_success_settled & arm_home
    _store_scissor_success_state(env, scissors_pos, in_tray, arm_home, home_tolerance_m, home_joint_pos_rad)
    return success


def reset_scissor_success_state(env, env_ids) -> None:
    idx = (
        env_ids.to(dtype=torch.long, device=env.device)
        if hasattr(env_ids, "to")
        else torch.tensor(env_ids, device=env.device)
    )
    if not hasattr(env, "_scissor_success_prev_pos"):
        env._scissor_success_prev_pos = env.scene["scissors"].data.root_pos_w.clone()
        env._scissor_success_settled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._scissor_success_home_ee_pos = _robot_ee_position(env).detach().clone()
    env._scissor_success_prev_pos[idx] = env.scene["scissors"].data.root_pos_w[idx]
    env._scissor_success_settled[idx] = False
    env._scissor_success_home_ee_pos[idx] = _robot_ee_position(env)[idx]


def _store_scissor_success_state(
    env,
    scissors_pos: torch.Tensor,
    in_tray: torch.Tensor,
    arm_home: torch.Tensor,
    home_tolerance_m: float,
    home_joint_pos_rad: Sequence[float] | None,
) -> None:
    current_arm, home_arm, error = _robot_arm_home_error(env, home_joint_pos_rad)
    if home_arm.ndim == 1:
        home_arm = home_arm.unsqueeze(0).expand_as(current_arm)
    ee_pos = _robot_ee_position(env).detach().clone()
    home_ee_pos = env._scissor_success_home_ee_pos.detach().clone()
    env._scissor_success_last = {
        "scissors_pos": scissors_pos.detach().clone(),
        "in_tray": in_tray.detach().clone(),
        "settled": env._scissor_success_settled.detach().clone(),
        "arm_home": arm_home.detach().clone(),
        "arm_current": current_arm.detach().clone(),
        "arm_target": home_arm.detach().clone(),
        "arm_error": error.detach().clone(),
        "ee_pos": ee_pos,
        "home_ee_pos": home_ee_pos,
        "ee_home_error_m": torch.linalg.norm(ee_pos - home_ee_pos, dim=-1),
        "home_tolerance_m": home_tolerance_m,
    }


def _robot_ee_near_home(env, tolerance_m: float) -> torch.Tensor:
    ee_pos = _robot_ee_position(env)
    home_ee_pos = env._scissor_success_home_ee_pos
    return torch.linalg.norm(ee_pos - home_ee_pos, dim=-1) <= tolerance_m


def _robot_ee_position(env) -> torch.Tensor:
    robot = env.scene["robot"]
    body_id = _robot_ee_body_index(robot)
    for attr in ("body_state_w", "body_link_state_w"):
        body_state = getattr(robot.data, attr, None)
        if body_state is not None:
            return body_state[:, body_id, :3]
    body_pos = getattr(robot.data, "body_pos_w", None)
    if body_pos is not None:
        return body_pos[:, body_id, :3] if body_pos.ndim == 3 else body_pos[:, :3]
    return robot.data.root_pos_w


def _robot_ee_body_index(robot) -> int:
    body_names = list(getattr(robot.data, "body_names", ()))
    if _ROBOT_EE_BODY_NAME not in body_names:
        raise RuntimeError(f"SO-ARM end-effector body '{_ROBOT_EE_BODY_NAME}' not found in robot bodies: {body_names}")
    return body_names.index(_ROBOT_EE_BODY_NAME)


def _robot_arm_home_error(
    env,
    home_joint_pos_rad: Sequence[float] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    joints = robot.data.joint_pos
    if joints.shape[-1] != len(_ROBOT_JOINT_NAMES):
        joint_ids = [_joint_index(robot, name, idx) for idx, name in enumerate(_ROBOT_JOINT_NAMES)]
        joints = joints[:, joint_ids]
    current = joints[:, :_ROBOT_ARM_JOINT_COUNT]
    if home_joint_pos_rad is None:
        home_joint_pos_rad = robot.data.default_joint_pos[0, :_ROBOT_ARM_JOINT_COUNT].detach().cpu().tolist()
    home = torch.tensor(home_joint_pos_rad[:_ROBOT_ARM_JOINT_COUNT], dtype=current.dtype, device=current.device)
    return current, home, torch.abs(current - home)


def _asset_world_position(asset) -> torch.Tensor:
    asset_data = getattr(asset, "data", None)
    if asset_data is not None and hasattr(asset_data, "root_pos_w"):
        return asset_data.root_pos_w
    positions, _ = asset.get_world_poses()
    return positions[:, :3]


def _points_inside_tray_bounds(
    env,
    points: torch.Tensor,
    env_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    tray_pos = _asset_world_position(env.scene["tray"])[_env_index(env_ids)]
    xy_delta = torch.abs(points[:, :2] - tray_pos[:, :2])
    half_extents = torch.tensor(_TRAY_SUCCESS_HALF_EXTENTS_XY, device=points.device, dtype=points.dtype)
    xy_inside = torch.all(xy_delta <= half_extents, dim=-1)
    z_delta = points[:, 2] - tray_pos[:, 2]
    z_inside = (_TRAY_SUCCESS_Z_RANGE[0] <= z_delta) & (z_delta <= _TRAY_SUCCESS_Z_RANGE[1])
    return xy_inside & z_inside


def _env_index(env_ids: Sequence[int] | None):
    if env_ids is None:
        return slice(None)
    return list(env_ids)


def _joint_index(robot, joint_name: str, fallback: int) -> int:
    if hasattr(robot, "find_joints"):
        found = robot.find_joints(joint_name, preserve_order=True)
        for value in (found[0], found[1]):
            if hasattr(value, "numel") and value.numel() > 0:
                return int(value[0])
            if len(value) > 0 and isinstance(value[0], int):
                return int(value[0])
    return fallback


@configclass
class ScissorPickAndPlaceEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_scissors = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.010),
                "y": (-0.005, 0.010),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("scissors"),
        },
    )
    reset_tray = EventTerm(
        func=reset_xform_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.000, 0.000)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray"),
        },
    )
    reset_success_state = EventTerm(func=reset_scissor_success_state, mode="reset")


def _termination_cfg(home_joint_pos_rad: Sequence[float] | None):
    @configclass
    class ScissorPickAndPlaceTerminationsCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        success = DoneTerm(
            func=success_scissors_placed,
            params={"home_joint_pos_rad": tuple(home_joint_pos_rad) if home_joint_pos_rad is not None else None},
            time_out=False,
        )

    return ScissorPickAndPlaceTerminationsCfg()


class ScissorPickAndPlaceTask(TaskBase):
    """Arena task wrapper for the SO-ARM scissor pick-and-place setup."""

    def __init__(
        self,
        episode_length_s: float = 8.0,
        env_spacing: float = 4.0,
        home_joint_pos_rad: Sequence[float] | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Pick up the surgical scissors and place them in the tray.",
        )
        self.env_spacing = env_spacing
        self.home_joint_pos_rad = home_joint_pos_rad

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return _termination_cfg(self.home_joint_pos_rad)

    def get_events_cfg(self):
        return ScissorPickAndPlaceEventsCfg()

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self):
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(2.0, 2.0, 1.5), lookat=(0.0, 0.0, 0.2))

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg.scene.env_spacing = self.env_spacing
        return env_cfg
