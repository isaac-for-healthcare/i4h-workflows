# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import numpy as np
import torch
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask
from isaaclab_arena.tasks.terminations import objects_in_proximity
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events


class TrayPickPlaceTask(G1LocomanipPickAndPlaceTask):
    """Rheo tray pick-and-place task in the pre-operative scene."""

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([1.0, -1.0, 1.5]),
        )

    def get_mimic_env_cfg(self, embodiment_name: str):
        env_cfg = super().get_mimic_env_cfg(embodiment_name)
        env_cfg.datagen_config.name = "g1_tray_pick_and_place_task_D0"
        env_cfg.subtask_configs = {
            "right": [
                SubTaskConfig(
                    object_ref=self.pick_up_object.name,
                    subtask_term_signal="right_before_grasp_box",
                    first_subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.001,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
                SubTaskConfig(
                    object_ref=self.pick_up_object.name,
                    subtask_term_signal="right_after_release_box",
                    first_subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.001,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
                _final_subtask(self.pick_up_object.name),
            ],
            "left": [
                SubTaskConfig(
                    object_ref=self.pick_up_object.name,
                    subtask_term_signal="left_before_grasp_box",
                    first_subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.001,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
                SubTaskConfig(
                    object_ref=self.pick_up_object.name,
                    subtask_term_signal="left_after_release_box",
                    first_subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.001,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
                _final_subtask(self.pick_up_object.name),
            ],
            "body": [
                SubTaskConfig(
                    object_ref=self.pick_up_object.name,
                    subtask_term_signal="body_face_box_front",
                    first_subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.001,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
                SubTaskConfig(
                    object_ref=self.pick_up_object.name,
                    subtask_term_signal="body_face_cart_front",
                    first_subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.001,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
                _final_subtask(self.pick_up_object.name),
            ],
        }
        return env_cfg

    def get_events_cfg(self):
        return I4HEventsCfg(pick_up_object=self.pick_up_object, destination_cart=self.destination_bin)

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=tray_placed_success,
            params={
                "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                "target_object_cfg": SceneEntityCfg(self.destination_bin.name),
                "max_x_separation": 0.16,
                "max_y_separation": 0.16,
                "max_z_separation": 0.88,
                "settle_steps": 20,
                "max_linear_speed": 0.08,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TerminationsCfg(success=success, object_dropped=object_dropped)

    def modify_env_cfg(self, env_cfg):
        if hasattr(super(), "modify_env_cfg"):
            env_cfg = super().modify_env_cfg(env_cfg)
        env_cfg.sim.render.rendering_mode = "quality"
        env_cfg.sim.render.antialiasing_mode = "DLAA"
        return env_cfg


def tray_placed_success(
    env,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_x_separation: float = 0.16,
    max_y_separation: float = 0.16,
    max_z_separation: float = 0.88,
    settle_steps: int = 20,
    max_linear_speed: float = 0.08,
) -> torch.Tensor:
    """Success only after the tray remains near the cart and stops moving."""
    placed = objects_in_proximity(
        env,
        object_cfg=object_cfg,
        target_object_cfg=target_object_cfg,
        max_x_separation=max_x_separation,
        max_y_separation=max_y_separation,
        max_z_separation=max_z_separation,
    )
    speed = _root_linear_speed(env, object_cfg)
    stable = speed <= max_linear_speed
    ready = placed & stable
    if not hasattr(env, "_tray_success_settle_count"):
        env._tray_success_settle_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._tray_success_settle_count = torch.where(
        ready,
        env._tray_success_settle_count + 1,
        torch.zeros_like(env._tray_success_settle_count),
    )
    return env._tray_success_settle_count >= settle_steps


def _root_linear_speed(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    velocity = getattr(data, "root_lin_vel_w", None)
    if velocity is None:
        root_vel = getattr(data, "root_vel_w", None)
        if root_vel is not None:
            velocity = root_vel[:, :3]
    if velocity is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.linalg.norm(velocity, dim=-1)


def reset_tray_success_state(env, env_ids: torch.Tensor) -> None:
    if hasattr(env, "_tray_success_settle_count"):
        env._tray_success_settle_count[env_ids] = 0


def _final_subtask(object_ref: str) -> SubTaskConfig:
    return SubTaskConfig(
        object_ref=object_ref,
        first_subtask_start_offset_range=(0, 0),
        subtask_term_offset_range=(0, 0),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=0.0,
        num_interpolation_steps=0,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class I4HEventsCfg:
    reset_success_state: EventTermCfg = MISSING
    reset_destination_cart_pose: EventTermCfg = MISSING
    reset_pick_up_object_pose: EventTermCfg = MISSING

    def __init__(self, pick_up_object: Asset, destination_cart: Asset):
        self.reset_success_state = EventTermCfg(func=reset_tray_success_state, mode="reset")
        object_initial_pose = pick_up_object.get_initial_pose()
        object_roll, object_pitch, object_yaw = euler_xyz_from_quat(
            torch.tensor(object_initial_pose.rotation_wxyz).reshape(1, 4)
        )
        self.reset_pick_up_object_pose = EventTermCfg(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": {
                    "x": (object_initial_pose.position_xyz[0] - 0.025, object_initial_pose.position_xyz[0] + 0.025),
                    "y": (object_initial_pose.position_xyz[1] - 0.025, object_initial_pose.position_xyz[1] + 0.025),
                    "z": (object_initial_pose.position_xyz[2], object_initial_pose.position_xyz[2]),
                    "roll": (object_roll, object_roll),
                    "pitch": (object_pitch, object_pitch),
                    "yaw": (object_yaw, object_yaw),
                },
                "asset_cfgs": [SceneEntityCfg(pick_up_object.name)],
            },
        )

        cart_initial_pose = destination_cart.get_initial_pose()
        cart_roll, cart_pitch, cart_yaw = euler_xyz_from_quat(
            torch.tensor(cart_initial_pose.rotation_wxyz).reshape(1, 4)
        )
        self.reset_destination_cart_pose = EventTermCfg(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": {
                    "x": (cart_initial_pose.position_xyz[0] - 0.01, cart_initial_pose.position_xyz[0] + 0.01),
                    "y": (cart_initial_pose.position_xyz[1] - 0.01, cart_initial_pose.position_xyz[1] + 0.01),
                    "z": (cart_initial_pose.position_xyz[2], cart_initial_pose.position_xyz[2]),
                    "roll": (cart_roll, cart_roll),
                    "pitch": (cart_pitch, cart_pitch),
                    "yaw": (cart_yaw, cart_yaw),
                },
                "asset_cfgs": [SceneEntityCfg(destination_cart.name)],
            },
        )
