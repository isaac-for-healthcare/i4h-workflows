# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import numpy as np
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events


def _object_at_destination(
    env: ManagerBasedRLEnv,
    cart_cfg: SceneEntityCfg,
    target_position_x: float,
    target_position_y: float,
    target_position_z: float,
    max_x_separation: float,
    max_y_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Termination: cart center within ``max_*_separation`` of the goal pose."""
    cart: RigidObject = env.scene[cart_cfg.name]
    cart_pos = cart.data.root_pos_w - env.scene.env_origins
    target_pos = torch.tensor([target_position_x, target_position_y, target_position_z], device=env.device).unsqueeze(0)
    done = torch.abs(cart_pos[:, 0] - target_pos[0, 0]) < max_x_separation
    done = torch.logical_and(done, torch.abs(cart_pos[:, 1] - target_pos[0, 1]) < max_y_separation)
    done = torch.logical_and(done, torch.abs(cart_pos[:, 2] - target_pos[0, 2]) < max_z_separation)
    return done


class PushCartTask(G1LocomanipPickAndPlaceTask):
    """Rheo cart-push task in the pre-operative scene."""

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([1.0, 0.5, 1.0]),
        )

    def get_mimic_env_cfg(self, embodiment_name: str):
        env_cfg = super().get_mimic_env_cfg(embodiment_name)
        env_cfg.datagen_config.name = "g1_push_cart_task_D0"
        env_cfg.subtask_configs = {
            "right": [
                _subtask("cart", "right_before_grasp_cart", 0.001),
                _subtask("cart", None, 0.0),
            ],
            "left": [
                _subtask("cart", "left_before_grasp_cart", 0.001),
                _subtask("cart", None, 0.0),
            ],
            "body": [
                _subtask("cart", "body_face_cart", 0.001),
                _subtask("cart", "body_pushing_cart", 0.001),
                _subtask("cart", None, 0.0),
            ],
        }
        return env_cfg

    def get_events_cfg(self):
        return I4HEventsCfg(pick_up_object=self.pick_up_object, destination_cart=self.destination_bin)

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=_object_at_destination,
            params={
                "cart_cfg": SceneEntityCfg(self.destination_bin.name),
                "target_position_x": 0.35,
                "target_position_y": -3.30,
                "target_position_z": -0.7875,
                "max_x_separation": 0.50,
                "max_y_separation": 0.30,
                "max_z_separation": 0.10,
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


def _subtask(object_ref: str, signal: str | None, action_noise: float) -> SubTaskConfig:
    kwargs = {"subtask_term_signal": signal} if signal is not None else {}
    return SubTaskConfig(
        object_ref=object_ref,
        first_subtask_start_offset_range=(0, 0),
        subtask_term_offset_range=(0, 0),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=action_noise,
        num_interpolation_steps=0,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
        **kwargs,
    )


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class I4HEventsCfg:
    reset_destination_cart_pose: EventTermCfg = MISSING
    reset_pick_up_object_pose: EventTermCfg = MISSING

    def __init__(self, pick_up_object: Asset, destination_cart: Asset):
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
                    "yaw": (object_yaw - 0.3, object_yaw + 0.3),
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
                    "x": (cart_initial_pose.position_xyz[0] - 0.01, cart_initial_pose.position_xyz[0] + 0.03),
                    "y": (cart_initial_pose.position_xyz[1] + 0.01, cart_initial_pose.position_xyz[1] + 0.04),
                    "z": (cart_initial_pose.position_xyz[2], cart_initial_pose.position_xyz[2]),
                    "roll": (cart_roll, cart_roll),
                    "pitch": (cart_pitch, cart_pitch),
                    "yaw": (cart_yaw, cart_yaw),
                },
                "asset_cfgs": [SceneEntityCfg(destination_cart.name)],
            },
        )
