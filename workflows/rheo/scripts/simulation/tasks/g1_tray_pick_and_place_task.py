# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow-specific locomanip pick-and-place task for the Unitree G1."""

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


class G1TrayPickPlaceTask(G1LocomanipPickAndPlaceTask):
    """Specialized pick-and-place task that includes pushing a cart in Pre-Operative scene."""

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([1.0, -1.0, 1.5]),
        )

    def get_mimic_env_cfg(self, embodiment_name: str):
        """Override parent to return Workflow-specific mimic environment configuration."""
        env_cfg = super().get_mimic_env_cfg(embodiment_name)

        # Override datagen config for the task
        env_cfg.datagen_config.name = "g1_tray_pick_and_place_task_D0"

        # Clear parent's subtask configs and define Workflow-specific ones
        env_cfg.subtask_configs = {}

        # Right hand subtasks: before grasp, after release, push cart
        subtask_configs = []
        subtask_configs.append(
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
            )
        )
        subtask_configs.append(
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
            )
        )
        # Last subtask: pushing cart (no signal needed)
        subtask_configs.append(
            SubTaskConfig(
                object_ref=self.pick_up_object.name,
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        env_cfg.subtask_configs["right"] = subtask_configs

        # Left hand subtasks
        subtask_configs = []
        subtask_configs.append(
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
            )
        )
        subtask_configs.append(
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
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=self.pick_up_object.name,
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        env_cfg.subtask_configs["left"] = subtask_configs

        # Body subtasks: face shelf, face cart, push to destination
        subtask_configs = []
        subtask_configs.append(
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
            )
        )
        subtask_configs.append(
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
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=self.pick_up_object.name,
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        env_cfg.subtask_configs["body"] = subtask_configs

        return env_cfg

    def get_events_cfg(self):
        """Override parent to add cart reset event."""
        events_cfg = I4HEventsCfg(
            pick_up_object=self.pick_up_object,
            destination_cart=self.destination_bin,
        )
        return events_cfg

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=objects_in_proximity,
            params={
                "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                "target_object_cfg": SceneEntityCfg(self.destination_bin.name),
                "max_x_separation": 0.16,
                "max_y_separation": 0.16,
                "max_z_separation": 0.88,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def modify_env_cfg(self, env_cfg):
        """Modify environment configuration for better rendering quality."""

        if hasattr(super(), "modify_env_cfg"):
            env_cfg = super().modify_env_cfg(env_cfg)

        # Set rendering configuration for better visibility
        env_cfg.sim.render.rendering_mode = "quality"
        env_cfg.sim.render.antialiasing_mode = "DLAA"

        return env_cfg


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING


@configclass
class I4HEventsCfg:
    """Event configuration for the task - adds box and cart reset events."""

    reset_destination_cart_pose: EventTermCfg = MISSING
    reset_pick_up_object_pose: EventTermCfg = MISSING

    def __init__(self, pick_up_object: Asset, destination_cart: Asset):
        """Initialize reset events for both pick_up_object and destination_cart.

        Args:
            pick_up_object: The object to pick up (e.g., surgical_tray)
            destination_cart: The cart to push (cart)
        """

        object_initial_pose = pick_up_object.get_initial_pose()
        object_roll, object_pitch, object_yaw = euler_xyz_from_quat(
            torch.tensor(object_initial_pose.rotation_wxyz).reshape(1, 4)
        )
        if object_initial_pose is not None:
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
        else:
            print(f"Pick up object {pick_up_object.name} has no initial pose.")
            self.reset_pick_up_object_pose = None

        cart_initial_pose = destination_cart.get_initial_pose()
        if cart_initial_pose is not None:
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
        else:
            print(f"Destination cart {destination_cart.name} has no initial pose.")
            self.reset_destination_cart_pose = None
