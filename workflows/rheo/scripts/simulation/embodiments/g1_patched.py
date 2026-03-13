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

"""Workflow-specific overrides for the G1 embodiment."""

from collections.abc import Sequence

import isaaclab.envs.mdp as base_mdp
import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_arena.embodiments.g1.g1 import *  # noqa: F401,F403
from isaaclab_arena.embodiments.g1.g1 import G1CameraCfg, G1EmbodimentBase, G1MimicEnv
from simulation.assets.assets import UNITREE_G1_29DOF_USD

_orig_get_scene_cfg = G1EmbodimentBase.get_scene_cfg


def _get_scene_cfg(self):
    scene_cfg = _orig_get_scene_cfg(self)
    if getattr(scene_cfg, "robot", None) is not None:
        scene_cfg.robot = scene_cfg.robot.copy()
        scene_cfg.robot.spawn.usd_path = UNITREE_G1_29DOF_USD
        scene_cfg.robot.spawn.semantic_tags = [("class", "robot")]

    return scene_cfg


G1EmbodimentBase.get_scene_cfg = _get_scene_cfg

# Patch camera config to add semantic segmentation support
_original_camera_post_init = G1CameraCfg.__post_init__


def _patched_camera_post_init(self):
    """Patched __post_init__ to add semantic_segmentation to data_types."""
    # Call original to set up basic config
    _original_camera_post_init(self)

    # Modify the camera config to include semantic segmentation
    if hasattr(self, "robot_head_cam") and self.robot_head_cam is not None:
        # Add semantic_segmentation to data_types
        current_data_types = list(self.robot_head_cam.data_types)
        if "semantic_segmentation" not in current_data_types:
            current_data_types.append("semantic_segmentation")
        if "distance_to_image_plane" not in current_data_types:
            current_data_types.append("distance_to_image_plane")

        mapping = {
            "class:robot": (255, 0, 0, 255),
            "class:cart": (0, 255, 0, 255),
            "class:box": (0, 0, 255, 255),
            "class:UNLABELLED": (0, 0, 0, 255),
        }
        self.robot_head_cam.semantic_segmentation_mapping = mapping
        self.robot_head_cam.data_types = current_data_types
        # Enable colorization for better visualization
        self.robot_head_cam.colorize_semantic_segmentation = True


G1CameraCfg.__post_init__ = _patched_camera_post_init

# Patch get_observation_cfg to dynamically add camera observations to policy group
_original_get_observation_cfg = G1EmbodimentBase.get_observation_cfg


def _patched_get_observation_cfg(self):
    """Patched get_observation_cfg to add camera observations to policy group."""
    obs_cfg = _original_get_observation_cfg(self)

    # If cameras are enabled, add camera observations to policy group
    if self.enable_cameras and hasattr(obs_cfg, "policy"):
        # Add RGB observation
        if not hasattr(obs_cfg.policy, "robot_head_cam"):
            obs_cfg.policy.robot_head_cam = ObsTerm(
                func=base_mdp.image,
                params={"sensor_cfg": SceneEntityCfg("robot_head_cam"), "data_type": "rgb", "normalize": False},
            )

        # Add semantic segmentation observation
        if not hasattr(obs_cfg.policy, "robot_head_cam_seg"):
            obs_cfg.policy.robot_head_cam_seg = ObsTerm(
                func=base_mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("robot_head_cam"),
                    "data_type": "semantic_segmentation",
                    "normalize": False,
                },
            )

    return obs_cfg


G1EmbodimentBase.get_observation_cfg = _patched_get_observation_cfg


def _get_object_poses(self, env_ids: Sequence[int] | None = None):
    """Extended pose extraction that handles both rigid objects and articulations."""
    if env_ids is None:
        env_ids = slice(None)

    pelvis_pose_w = self.scene["robot"].data.body_link_state_w[
        :, self.scene["robot"].data.body_names.index("pelvis"), :
    ]
    pelvis_position_w = pelvis_pose_w[:, :3] - self.scene.env_origins
    pelvis_rot_mat_w = PoseUtils.matrix_from_quat(pelvis_pose_w[:, 3:7])
    pelvis_pose_mat_w = PoseUtils.make_pose(pelvis_position_w, pelvis_rot_mat_w)
    pelvis_pose_inv = PoseUtils.pose_inv(pelvis_pose_mat_w)

    state = self.scene.get_state(is_relative=True)
    object_pose_matrix = {}

    for group_name in ["rigid_object", "articulation"]:
        if group_name not in state:
            continue
        for obj_name, obj_state in state[group_name].items():
            object_pose_mat_w = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3],
                PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
            )
            object_pose_pelvis_frame = torch.matmul(pelvis_pose_inv, object_pose_mat_w)
            object_pose_matrix[obj_name] = object_pose_pelvis_frame

    return object_pose_matrix


G1MimicEnv.get_object_poses = _get_object_poses
