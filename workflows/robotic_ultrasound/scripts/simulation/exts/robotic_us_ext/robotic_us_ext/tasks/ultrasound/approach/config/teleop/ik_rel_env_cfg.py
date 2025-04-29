# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
from robotic_us_ext.lab_assets.franka import FRANKA_PANDA_HIGH_PD_FORCE_CFG, FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG

from ..franka import franka_manager_rl_env_cfg


@configclass
class ModFrankaUltrasoundTeleopEnv(franka_manager_rl_env_cfg.RoboticIkRlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="TCP",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0],
                rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, 0.0]), degrees=True),
            ),
        )
        #
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/TCP",
                    name="end_effector",
                    # Uncomment and configure the offset if needed:
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, 0.0]), degrees=True),
                    ),
                ),
            ],
        )
        mapping = {
            "class:table": (0, 255, 0, 255),
            "class:organ": (0, 0, 255, 255),
            "class:robot": (255, 255, 0, 255),
            "class:ground": (255, 0, 0, 255),
            "class:UNLABELLED": (0, 0, 0, 255),
        }
        self.scene.wrist_camera = CameraCfg(
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            prim_path="{ENV_REGEX_NS}/Robot/D405_rigid/D405/Camera_OmniVision_OV9782_Color",
            spawn=None,
            height=224,
            width=224,
            update_period=0.0,
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=mapping
        )
        self.scene.room_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/third_person_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=100.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.55942, 0.56039, 0.36243),
                rot=euler_angles_to_quats(torch.tensor([248.0, 0.0, 180.0]), degrees=True),
                convention="ros",
            ),
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=mapping
        )


# Isaac-Teleop-Torso-Franka-IK-RL-Abs-v0
@configclass
class FrankaUltrasoundTeleopEnv(franka_manager_rl_env_cfg.RoboticIkRlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_FORCE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    # Uncomment and configure the offset if needed:
                    # offset=OffsetCfg(
                    #     pos=[0.0, 0.0, 0.1034],
                    # ),
                )
            ],
        )


@configclass
class FrankaUltrasoundTeleopEnv_PLAY(FrankaUltrasoundTeleopEnv):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
