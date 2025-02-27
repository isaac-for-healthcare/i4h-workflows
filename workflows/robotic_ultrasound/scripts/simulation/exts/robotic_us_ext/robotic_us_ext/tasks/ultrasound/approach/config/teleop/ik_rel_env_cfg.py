# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats
import torch


from ..franka import franka_manager_rl_env_cfg


##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

#  FRANKA_PANDA_REALSENSE_CFG for camera in USD
from robotic_us_ext.lab_assets.franka import (
    FRANKA_PANDA_HIGH_PD_FORCE_CFG,
    FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG
)


@configclass
class ModFrankaUltrasoundTeleopEnv(franka_manager_rl_env_cfg.RoboticIkRlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="TCP",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0], rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, 0.0]), degrees=True),),
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
                        pos=[0.0, 0.0, 0.0], rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, 0.0]), degrees=True),  
                )),
            ],
        )


# Isaac-Teleop-Torso-Franka-IK-RL-Abs-v0
@configclass
class FrankaUltrasoundTeleopEnv(franka_manager_rl_env_cfg.RoboticIkRlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_FORCE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
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
