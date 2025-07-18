# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from robotic.surgery.assets.star import STAR_HIGH_PD_CFG  # isort: skip


@configclass
class STARReachEnvCfg(joint_pos_env_cfg.STARReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set STAR as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = STAR_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (STAR)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["star_joint_.*"],
            body_name="endo360_needle",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )


@configclass
class STARReachEnvCfg_PLAY(STARReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
