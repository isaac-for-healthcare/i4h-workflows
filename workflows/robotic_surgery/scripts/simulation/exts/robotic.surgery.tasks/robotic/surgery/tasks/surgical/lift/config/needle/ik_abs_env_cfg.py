# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from robotic.surgery.tasks.surgical.lift import mdp
from simulation.utils.assets import robotic_surgery_assets

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from robotic.surgery.assets.psm import PSM_HIGH_PD_CFG  # isort: skip


@configclass
class NeedleLiftEnvCfg(joint_pos_env_cfg.NeedleLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (PSM)
        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )


@configclass
class NeedleLiftEnvCfg_PLAY(NeedleLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


##
# Operating Room (OR) environment.
##


@configclass
class NeedleLiftOREnvCfg(NeedleLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # simulation settings
        self.viewer.eye = (-0.32, 0.12, 0.12)

        # lights
        self.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, 0.0, 0.1), rot=(0.7071068, 0.0, -0.7071068, 0.0)),
            spawn=sim_utils.DiskLightCfg(radius=0.2, color=(0.75, 0.75, 0.75), intensity=3000.0),
        )

        # Set table to None
        self.scene.table = None

        # Operating Room (OR)
        self.scene.organs = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Organs",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, -0.14, -0.85), rot=(0.7071068, 0.0, 0.0, 0.7071068)),
            spawn=UsdFileCfg(usd_path=robotic_surgery_assets.Organs, scale=(0.01, 0.01, 0.01)),
        )

        # Set Suture Needle as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=robotic_surgery_assets.Needle,
                scale=(0.4, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.03, 0.02), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )


@configclass
class NeedleLiftOREnvCfg_PLAY(NeedleLiftOREnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # Set the camera for table view
        self.scene.static_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/WorldStaticCamera",
            update_period=0.02,
            height=1080,
            width=1920,
            data_types=["rgb"],
            spawn=sim_utils.FisheyeCameraCfg(
                projection_type="fisheyePolynomial",
                fisheye_max_fov=150.0,
            ),
            offset=CameraCfg.OffsetCfg(pos=(-0.03, 0.0, 0.06), rot=(0.25, 0.93301, 0.0669, 0.25), convention="ros"),
        )
