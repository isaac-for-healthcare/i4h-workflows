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

import tempfile
import torch
from dataclasses import MISSING

from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.devices.openxr.retargeters.humanoid.fourier.g1_retargeter import G1UpperBodyRetargeterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip
from isaaclab_assets.robots.unitree import G1_CFG
from robotic_us_ext.tasks.ultrasound.pick import mdp
from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.00415], rot=[1.0, 0.0, 0.0, -180.0]),
        spawn=UsdFileCfg(
            usd_path=robot_us_assets.table_with_cover,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
            ),
        ),
    )

    # Object
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, -0.1, 0.84], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yunliu/Downloads/c3hd3_fixed_v2.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            scale=(0.01, 0.01, 0.01),  # Example scale
        ),
    )
    
    # Phantom
    organs = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.93], rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, 90.0]), degrees=True)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_us_assets.phantom,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            semantic_tags=[("class", "organ")],
        ),
    )

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = G1_CFG.replace(
        spawn=G1_CFG.spawn.replace(usd_path="/home/yunliu/Downloads/g1_29dof_with_hand_rev_1_0.usd"),
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, -0.6, 0.802),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": -0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": -0.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # 躯干和腿部
                "waist_pitch_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                ".*_hip_pitch_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": 0,
                ".*_ankle_roll_joint": 0.0,
                ".*_hand_.*": 0.0,
                # "head_joint": 0.0,
                # "imu_in_pelvis_joint": 0.0,
                # "imu_in_torso_joint": 0.0,
                # "pelvis_contour_joint": 0.0,
                # "d435_joint": 0.0,
                # "mid360_joint": 0.0,
                # "logo_joint": 0.0,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint", 
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                },
                armature={
                    ".*_hip_.*": 0.01,
                    ".*_knee_joint": 0.01,
                },
            ),
            "feet": ImplicitActuatorCfg(
                effort_limit=20,
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=20.0,
                damping=2.0,
                armature=0.02,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_hand_.*_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
                armature={
                    ".*_shoulder_.*": 0.01,
                    ".*_elbow_.*": 0.01,
                    ".*_hand_.*": 0.001,
                },
            ),
        },
    )


    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    gr1_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        # object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        # robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        # left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        # left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        # right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        # right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        # hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        # head_joint_state = ObsTerm(func=mdp.get_head_state)

        # object = ObsTerm(func=mdp.object_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.05, 0.0],
    #             "y": [0.0, 0.05],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )


@configclass
class PickReachG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (7), right hand joint pos (7)]
    idle_action = torch.tensor([
        -0.22878,
        0.2536,
        1.0953,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.22878,
        0.2536,
        1.0953,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    def __post_init__(self):
        """Post initialization."""
        
        # self.actions.g1_action = mdp.JointPositionActionCfg(
        #     asset_name="robot", joint_names=joint_names, scale=1.0, use_default_offset=False
        # )
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 60  # 100Hz
        self.sim.render_interval = 2
        self.viewer.eye = (0.0, 1.8, 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        self.actions.gr1_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
            ],
            ik_urdf_fixed_joint_names=[
                "left_hand_index_0_joint",
                "left_hand_middle_0_joint",
                "left_hand_thumb_0_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_1_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "right_hand_index_0_joint",
                "right_hand_middle_0_joint",
                "right_hand_thumb_0_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_1_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
                "left_hip_roll_joint",
                "right_hip_roll_joint",
                "left_hip_yaw_joint",
                "right_hip_yaw_joint",
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_knee_joint",
                "right_knee_joint",
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
                "head_joint",
                "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            hand_joint_names=[
                "left_hand_index_0_joint",
                "left_hand_middle_0_joint",
                "left_hand_thumb_0_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_1_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "right_hand_index_0_joint",
                "right_hand_middle_0_joint",
                "right_hand_thumb_0_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_1_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
            ],
            # the robot in the sim scene we are controlling
            asset_name="robot",
            # Configuration for the IK controller
            # The frames names are the ones present in the URDF file
            # The urdf has to be generated from the USD that is being used in the scene
            controller=PinkIKControllerCfg(
                articulation_name="robot",
                base_link_name="pelvis",
                num_hand_joints=14,
                show_ik_warnings=False,
                variable_input_tasks=[
                    FrameTask(
                        "left_hand_palm_link",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=1.0,  # [cost] / [rad]
                        lm_damping=10,  # dampening for solver for step jumps
                        gain=0.1,
                    ),
                    FrameTask(
                        "right_hand_palm_link",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=1.0,  # [cost] / [rad]
                        lm_damping=10,  # dampening for solver for step jumps
                        gain=0.1,
                    ),
                ],
                fixed_input_tasks=[
                    FrameTask(
                        "head_joint",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=0.05,  # [cost] / [rad]
                    ),
                    FrameTask(
                        "waist_yaw_joint",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=0.05,  # [cost] / [rad]
                    ),
                    FrameTask(
                        "waist_pitch_joint",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=0.05,  # [cost] / [rad]
                    ),
                    FrameTask(
                        "waist_roll_joint",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=0.05,  # [cost] / [rad]
                    ),
                ],
            ),
        )

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        print(f"temp_urdf_output_path: {temp_urdf_output_path}")
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.actions.gr1_action.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.gr1_action.controller.urdf_path = temp_urdf_output_path
        self.actions.gr1_action.controller.mesh_path = temp_urdf_meshes_output_path

        self.scene.robot_pov_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/RobotPOVCam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, -0.57, 0.4), rot=(0.4226, -0.9063, 0.0, 0.0), convention="ros"),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1UpperBodyRetargeterCfg(
                            enable_visualization=True,
                            # OpenXR hand tracking has 26 joints per hand
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.gr1_action.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
