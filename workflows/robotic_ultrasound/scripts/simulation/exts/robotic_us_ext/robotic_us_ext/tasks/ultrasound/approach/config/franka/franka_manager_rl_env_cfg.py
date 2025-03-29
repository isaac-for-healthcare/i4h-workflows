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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from robotic_us_ext.lab_assets.franka import FRANKA_PANDA_HIGH_PD_FORCE_CFG, FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG
from robotic_us_ext.tasks.ultrasound.approach import mdp
from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
FRAME_MARKER_TINY_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_TINY_CFG.markers["frame"].scale = (0.01, 0.01, 0.01)


@configclass
class RoboticSoftCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.84]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.4804, 0.02017, -0.83415], rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, -90.0]), degrees=True)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_us_assets.table_with_cover,
        ),
    )

    # body
    # spawn the organ model onto the table, it needs to be scaled (1/10 of an inch?)
    # the model with _rigid was modified in USDComposer to have rigid body properties.
    # Leaving the props empty will use the default values.
    organs = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.6, 0.0, 0.09], rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, 180.0]), degrees=True)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_us_assets.phantom,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # articulation
    # configure alternative robots in derived environments.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_FORCE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # robot: ArticulationCfg = FRANKA_PANDA_REALSENSE_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Robot"
    # )
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # sensors
    # ToDo: switch to a tiled camera
    room_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/third_person_cam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane"],
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
    )

    wrist_camera = CameraCfg(
        data_types=["rgb", "distance_to_image_plane"],
        prim_path="{ENV_REGEX_NS}/Robot/D405_rigid/D405/Camera_OmniVision_OV9782_Color",
        spawn=None,
        height=224,
        width=224,
        update_period=0.0,
    )

    # Frame definitions for the goal frame
    goal_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/goal_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/organs",
                name="goal_frame",
                offset=OffsetCfg(
                    pos=(0.0, -0.25, 0.75),
                    rot=(
                        0,
                        1,
                        0,
                        0,
                    ),  # rotate 180 about x-axis to make the end-effector point down
                ),
            ),
        ],
    )

    # Transform Mesh to Organ Frame.
    # Mesh object files are used to simulate ultrasound images.
    # It abides by the original coordinate system specified in the object files.
    # Usually it doesn't align with the organ frame, which follows USD convention.
    # We need this transsform to compute the transformation between.
    # Particularly, we used to to derive how the ultrasound probe sees the mesh objects
    mesh_to_organ_transform = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_TINY_CFG.replace(prim_path="/Visuals/mesh_to_organ_transform"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/organs",
                name="mesh_to_organ_transform",
                offset=OffsetCfg(
                    # Describe how to position the mesh objects relative to the organ.
                    # The mesh objects needs to be aligned so that it looks like organs in the body.
                    # The center of the meshes can be offset from the origin of the mesh coordinate frame.
                    # The center of the organ seems to be always the origin of the USD file.
                    # To align center to center, we need to offset the mesh to the center
                    # This value[2] can be a large negative number, e.g. -360 in some previous asset case.
                    pos=[0.0, 0.0, 0.0],  # unit: m, offset from the center of the organ to the origin
                    # Describe the orientation of the organ frame in the  mesh coords frame
                    # The mesh is used to generate US raytracing, and can be found in the assets folder
                    # It can be also be the Nifti frame, if the axes are aligned.
                    rot=(0.7071, 0.7071, 0, 0),  # quaternion
                ),
            ),
        ],
    )

    # Transform Organ to EE Frame.
    # The displacement/rotation between the organ and the end-effector changes during the task
    # This transform is used to track the relative displacement/rotation.
    organ_to_ee_transform = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/organ_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/TCP",
                name="organ_to_ee_transform",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ],
    )

    # Transform EE Frame to US Frame.
    # The end-effector follows the USD convention but the US image does not.
    # Usually US image uses z for the depth direction, and x for the lateral.
    # End-effector uses z for the depth direction, but y for the lateral.
    # This transform is used to compute the transformation between the two coordinate systems.
    ee_to_us_transform = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/TCP",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_TINY_CFG.replace(prim_path="/Visuals/ee_to_us_transform"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/TCP",
                name="organ_to_ee_transform",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    # The EE and US frame are following different conventions, though both are right-handed.
                    # Please check https://github.com/isaac-for-healthcare/i4h-workflows/pull/60#discussion_r1996523645
                    # for more details.
                    rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, -90.0]), degrees=True),
                ),
            ),
        ],
    )


##
# MDP settings
##
@configclass
class EmptyCommandsCfg:
    """Command terms for the MDP."""

    pass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # We can later use this to alternate goals for the robot
    # It's not strictly necessary. The agent can learn based on observations, actions and rewards.
    target_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.45),
            pos_y=(0.0, 0.0),
            pos_z=(0.75, 0.75),
            roll=(1.5708, 1.5708),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # set the joint positions as target
    # joint_pos_des = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0,
    # use_default_offset=True)
    # overwrite in post_init
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING


@configclass
class PoseObservationsCfg:
    """Observation specifications for the environment."""

    # todo: add camera as observation term

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RGBDObservationsCfg:
    """Observation specifications for the environment."""

    # todo: add camera as observation term

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Add camera observation, which combined rgb and depth images
        # use func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"}) instead
        camera_rgbd = ObsTerm(func=mdp.camera_rgbd_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RGBDPoseObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """obersvations of camera and robot pose"""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_pose"})
        actions = ObsTerm(func=mdp.last_action)

        # Add camera observation, which combined rgb and depth images
        camera_rgbd = ObsTerm(func=mdp.camera_rgbd_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # the reset scene to event function already resets all rigid objects and articulations to rheir default states.
    # this needs to be executed before any other reset function, to not overwrite the reset scene to default.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # the second reset only affects the organ body, and adds a random offset to the organ body, w.r.t to
    # the current position.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.0, 0.0), "z": (-0, -0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("organs"),
        },
    )

    # on reset, change the start position of the robot, by slightly modifying the joint positions.
    reset_joint_position = EventTerm(
        func=mdp.reset_panda_joints_by_fraction_of_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    reaching_object = RewTerm(func=mdp.object_ee_distance, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=2.5)

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # distance_to_patient = RewTerm(func=mdp.distance_to_patient, weight=1.0)
    # align_ee_patient = RewTerm(func=mdp.align_ee_patient, weight=1.0)

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


@configclass
class RoboticIkRlEnvCfg(ManagerBasedRLEnvCfg):
    """Base Configuration for the robotic ultrasound environment."""

    # Scene settings
    scene: RoboticSoftCfg = RoboticSoftCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: PoseObservationsCfg = PoseObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # The command generator should ...
    commands: CommandsCfg = EmptyCommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        self.episode_length_s = 5

        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation

        # configure the action
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Set the body name for the end effector
        # self.commands.target_pose.body_name = "panda_hand"

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/ee_frame"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaModRGBDIkRlEnvCfg(RoboticIkRlEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # use the modified franka robot
        self.scene.robot = FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="TCP",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[-0.0, 0.0, 0.0], rot=euler_angles_to_quats(torch.tensor([-0, -0.0, 0.0]), degrees=True)
            ),
        )

        # marker for ee
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.0, 0.0, 0.0)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/TCP",
                    name="end_effector",
                    # Uncomment and configure the offset if needed:
                    offset=OffsetCfg(
                        pos=[-0.0, 0.0, 0.0],
                        rot=euler_angles_to_quats(torch.tensor([-0, -0.0, 0.0]), degrees=True),
                    ),
                )
            ],
        )
