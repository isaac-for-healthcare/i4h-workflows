# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import tempfile
import torch
from pathlib import Path

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.managers import EventTermCfg as EventTerm, EventTermCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_env_mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.devices.openxr.retargeters import G1UpperBodyRetargeterCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation import mdp

from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.g1_locomanipulation_robot_cfg import (  # isort: skip
    G1_LOCOMANIPULATION_ROBOT_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.pink_controller_cfg import (  # isort: skip
    G1_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
from robotic_us_ext.tasks.ultrasound.pick import mdp
from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets


##
# Scene definition
##
@configclass
class FixedBaseUpperBodyIKG1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for fixed base upper body IK environment with G1 robot.

    This configuration sets up the G1 humanoid robot with fixed pelvis and legs,
    allowing only arm manipulation while the base remains stationary. The robot is
    controlled using upper body IK.
    """

    # Unitree G1 Humanoid robot - fixed base configuration
    # robot: ArticulationCfg = G1_LOCOMANIPULATION_ROBOT_CFG
    robot: ArticulationCfg = G1_LOCOMANIPULATION_ROBOT_CFG.replace(spawn=G1_LOCOMANIPULATION_ROBOT_CFG.spawn.replace(usd_path="/home/yunliu/Downloads/g1_29dof_with_hand_rev_1_0.usd"))

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.50, 0.0015], rot=[1.0, 0.0, 0.0, -180.0]),
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
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.323, 0.40, 0.90], rot=[0.5756, -0.6583, 0.3795, -0.3020]), #v2
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.31706, 0.4358, 0.88759], rot=[0.28662, -0.501798, -0.62415, 0.50988]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.30232, 0.41904, 0.89066], rot=[0.38452, -0.62301, -0.51475, 0.44613]),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yunliu/Workspace/Code/i4h-workflows/RobotTool_US_Probe_v2/RobotTool_US_Probe.usd",
            # usd_path="/home/yunliu/Downloads/HD3C3.usd",
            # usd_path="/home/yunliu/Downloads/HD3C3_visible.usd",
            # usd_path="/home/yunliu/Downloads/c3hd3_fixed_v2.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=10.0,         # increase linear damping to reduce sliding
                angular_damping=10.0,        # increase angular damping to reduce rotation
                max_linear_velocity=5.0,     # limit maximum linear velocity
                max_angular_velocity=5.0,    # limit maximum angular velocity
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,        # contact offset distance
                rest_offset=0.0,             # rest offset distance
            ),
            # scale=(0.01, 0.01, 0.01),  # Example scale
        ),
    )

    black_sorting_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlackSortingBin",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.32, 0.40, 0.83752], rot=[1.0, 0, 0, 0]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.31833, 0.40, 0.83909], rot=[1.0, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
            scale=(0.75, 0.75, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,         # disable gravity
                max_linear_velocity=0.0,      # limit linear velocity to 0
                max_angular_velocity=0.0,     # limit angular velocity to 0
                linear_damping=1000.0,        # high linear damping
                angular_damping=1000.0,       # high angular damping
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000),  # high mass to make it difficult to move
        ),
    )


    # Phantom
    organs = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/organs",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.49, 0.933], rot=euler_angles_to_quats(torch.tensor([0.0, 0.0, 90.0]), degrees=True)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_us_assets.phantom,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                # kinematic_enabled=False,
                # disable_gravity=True,
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

    robot_pov_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotPOVCam",
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=8, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.0978, 0.0917, 1.3136), rot=euler_angles_to_quats(torch.tensor([-123, 0.0, 0.0]), degrees=True), convention="ros"),
    )

    room_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RoomCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.01754, 1.18538, 1.566),
            rot=euler_angles_to_quats(torch.tensor([-118, 0.0, -180.0]), degrees=True),
            convention="ros",
        ),
    )


    def __post_init__(self):
        """Post initialization."""
        # Set the robot to fixed base
        self.robot.spawn.articulation_props.fix_root_link = True

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    This class is required by the environment configuration but not used in this implementation
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        image_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb"})
        # image_distance_to_image_plane = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "distance_to_image_plane"})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )
    success = DoneTerm(
        func=mdp.ultrasound_scanning_task_success,
        params={
            "scan_height_tolerance": 0.20,
            "min_pickup_distance": 0.1,
            "min_scan_distance": 0.2,
        }
    )


##
# Event configuration
##
@configclass 
class EventCfg:
    """Configuration for events."""

    # the reset scene to event function already resets all rigid objects and articulations to rheir default states.
    # this needs to be executed before any other reset function, to not overwrite the reset scene to default.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_my_object = EventTerm(
        func=mdp.reset_specific_object_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object")}
    )
    # Reset task state for ultrasound scanning task  
    reset_task_state = EventTerm(func=mdp.reset_task_state, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.05), "y": (-0.1, 0.05), "z": (-0, -0.0), "yaw": (-3.14 / 2, 3.14 / 2)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )

##
# MDP settings
##


@configclass
class FixedBaseUpperBodyIKG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 fixed base upper body IK environment.

    This environment is designed for manipulation tasks where the G1 humanoid robot
    has a fixed pelvis and legs, allowing only arm and hand movements for manipulation. The robot is
    controlled using upper body IK.
    """

    # Scene settings
    scene: FixedBaseUpperBodyIKG1SceneCfg = FixedBaseUpperBodyIKG1SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 2
        self.viewer.eye = (0.0, 1.8, 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, tempfile.gettempdir(), force_conversion=True
        )

        # Convert revolute joints to fixed joints for pelvis and legs
        ControllerUtils.change_revolute_to_fixed_regex(
            temp_urdf_output_path, self.actions.upper_body_ik.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path


        
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1UpperBodyRetargeterCfg(
                            enable_visualization=False,
                            # OpenXR hand tracking has 26 joints per hand
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
