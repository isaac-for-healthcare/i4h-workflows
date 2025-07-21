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

import os
from typing import Any
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import  TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# Import the base robot configuration
from dataclasses import MISSING
# from . import mdp
import isaaclab.envs.mdp as mdp
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action


ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

SOARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "so101_add_wrist_camera_change_collision.usd"),
        visible=True,  
        copy_from_source=True,  
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01,    
            rest_offset=0.001,      
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Position robot on top of the table surface  
        pos=(0.4, 0.1,-0.02),
        rot=(0.707, 0.0, 0.0, -0.707), 
        joint_pos={
            "shoulder_pan": 0.0,  
            "shoulder_lift": 0.0,  
            "elbow_flex": 0.0,    
            "wrist_flex": 0.0,     
            "wrist_roll": 0.0,    
            "gripper": 0.0,     
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit=5.2,
            velocity_limit=6.28,
            stiffness=80.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit=12.0,   
            velocity_limit=16.0,   
            stiffness=80.0,       
            damping=50.0,         
        ),
    },
)

@configclass
class SoArm101TableSceneCfg(InteractiveSceneCfg):
    """Configuration for SO-ARM 101 with table environment and camera sensors for recording."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = SOARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.15, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"), # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1 / 30.0, # 30FPS
    )

    room:TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/room_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.1, 0.6), rot=(0.0, 0.7071, -0.7071, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=640,
        height=480,
        update_period=1 / 30.0, # 30FPS
    )

    # Table - Seattle Lab Table from Isaac Nucleus
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.4,-0.1, 0.0),  # Table at origin
            rot=(1.0, 0.0, 0.0, 0.0),  # Keep table upright
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.90, 1.90, 1.0), 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
    )

    scissors = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SurgicalScissors",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.02, 0.05, 0.00),
            rot=(0.707, 0, 0, 0.707),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "SurgicalScissors.usd"), 
            scale=(0.006, 0.006, 0.010), 
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.4),
                metallic=0.8,
                roughness=0.3,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=4,   
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,    
                rest_offset=0.001,      
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.1,  
            ),
        ),
    )

    tray = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SurgicalTray",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0,  0.25, 0.012),
            rot=(0.5, 0.5, 0.5, 0.5),  
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "SurgicalTray.usd"),
            scale=(0.75, 0.75, 0.4), 
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.4), # Silver appearance
                metallic=0.8,
                roughness=0.15,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=5.0,  
            ),
        ),
    )

    # Dome light for proper lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0, 
            color=(0.75, 0.75, 0.75)
        ),
    )

    # Additional directional light for better robot visibility
    directional_light = AssetBaseCfg(
        prim_path="/World/DirectionalLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=1000.0,
            color=(1.0, 1.0, 1.0),
            angle=45.0,
        ),
    )



@configclass
class ActionsCfg:
    """Configuration for the actions."""
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING

@configclass
class EventCfg:
    """Configuration for the events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    # Reset scissors to initial position
    reset_scissors = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},  # No randomization, just reset to initial position
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("scissors"),
        },
    )
    
    # Reset tray to initial position  
    reset_tray = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},  # No randomization, just reset to initial position
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray"),
        },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        wrist = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False})
        room = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("room"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Configuration for the rewards"""

@configclass
class TerminationsCfg:
    """Configuration for the termination"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(func=mdp.task_done, params={
        "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
        "plate_cfg": SceneEntityCfg("Plate")
    })

@configclass
class SurgeryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick orange environment."""

    scene: SoArm101TableSceneCfg = SoArm101TableSceneCfg(env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    def __post_init__(self) -> None:
        super().__post_init__()


        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.actions = init_action_cfg(self.actions)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True


    def use_teleop_device(self, teleop_device) -> None:
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device == "keyboard":
            self.scene.robot.spawn.rigid_props.disable_gravity = True
    
    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)
