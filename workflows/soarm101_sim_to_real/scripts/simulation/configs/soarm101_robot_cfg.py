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


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# SO-ARM 101 robot configuration with table environment
SOARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/so101_add_wrist_camera.usd",
        visible=True,  # Ensure all geometry is visible
        copy_from_source=True,  # Copy all geometry from source USD file
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Position robot on top of the table surface  
        pos=(0.4, 0.1, 0.0),
        rot=(0.707, 0.0, 0.0, -0.707),  # -90-degree rotation from y-axis to x-axis
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
            effort_limit=12.0,     # REAL ROBOT: Calibrated gripper force
            velocity_limit=16.0,   # REAL ROBOT: Full speed to match hardware
            stiffness=30.0,        # Moderate stiffness for precise control
            damping=30.0,          # Stable damping for limited range
        ),
    },
)

@configclass
class SoArm101TableSceneCfg(InteractiveSceneCfg):
    """Configuration for SO-ARM 101 with table environment."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Table - Seattle Lab Table from Isaac Nucleus
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # Table at origin
            rot=(1.0, 0.0, 0.0, 0.0),  # Keep table upright
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
    )

    # SO-ARM 101 robot positioned on top of the table
    soarm101 = SOARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/SoArm101")

    # 🎯 SCISSORS - Simple Static Object (Compatible with Isaac Lab)
    scissor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgicalScissor",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.10, 0.0, 0.05),
            rot=(0.707, 0, 0, 0.707),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/new_surgical_scissor.usd", 
            scale=(0.008, 0.006, 0.010), 
            copy_from_source=True,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.75),  # Silver appearance
                metallic=0.8,
                roughness=0.2,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=4,   
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,    
                rest_offset=0.0002,      
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.2,  
            ),
        ),
    )

    tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgicalInstrumentTray",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.08,  0.18, 0.032),
            rot=(0.707, 0.0, 0.0, 0.707),  # 90-degree rotation around x-axis
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT_visual_collision.usd",
            scale=(0.75, 0.75, 0.4),  # Make tray 50% smaller in all dimensions
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.75),  # Silver appearance
                metallic=0.8,
                roughness=0.2,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=1.0,  
            ),
        ),
    )

    # Room Camera - simple camera prim for viewing (not a sensor)
    room_camera = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoomCamera",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.1, 0.05, 0.6),  # Position as specified
            rot=(0.707, 0.0, 0.0, -0.707),
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
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

SOARM101_TABLE_SCENE_CFG = SoArm101TableSceneCfg
