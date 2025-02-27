# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extension to the configuration for the Franka Emika robots.

From: Reference: https://github.com/frankaemika/franka_ros
The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

They are now extended by
* :obj:`FRANKA_PANDA_REALSENSE_CFG`: Franka Emika Panda robot with Panda hand and Intel Realsense camera


"""
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os 
##
# Configuration
##

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

NOHAND_FRANKA_PANDA = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/localhost/Library/ultrasound/assemblies/assembly.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

""" Configuration for the Franka Emika robot with realsense camera."""
FRANKA_PANDA_REALSENSE_CFG = FRANKA_PANDA_CFG.copy()

# local filepath
spawn = sim_utils.UsdFileCfg(
    usd_path="omniverse://localhost/Library/ultrasound/franka_realsense_no_world.usd",
    activate_contact_sensors=False,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
    ),
    # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
)
FRANKA_PANDA_REALSENSE_CFG.spawn = spawn
FRANKA_PANDA_REALSENSE_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_REALSENSE_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_REALSENSE_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_REALSENSE_CFG.actuators["panda_forearm"].damping = 80.0


""" Configuration for the Franka Emika robot with realsense camera."""
FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG = NOHAND_FRANKA_PANDA.copy()
# local filepath
spawn = sim_utils.UsdFileCfg(
    usd_path=os.path.join(os.getcwd(), "assets/Collected_panda_assembly/panda_assembly.usda"),
    # usd_path="omniverse://localhost/Library/sage/2025-10-02/robot/Collected_panda_assembly/panda_assembly.usda",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
    ),
    # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
)
FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.spawn = spawn
FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.actuators["panda_forearm"].damping = 80.0

# High PD Force Control
FRANKA_PANDA_HIGH_PD_FORCE_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_FORCE_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_FORCE_CFG.spawn.activate_contact_sensors = True
FRANKA_PANDA_HIGH_PD_FORCE_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_FORCE_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_FORCE_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_FORCE_CFG.actuators["panda_forearm"].damping = 80.0
