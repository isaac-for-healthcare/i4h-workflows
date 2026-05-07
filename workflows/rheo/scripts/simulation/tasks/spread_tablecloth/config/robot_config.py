# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Robot configuration for the `spread_tablecloth` task.

Supported robot: Unitree G1 (29 DOF body) + Inspire hands, base-fixed.
"""

from typing import Dict, Literal, Optional, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

G1_INSPIRE_USD_PATH = (
    "/home/mxgu/Workspace/Omniverse/gmx/unitree/unitree_sim_isaaclab/"
    "unitree_sim_isaaclab_usds/assets/robots/g1-29dof-inspire-base-fix-usd/"
    "g1_29dof_with_inspire_rev_1_0.usd"
)

DEFAULT_JOINT_POS: Dict[str, float] = {
    # legs
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.3,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.3,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    # inspire hands (left)
    "L_index_proximal_joint": 0.0,
    "L_index_intermediate_joint": 0.0,
    "L_middle_proximal_joint": 0.0,
    "L_middle_intermediate_joint": 0.0,
    "L_pinky_proximal_joint": 0.0,
    "L_pinky_intermediate_joint": 0.0,
    "L_ring_proximal_joint": 0.0,
    "L_ring_intermediate_joint": 0.0,
    "L_thumb_proximal_yaw_joint": 0.0,
    "L_thumb_proximal_pitch_joint": 0.0,
    "L_thumb_intermediate_joint": 0.0,
    "L_thumb_distal_joint": 0.0,
    # inspire hands (right)
    "R_index_proximal_joint": 0.0,
    "R_index_intermediate_joint": 0.0,
    "R_middle_proximal_joint": 0.0,
    "R_middle_intermediate_joint": 0.0,
    "R_pinky_proximal_joint": 0.0,
    "R_pinky_intermediate_joint": 0.0,
    "R_ring_proximal_joint": 0.0,
    "R_ring_intermediate_joint": 0.0,
    "R_thumb_proximal_yaw_joint": 0.0,
    "R_thumb_proximal_pitch_joint": 0.0,
    "R_thumb_intermediate_joint": 0.0,
    "R_thumb_distal_joint": 0.0,
}

SPREAD_TABLECLOTH_INIT_POS: Tuple[float, float, float] = (-0.95, 0.0, 0.80)
SPREAD_TABLECLOTH_INIT_ROT: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
SPREAD_TABLECLOTH_CUSTOM_JOINT_POS: Dict[str, float] = {
    "left_shoulder_pitch_joint": -0.3,
    "left_shoulder_roll_joint": 0.5,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.5,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": -0.5,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


G129_CFG_WITH_INSPIRE_HAND = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_INSPIRE_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            "L_index_proximal_joint": 0.0,
            "L_index_intermediate_joint": 0.0,
            "L_middle_proximal_joint": 0.0,
            "L_middle_intermediate_joint": 0.0,
            "L_pinky_proximal_joint": 0.0,
            "L_pinky_intermediate_joint": 0.0,
            "L_ring_proximal_joint": 0.0,
            "L_ring_intermediate_joint": 0.0,
            "L_thumb_proximal_yaw_joint": 0.0,
            "L_thumb_proximal_pitch_joint": 0.0,
            "L_thumb_intermediate_joint": 0.0,
            "L_thumb_distal_joint": 0.0,
            "R_index_proximal_joint": 0.0,
            "R_index_intermediate_joint": 0.0,
            "R_middle_proximal_joint": 0.0,
            "R_middle_intermediate_joint": 0.0,
            "R_pinky_proximal_joint": 0.0,
            "R_pinky_intermediate_joint": 0.0,
            "R_ring_proximal_joint": 0.0,
            "R_ring_intermediate_joint": 0.0,
            "R_thumb_proximal_yaw_joint": 0.0,
            "R_thumb_proximal_pitch_joint": 0.0,
            "R_thumb_intermediate_joint": 0.0,
            "R_thumb_distal_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness={"waist_yaw_joint": 10000.0, "waist_roll_joint": 10000.0, "waist_pitch_joint": 10000.0},
            damping={"waist_yaw_joint": 10000.0, "waist_roll_joint": 10000.0, "waist_pitch_joint": 10000.0},
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=None,
            stiffness=None,
            damping=None,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness={
                ".*_shoulder_.*_joint": 25.0,
                ".*_elbow_joint": 50.0,
                ".*_wrist_.*_joint": 40.0,
            },
            damping={
                ".*_shoulder_.*_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_.*_joint": 2.0,
            },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_proximal_joint",
                ".*_index_intermediate_joint",
                ".*_middle_proximal_joint",
                ".*_middle_intermediate_joint",
                ".*_pinky_proximal_joint",
                ".*_pinky_intermediate_joint",
                ".*_ring_proximal_joint",
                ".*_ring_intermediate_joint",
                ".*_thumb_proximal_yaw_joint",
                ".*_thumb_proximal_pitch_joint",
                ".*_thumb_intermediate_joint",
                ".*_thumb_distal_joint",
            ],
            effort_limit=100.0,
            velocity_limit=50.0,
            stiffness={
                ".*_index_proximal_joint": 1000.0,
                ".*_index_intermediate_joint": 1000.0,
                ".*_middle_proximal_joint": 1000.0,
                ".*_middle_intermediate_joint": 1000.0,
                ".*_pinky_proximal_joint": 1000.0,
                ".*_pinky_intermediate_joint": 1000.0,
                ".*_ring_proximal_joint": 1000.0,
                ".*_ring_intermediate_joint": 1000.0,
                ".*_thumb_proximal_yaw_joint": 1000.0,
                ".*_thumb_proximal_pitch_joint": 1000.0,
                ".*_thumb_intermediate_joint": 1000.0,
                ".*_thumb_distal_joint": 1000.0,
            },
            damping={
                ".*_index_proximal_joint": 15.0,
                ".*_index_intermediate_joint": 15.0,
                ".*_middle_proximal_joint": 15.0,
                ".*_middle_intermediate_joint": 15.0,
                ".*_pinky_proximal_joint": 15.0,
                ".*_pinky_intermediate_joint": 15.0,
                ".*_ring_proximal_joint": 15.0,
                ".*_ring_intermediate_joint": 15.0,
                ".*_thumb_proximal_yaw_joint": 15.0,
                ".*_thumb_proximal_pitch_joint": 15.0,
                ".*_thumb_intermediate_joint": 15.0,
                ".*_thumb_distal_joint": 15.0,
            },
            armature={".*": 0.0},
        ),
    },
)


def make_g1_29dof_inspire_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.76),
    init_rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.7071, 0.7071),
    custom_joint_pos: Optional[Dict[str, float]] = None,
    base_config: ArticulationCfg = G129_CFG_WITH_INSPIRE_HAND,
) -> ArticulationCfg:
    """Create the robot articulation cfg for this task."""
    joint_pos = DEFAULT_JOINT_POS.copy()
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return base_config.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class G1RobotPresets:
    """G1 robot preset configuration collection for the spread_tablecloth task."""

    @classmethod
    def g1_29dof_inspire_base_fix(
        cls,
        init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.76),
        init_rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.7071, 0.7071),
        custom_joint_pos: Optional[Dict[str, float]] = None,
    ) -> ArticulationCfg:
        """G1 29DOF + Inspire hands, base-fixed configuration."""
        return make_g1_29dof_inspire_cfg(
            init_pos=init_pos,
            init_rot=init_rot,
            custom_joint_pos=custom_joint_pos,
        )
