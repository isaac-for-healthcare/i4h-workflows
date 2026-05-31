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

"""Extension to the configuration for the Franka Emika robots.

From: Reference: https://github.com/frankaemika/franka_ros
The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

They are now extended by
* :obj:`FRANKA_PANDA_REALSENSE_CFG`: Franka Emika Panda robot with Panda hand and Intel Realsense camera


"""

import isaaclab.sim as sim_utils
from arena.assets.constants import PANDA_USD
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

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

FRANKA_PANDA_REALSENSE_CFG = FRANKA_PANDA_CFG.copy()

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


FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG = NOHAND_FRANKA_PANDA.copy()
spawn = sim_utils.UsdFileCfg(
    usd_path=PANDA_USD,
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
    semantic_tags=[("class", "robot")],
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


# ---- Franka ultrasound Arena embodiment -------------------------------------


import isaaclab.envs.mdp as _mdp  # noqa: E402
import torch as _torch  # noqa: E402
from isaaclab.controllers import DifferentialIKControllerCfg as _DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg as _IKActionCfg  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as _ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as _ObsTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG as _FRAME_MARKER_CFG  # noqa: E402
from isaaclab.sensors import CameraCfg as _CameraCfg  # noqa: E402
from isaaclab.sensors import FrameTransformerCfg as _FrameTransformerCfg  # noqa: E402
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg as _OffsetCfg  # noqa: E402
from isaaclab.utils import configclass as _configclass  # noqa: E402
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase as _EmbodimentBase  # noqa: E402
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats as _euler_to_quats  # noqa: E402

_FRANKA_SEMANTIC_MAPPING = {
    "class:table": (0, 255, 0, 255),
    "class:organ": (0, 0, 255, 255),
    "class:robot": (255, 255, 0, 255),
    "class:ground": (255, 0, 0, 255),
    "class:UNLABELLED": (0, 0, 0, 255),
}


class FrankaUltrasoundEmbodiment(_EmbodimentBase):
    """Franka Panda + Realsense + ultrasound probe, IK-controlled on the TCP."""

    name: str = "franka_ultrasound"
    tags: list[str] = ["embodiment", "franka", "ultrasound"]

    def __init__(self, enable_cameras: bool = True, initial_pose=None):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = _RobotSceneCfg()
        self.camera_config = _CameraSceneCfg()
        self.action_config = _ActionsCfg()
        self.observation_config = _ObservationsCfg()

    def modify_env_cfg(self, env_cfg):
        env_cfg.decimation = 4
        env_cfg.sim.dt = 1 / 200
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.sim.render.enable_translucency = True
        return env_cfg

    def get_recorder_term_cfg(self):
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

        return ActionStateRecorderManagerCfg()


@_configclass
class _RobotSceneCfg:
    robot = FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: _FrameTransformerCfg = _FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            _FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/TCP",
                name="end_effector",
                offset=_OffsetCfg(pos=[0.0, 0.0, 0.0]),
            )
        ],
    )
    ee_to_us_transform: _FrameTransformerCfg = _FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/TCP",
        debug_vis=False,
        visualizer_cfg=_FRAME_MARKER_CFG.replace(prim_path="/Visuals/ee_to_us_transform"),
        target_frames=[
            _FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/TCP",
                name="ee_to_us_transform",
                offset=_OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot=_euler_to_quats(_torch.tensor([0.0, 0.0, -90.0]), degrees=True),
                ),
            )
        ],
    )


@_configclass
class _CameraSceneCfg:
    wrist_camera: _CameraCfg = _CameraCfg(
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        prim_path="{ENV_REGEX_NS}/Robot/D405_rigid/D405/Camera_OmniVision_OV9782_Color",
        spawn=None,
        height=224,
        width=224,
        update_period=0.0,
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping=_FRANKA_SEMANTIC_MAPPING,
    )
    room_camera: _CameraCfg = _CameraCfg(
        prim_path="{ENV_REGEX_NS}/third_person_cam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=_CameraCfg.OffsetCfg(
            pos=(0.55942, 0.56039, 0.36243),
            rot=_euler_to_quats(_torch.tensor([248.0, 0.0, 180.0]), degrees=True),
            convention="ros",
        ),
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping=_FRANKA_SEMANTIC_MAPPING,
    )


@_configclass
class _ActionsCfg:
    arm_action: _IKActionCfg = _IKActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="TCP",
        controller=_DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
        body_offset=_IKActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0],
            rot=_euler_to_quats(_torch.tensor([0.0, 0.0, 0.0]), degrees=True),
        ),
    )


from arena.tasks.ultrasound_liver_scan import object_position_in_robot_root_frame as _object_position  # noqa: E402


@_configclass
class _ObservationsCfg:
    @_configclass
    class PolicyCfg(_ObsGroup):
        joint_pos_rel = _ObsTerm(func=_mdp.joint_pos_rel)
        joint_vel_rel = _ObsTerm(func=_mdp.joint_vel_rel)
        object_position = _ObsTerm(func=_object_position)
        actions = _ObsTerm(func=_mdp.last_action)
        wrist = _ObsTerm(
            func=_mdp.image,
            params={"sensor_cfg": _SceneEntityCfg("wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        room = _ObsTerm(
            func=_mdp.image,
            params={"sensor_cfg": _SceneEntityCfg("room_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
