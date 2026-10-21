# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-ARM 101 Arena embodiment (5-DoF arm + 1-DoF jaw).

Defines the SO-ARM articulation cfg (:data:`SOARM101_CFG`), its onboard wrist
camera (built into the USD), and a default room camera. :class:`SoArm101Embodiment`
is the Arena embodiment built from those. Other modules (the scissor scene cfg)
import these directly.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas import CollisionBaseCfg, RigidBodyBaseCfg
from isaaclab.utils import configclass
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

from i4h_arena.assets.constants import SOARM101_USD
from i4h_arena.embodiments.recorders import action_state_recorders

_ARM_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_GRIPPER_NAMES = ["gripper"]
_ZERO_JOINT_POS_RAD = (0.0,) * (len(_ARM_NAMES) + len(_GRIPPER_NAMES))


SOARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SOARM101_USD,
        visible=True,
        copy_from_source=True,
        activate_contact_sensors=True,
        rigid_props=RigidBodyBaseCfg(disable_gravity=False),
        collision_props=CollisionBaseCfg(contact_offset=0.005, rest_offset=0.001),
        articulation_props=PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.1, 0.2),
        rot=(0.0, 0.0, -0.707, 0.707),
        joint_pos=dict(zip((*_ARM_NAMES, *_GRIPPER_NAMES), _ZERO_JOINT_POS_RAD, strict=True)),
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=_ARM_NAMES,
            effort_limit_sim=5.2,
            stiffness=80.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=_GRIPPER_NAMES,
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=10.0,
        ),
    },
)


SOARM_WRIST_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/gripper/visuals/pcb_board_36x36/Camera",
    spawn=None,
    data_types=["rgb"],
    width=640,
    height=480,
    update_period=1 / 30.0,
)

SOARM_ROOM_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/RoomCamera",
    offset=CameraCfg.OffsetCfg(pos=(0.12, 0.08, 0.7), rot=(0.7071, -0.7071, 0.0, 0.0), convention="ros"),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0,
        focus_distance=100.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    ),
    width=640,
    height=480,
    update_period=1 / 30.0,
)


@configclass
class _ActionsCfg:
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class _ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        wrist = ObsTerm(
            func=mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )
        room = ObsTerm(
            func=mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("room"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _default_events_cfg():
    @configclass
    class EmbodimentEventsCfg:
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

    return EmbodimentEventsCfg()


def _action_cfg(action_device: str) -> _ActionsCfg:
    cfg = _ActionsCfg()
    if action_device == "joint_position":
        # The Actuation protocol carries absolute joint targets. IsaacLab's
        # default would add the articulation home pose a second time.
        cfg.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=_ARM_NAMES, scale=1.0, use_default_offset=False
        )
        cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=_GRIPPER_NAMES, scale=1.0, use_default_offset=False
        )
    elif action_device == "keyboard":
        cfg.arm_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=_ARM_NAMES, scale=1.0)
        cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=_GRIPPER_NAMES, scale=0.7
        )
    else:
        raise ValueError(f"Unsupported action device: {action_device}")
    return cfg


@configclass
class _RobotSceneCfg:
    robot = SOARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")


@configclass
class _CameraSceneCfg(ArenaCameraCfg):
    wrist = SOARM_WRIST_CAMERA_CFG
    room = SOARM_ROOM_CAMERA_CFG


class SoArm101Embodiment(EmbodimentBase):
    """SO-ARM 101 with wrist and room cameras."""

    name = "so_arm101"
    tags = ["embodiment", "so_arm", "manipulation"]
    #: Merged into the scene's RTX settings by :meth:`i4h_arena.scenes.base.Scene.arena_environment`.
    rtx_settings = {"enable_translucency": True}

    def __init__(
        self,
        enable_cameras: bool = True,
        initial_pose: Pose | None = None,
        action_device: str = "joint_position",
        home_joint_pos_rad: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.action_device = action_device
        self.home_joint_pos_rad = tuple(home_joint_pos_rad or _ZERO_JOINT_POS_RAD)
        self.scene_config = _RobotSceneCfg()
        self.camera_config = _CameraSceneCfg()
        # Isaac Lab 3's Camera already renders tiled; TiledCameraCfg is deprecated.
        self.camera_config.set_use_tiled_camera(False)
        self.action_config = _action_cfg(action_device)
        self.observation_config = _ObservationsCfg()
        self.event_config = _default_events_cfg()

    def get_scene_cfg(self) -> Any:
        self.scene_config.robot.init_state.joint_pos = dict(
            zip((*_ARM_NAMES, *_GRIPPER_NAMES), self.home_joint_pos_rad, strict=True)
        )
        return super().get_scene_cfg()

    def get_observation_cfg(self) -> Any:
        # The policy group already carries both camera images under their own
        # names; Arena's generated camera group would add second copies.
        return self.observation_config

    def get_recorder_term_cfg(self, record_trajectories: bool = False) -> Any:
        return action_state_recorders(super().get_recorder_term_cfg(record_trajectories))

    def get_ee_frame_transformer_names(self) -> list[str]:
        # The SO-ARM scene tracks no end-effector frame sensor.
        return []

    def tune_env_cfg(self, env_cfg: Any) -> Any:
        env_cfg.sim.dt = 1 / 60.0
        env_cfg.sim.render_interval = 1
        env_cfg.decimation = 1
        env_cfg.scene.replicate_physics = True
        env_cfg.sim.physics.bounce_threshold_velocity = 0.01
        env_cfg.sim.physics.friction_correlation_distance = 0.00625

        if self.action_device == "keyboard":
            env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        return env_cfg
