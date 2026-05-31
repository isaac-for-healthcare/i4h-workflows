# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-ARM 101 Arena embodiment (5-DoF arm + 1-DoF jaw).

Defines the SO-ARM articulation cfg (:data:`SOARM101_CFG`), its onboard wrist
camera (built into the USD), and a default room camera. :data:`SoArm101Embodiment`
is built from those via :func:`_make_embodiment`. Other modules (the scissor
scene cfg, the embodiment factory) import these directly.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any, Callable

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from arena.assets.constants import SOARM101_USD
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_arena.utils.configclass import combine_configclass_instances
from isaaclab_arena.utils.pose import Pose

_ARM_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_GRIPPER_NAMES = ["gripper"]
_ZERO_JOINT_POS_RAD = (0.0,) * (len(_ARM_NAMES) + len(_GRIPPER_NAMES))


SOARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SOARM101_USD,
        visible=True,
        copy_from_source=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.001),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.1, 0.2),
        rot=(0.707, 0.0, 0.0, -0.707),
        joint_pos=dict(zip((*_ARM_NAMES, *_GRIPPER_NAMES), _ZERO_JOINT_POS_RAD)),
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=_ARM_NAMES,
            effort_limit=5.2,
            velocity_limit=6.28,
            stiffness=80.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=_GRIPPER_NAMES,
            effort_limit=12.0,
            velocity_limit=31.4,
            stiffness=80.0,
            damping=10.0,
        ),
    },
)


SOARM_WRIST_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/gripper/visuals/pcb_board_36x36/Camera",
    spawn=None,
    data_types=["rgb"],
    width=640,
    height=480,
    update_period=1 / 30.0,
)

SOARM_ROOM_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/RoomCamera",
    offset=TiledCameraCfg.OffsetCfg(pos=(0.12, 0.08, 0.7), rot=(0.0, 0.7071, -0.7071, 0.0), convention="ros"),
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
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )
        room = ObsTerm(
            func=mdp.image,
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


def _make_embodiment(
    *,
    name: str,
    tags: list[str],
    robot_cfg: ArticulationCfg,
    wrist_cfg: TiledCameraCfg,
    room_cfg: TiledCameraCfg,
    action_cfg_fn: Callable[[str], _ActionsCfg],
):
    """Build a per-robot Arena embodiment class with the shared `get_*` interface."""
    cfg_name = name
    cfg_tags = list(tags)
    cfg_robot = robot_cfg
    cfg_wrist = wrist_cfg
    cfg_room = room_cfg

    @configclass
    class RobotSceneCfg:
        robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/robot")

    @configclass
    class CameraSceneCfg:
        wrist = cfg_wrist
        room = cfg_room

    class _Embodiment:
        """Generated Arena embodiment for the active robot."""

        name = cfg_name
        tags = cfg_tags

        def __init__(
            self,
            enable_cameras: bool = True,
            initial_pose: Pose | None = None,
            action_device: str = "joint_position",
            home_joint_pos_rad: list[float] | tuple[float, ...] | None = None,
        ):
            self.enable_cameras = enable_cameras
            self.initial_pose = initial_pose
            self.action_device = action_device
            self.home_joint_pos_rad = tuple(home_joint_pos_rad or _ZERO_JOINT_POS_RAD)
            self.scene_config = RobotSceneCfg()
            self.camera_config = CameraSceneCfg()
            self.action_config = action_cfg_fn(action_device)
            self.observation_config = _ObservationsCfg()
            self.event_config = _default_events_cfg()

        def use_action_device(self, action_device: str) -> None:
            self.action_device = action_device
            self.action_config = action_cfg_fn(action_device)

        def set_initial_pose(self, pose: Pose) -> None:
            self.initial_pose = pose

        def get_scene_cfg(self) -> Any:
            self.scene_config.robot.init_state.joint_pos = dict(
                zip((*_ARM_NAMES, *_GRIPPER_NAMES), self.home_joint_pos_rad)
            )
            if self.initial_pose is not None:
                self.scene_config.robot.init_state.pos = self.initial_pose.position_xyz
                self.scene_config.robot.init_state.rot = self.initial_pose.rotation_wxyz
            if self.enable_cameras:
                return combine_configclass_instances("SceneCfg", self.scene_config, self.camera_config)
            return self.scene_config

        def get_action_cfg(self) -> Any:
            return self.action_config

        def get_observation_cfg(self) -> Any:
            return self.observation_config

        def get_rewards_cfg(self) -> Any:
            return None

        def get_curriculum_cfg(self) -> Any:
            return None

        def get_commands_cfg(self) -> Any:
            return None

        def get_events_cfg(self) -> Any:
            return self.event_config

        def get_xr_cfg(self) -> Any:
            return None

        def get_mimic_env(self) -> Any:
            return None

        def get_recorder_term_cfg(self) -> Any:
            return RecordTerm()

        def get_termination_cfg(self) -> Any:
            return None

        def modify_env_cfg(self, env_cfg: Any) -> Any:
            env_cfg.sim.dt = 1 / 60.0
            env_cfg.sim.render_interval = 1
            env_cfg.decimation = 1
            env_cfg.scene.replicate_physics = True
            env_cfg.sim.physx.bounce_threshold_velocity = 0.01
            env_cfg.sim.physx.friction_correlation_distance = 0.00625
            env_cfg.sim.render.enable_translucency = True

            if self.action_device == "keyboard":
                env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
            return env_cfg

    _Embodiment.__name__ = f"{cfg_name.title().replace('_', '')}Embodiment"
    return _Embodiment


def _action_cfg(action_device: str) -> _ActionsCfg:
    cfg = _ActionsCfg()
    if action_device == "joint_position":
        cfg.arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=_ARM_NAMES, scale=1.0)
        cfg.gripper_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=_GRIPPER_NAMES, scale=1.0)
    elif action_device == "keyboard":
        cfg.arm_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=_ARM_NAMES, scale=1.0)
        cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=_GRIPPER_NAMES, scale=0.7
        )
    else:
        raise ValueError(f"Unsupported action device: {action_device}")
    return cfg


SoArm101Embodiment = _make_embodiment(
    name="so_arm101",
    tags=["embodiment", "so_arm", "manipulation"],
    robot_cfg=SOARM101_CFG,
    wrist_cfg=SOARM_WRIST_CAMERA_CFG,
    room_cfg=SOARM_ROOM_CAMERA_CFG,
    action_cfg_fn=_action_cfg,
)
