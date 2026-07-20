# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""STAR Arena embodiment ported from workflows/robotic_surgery."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from arena.assets.constants import STAR_USD
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_arena.utils.configclass import combine_configclass_instances
from isaaclab_arena.utils.pose import Pose

_ARM_NAMES = [
    "star_joint_1",
    "star_joint_2",
    "star_joint_3",
    "star_joint_4",
    "star_joint_5",
    "star_joint_6",
    "star_joint_7",
]
_TOOL_NAMES = ["endo360_joint_1"]
_DEFAULT_HOME = (0.0, -0.569, 0.0, -2.0, 0.0, 2.037, 0.741, 0.04)


STAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=STAR_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=dict(zip((*_ARM_NAMES, *_TOOL_NAMES), _DEFAULT_HOME)),
    ),
    actuators={
        "star": ImplicitActuatorCfg(
            joint_names_expr=["star_joint_[1-7]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "endo360": ImplicitActuatorCfg(
            joint_names_expr=_TOOL_NAMES,
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

STAR_HIGH_PD_CFG = STAR_CFG.copy()
STAR_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
STAR_HIGH_PD_CFG.actuators["star"].stiffness = 400.0
STAR_HIGH_PD_CFG.actuators["star"].damping = 80.0

STAR_ROOM_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/RoomCamera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.62, -0.36, 0.62),
        rot=(-0.395891, 0.791743, 0.372507, -0.276788),
        convention="ros",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=12.0,
        focus_distance=100.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 1.0e5),
    ),
    width=640,
    height=480,
    update_period=1 / 30.0,
)

_MARKER_CFG = FRAME_MARKER_CFG.copy()
_MARKER_CFG.markers["frame"].scale = (0.03, 0.03, 0.03)
_MARKER_CFG.prim_path = "/Visuals/STARFrameTransformer"


@configclass
class _RobotSceneCfg:
    robot = STAR_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/star_link_0",
        debug_vis=False,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/endo360_needle",
                name="end_effector",
            ),
        ],
    )


@configclass
class _CameraSceneCfg:
    room = STAR_ROOM_CAMERA_CFG


@configclass
class _ActionsCfg:
    joint_action: mdp.ActionTermCfg = MISSING


@configclass
class _ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        room = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("room"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class _EventsCfg:
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


def _action_cfg(action_device: str) -> _ActionsCfg:
    cfg = _ActionsCfg()
    if action_device == "joint_position":
        cfg.joint_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[*_ARM_NAMES, *_TOOL_NAMES],
            scale=0.5,
            use_default_offset=True,
        )
    elif action_device == "keyboard":
        cfg.joint_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[*_ARM_NAMES, *_TOOL_NAMES],
            scale=0.5,
        )
    elif action_device == "ik_abs":
        cfg.joint_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["star_joint_.*"],
            body_name="endo360_needle",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )
    else:
        raise ValueError(f"Unsupported action device: {action_device}")
    return cfg


class StarEmbodiment:
    name = "star"
    tags = ["embodiment", "star", "surgical"]

    def __init__(
        self,
        enable_cameras: bool = True,
        initial_pose: Pose | None = None,
        action_device: str = "joint_position",
        home_joint_pos_rad: list[float] | tuple[float, ...] | None = None,
        sim_dt: float = 1.0 / 60.0,
        sim_decimation: int = 1,
        render_interval: int | None = None,
        enable_material_randomization: bool = True,
    ):
        self.enable_cameras = enable_cameras
        self.initial_pose = initial_pose
        self.action_device = action_device
        self.home_joint_pos_rad = tuple(home_joint_pos_rad or _DEFAULT_HOME)
        self.sim_dt = sim_dt
        self.sim_decimation = sim_decimation
        self.render_interval = sim_decimation if render_interval is None else render_interval
        self.enable_material_randomization = enable_material_randomization
        self.scene_config = _RobotSceneCfg()
        self.camera_config = _CameraSceneCfg()
        self.action_config = _action_cfg(action_device)
        self.observation_config = _ObservationsCfg()
        if not self.enable_cameras:
            self.observation_config.policy.room = None
        self.event_config = _EventsCfg() if self.enable_material_randomization else None

    def use_action_device(self, action_device: str) -> None:
        self.action_device = action_device
        self.action_config = _action_cfg(action_device)

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_scene_cfg(self) -> Any:
        self.scene_config.robot.init_state.joint_pos = dict(zip((*_ARM_NAMES, *_TOOL_NAMES), self.home_joint_pos_rad))
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
        env_cfg.sim.dt = self.sim_dt
        env_cfg.sim.render_interval = self.render_interval
        env_cfg.decimation = self.sim_decimation
        env_cfg.scene.replicate_physics = True
        env_cfg.sim.physx.bounce_threshold_velocity = 0.01
        env_cfg.sim.physx.friction_correlation_distance = 0.00625
        env_cfg.sim.render.enable_translucency = True
        return env_cfg
