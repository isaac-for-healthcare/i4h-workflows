# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM Arena embodiment.

The policy-facing action contract is seven-dimensional: six arm joints plus
one logical gripper aperture. The physical USD has two opposing jaw joints;
the binary gripper action maps the logical channel onto both jaws.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from arena.assets.constants import DVRK_PSM_USD
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
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
]
_GRIPPER_NAMES = ["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"]
_DEFAULT_LOGICAL_HOME = (0.01, 0.01, 0.07, 0.01, 0.01, 0.01, 0.09)


def _physical_joint_pos(home_joint_pos: list[float] | tuple[float, ...] | None) -> dict[str, float]:
    values = tuple(home_joint_pos or _DEFAULT_LOGICAL_HOME)
    if len(values) == len(_ARM_NAMES) + len(_GRIPPER_NAMES):
        return dict(zip((*_ARM_NAMES, *_GRIPPER_NAMES), values))
    if len(values) != len(_DEFAULT_LOGICAL_HOME):
        raise ValueError(
            f"PSM home pose must have {len(_DEFAULT_LOGICAL_HOME)} logical values "
            f"or {len(_ARM_NAMES) + len(_GRIPPER_NAMES)} physical values, got {len(values)}"
        )
    arm = values[: len(_ARM_NAMES)]
    aperture = float(values[-1])
    return dict(zip((*_ARM_NAMES, *_GRIPPER_NAMES), (*arm, -aperture, aperture)))


PSM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DVRK_PSM_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=_physical_joint_pos(_DEFAULT_LOGICAL_HOME),
        pos=(0.0, 0.0, 0.15),
    ),
    actuators={
        "psm": ImplicitActuatorCfg(
            joint_names_expr=_ARM_NAMES,
            effort_limit=12.0,
            velocity_limit=1.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "psm_tool": ImplicitActuatorCfg(
            joint_names_expr=["psm_tool_gripper.*"],
            effort_limit=0.1,
            velocity_limit=0.2,
            stiffness=500.0,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

PSM_HIGH_PD_CFG = PSM_CFG.copy()
PSM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
PSM_HIGH_PD_CFG.actuators["psm"].stiffness = 800.0
PSM_HIGH_PD_CFG.actuators["psm"].damping = 40.0

PSM_ROOM_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/RoomCamera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.22, -0.32, 0.28),
        rot=(-0.478864, 0.83348, 0.239039, -0.137336),
        convention="ros",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=10.5,
        focus_distance=100.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 1.0e5),
    ),
    width=640,
    height=480,
    update_period=1 / 30.0,
)

_MARKER_CFG = FRAME_MARKER_CFG.copy()
_MARKER_CFG.markers["frame"].scale = (0.01, 0.01, 0.01)
_MARKER_CFG.prim_path = "/Visuals/PSMFrameTransformer"


@configclass
class _RobotSceneCfg:
    robot = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_base_link",
        debug_vis=False,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_link",
                name="end_effector",
            ),
        ],
    )


@configclass
class _CameraSceneCfg:
    room = PSM_ROOM_CAMERA_CFG


@configclass
class _ActionsCfg:
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg | None = None


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


def _action_cfg(action_device: str, *, gripper_close: float = 0.09, include_gripper: bool = True) -> _ActionsCfg:
    cfg = _ActionsCfg()
    if action_device == "joint_position":
        cfg.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_ARM_NAMES,
            scale=0.5,
            use_default_offset=True,
        )
    elif action_device == "keyboard":
        cfg.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=_ARM_NAMES,
            scale=0.5,
        )
    elif action_device == "ik_abs":
        cfg.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=_ARM_NAMES,
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )
    else:
        raise ValueError(f"Unsupported action device: {action_device}")
    if include_gripper:
        cfg.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -gripper_close, "psm_tool_gripper2_joint": gripper_close},
        )
    return cfg


class PsmEmbodiment:
    name = "psm"
    tags = ["embodiment", "psm", "surgical"]

    def __init__(
        self,
        enable_cameras: bool = True,
        initial_pose: Pose | None = None,
        action_device: str = "joint_position",
        home_joint_pos_rad: list[float] | tuple[float, ...] | None = None,
        sim_dt: float = 1.0 / 60.0,
        sim_decimation: int = 1,
        render_interval: int | None = None,
        gripper_close: float = 0.09,
        include_gripper_action: bool = True,
        enable_material_randomization: bool = True,
    ):
        self.enable_cameras = enable_cameras
        self.initial_pose = initial_pose
        self.action_device = action_device
        self.home_joint_pos_rad = tuple(home_joint_pos_rad or _DEFAULT_LOGICAL_HOME)
        self.sim_dt = sim_dt
        self.sim_decimation = sim_decimation
        self.render_interval = sim_decimation if render_interval is None else render_interval
        self.gripper_close = gripper_close
        self.include_gripper_action = include_gripper_action
        self.enable_material_randomization = enable_material_randomization
        self.scene_config = _RobotSceneCfg()
        self.camera_config = _CameraSceneCfg()
        self.action_config = _action_cfg(
            action_device,
            gripper_close=self.gripper_close,
            include_gripper=self.include_gripper_action,
        )
        self.observation_config = _ObservationsCfg()
        if not self.enable_cameras:
            self.observation_config.policy.room = None
        self.event_config = _EventsCfg() if self.enable_material_randomization else None

    def use_action_device(self, action_device: str) -> None:
        self.action_device = action_device
        self.action_config = _action_cfg(
            action_device,
            gripper_close=self.gripper_close,
            include_gripper=self.include_gripper_action,
        )

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_scene_cfg(self) -> Any:
        self.scene_config.robot.init_state.joint_pos = _physical_joint_pos(self.home_joint_pos_rad)
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


PSMEmbodiment = PsmEmbodiment


@configclass
class _DualRobotSceneCfg:
    robot_1 = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
    robot_2 = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    robot_1.init_state.pos = (0.2, 0.0, 0.15)
    robot_2.init_state.pos = (-0.2, 0.0, 0.15)
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/psm_base_link",
        debug_vis=False,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_tip_link",
                name="end_effector",
            ),
        ],
    )
    ee_2_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot_2/psm_base_link",
        debug_vis=False,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_tip_link",
                name="end_effector",
            ),
        ],
    )


@configclass
class _DualActionsCfg:
    arm_1_action: mdp.ActionTermCfg = MISSING
    gripper_1_action: mdp.ActionTermCfg | None = None
    arm_2_action: mdp.ActionTermCfg = MISSING
    gripper_2_action: mdp.ActionTermCfg | None = None


@configclass
class _DualObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_1")})
        right_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_2")})
        left_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        right_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot_2")})
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
class _DualEventsCfg:
    robot_1_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    robot_2_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_2", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )


def _dual_action_cfg(action_device: str) -> _DualActionsCfg:
    cfg = _DualActionsCfg()
    if action_device == "joint_position":
        arm_cls = mdp.JointPositionActionCfg
        arm_kwargs = {"scale": 0.5, "use_default_offset": True}
    elif action_device == "keyboard":
        arm_cls = mdp.RelativeJointPositionActionCfg
        arm_kwargs = {"scale": 0.5}
    elif action_device == "ik_abs":
        arm_cls = DifferentialInverseKinematicsActionCfg
        arm_kwargs = {
            "body_name": "psm_tool_tip_link",
            "controller": DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        }
    else:
        raise ValueError(f"Unsupported action device: {action_device}")
    cfg.arm_1_action = arm_cls(asset_name="robot_1", joint_names=_ARM_NAMES, **arm_kwargs)
    cfg.arm_2_action = arm_cls(asset_name="robot_2", joint_names=_ARM_NAMES, **arm_kwargs)
    if action_device != "ik_abs":
        cfg.gripper_1_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.09, "psm_tool_gripper2_joint": 0.09},
        )
        cfg.gripper_2_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.09, "psm_tool_gripper2_joint": 0.09},
        )
    return cfg


def _dual_physical_joint_pos(
    home_joint_pos: list[float] | tuple[float, ...] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    values = tuple(home_joint_pos or (*_DEFAULT_LOGICAL_HOME, *_DEFAULT_LOGICAL_HOME))
    if len(values) != 2 * len(_DEFAULT_LOGICAL_HOME):
        raise ValueError(
            f"dual PSM home pose must have {2 * len(_DEFAULT_LOGICAL_HOME)} logical values, got {len(values)}"
        )
    return _physical_joint_pos(values[: len(_DEFAULT_LOGICAL_HOME)]), _physical_joint_pos(
        values[len(_DEFAULT_LOGICAL_HOME) :]
    )


class DualPsmEmbodiment:
    name = "dual_psm"
    tags = ["embodiment", "dual_psm", "surgical"]

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
        self.home_joint_pos_rad = tuple(home_joint_pos_rad or (*_DEFAULT_LOGICAL_HOME, *_DEFAULT_LOGICAL_HOME))
        self.sim_dt = sim_dt
        self.sim_decimation = sim_decimation
        self.render_interval = sim_decimation if render_interval is None else render_interval
        self.enable_material_randomization = enable_material_randomization
        self.scene_config = _DualRobotSceneCfg()
        self.camera_config = _CameraSceneCfg()
        self.action_config = _dual_action_cfg(action_device)
        self.observation_config = _DualObservationsCfg()
        if not self.enable_cameras:
            self.observation_config.policy.room = None
        self.event_config = _DualEventsCfg() if self.enable_material_randomization else None

    def use_action_device(self, action_device: str) -> None:
        self.action_device = action_device
        self.action_config = _dual_action_cfg(action_device)

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_scene_cfg(self) -> Any:
        robot_1_joint_pos, robot_2_joint_pos = _dual_physical_joint_pos(self.home_joint_pos_rad)
        self.scene_config.robot_1.init_state.joint_pos = robot_1_joint_pos
        self.scene_config.robot_2.init_state.joint_pos = robot_2_joint_pos
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
        env_cfg.scene.replicate_physics = False
        env_cfg.sim.physx.bounce_threshold_velocity = 0.01
        env_cfg.sim.physx.friction_correlation_distance = 0.00625
        env_cfg.sim.render.enable_translucency = True
        return env_cfg
