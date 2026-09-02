# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KUKA LBR14Med Arena embodiment."""

from __future__ import annotations

from typing import Any, ClassVar

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

ASSET_PATH_0_7 = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.7.0/724f82e/"
)
KUKA_LBR14_MED_USD = ASSET_PATH_0_7 + "Robots/KUKA_LBR/LBR14/USD/LBR14Med/LBR14Med.usd"
KUKA_JOINT_NAMES = tuple(f"axis{i}" for i in range(1, 8))
KUKA_HOME_JOINT_POS = {
    "axis1": 0.049207232,
    "axis2": 0.574323643,
    "axis3": -0.203685962,
    "axis4": -1.372580193,
    "axis5": 0.117885163,
    "axis6": 1.206499267,
    "axis7": -0.164665504,
}

_JOINT_FRICTION = 0.20
_JOINT_ARMATURE = 0.03
_JOINT_EFFORT_LIMITS_NM = {
    "axis1": 320.0,
    "axis2": 320.0,
    "axis3": 176.0,
    "axis4": 176.0,
    "axis5": 110.0,
    "axis6": 40.0,
    "axis7": 40.0,
}
_JOINT_VELOCITY_LIMITS_RAD_S = {
    "axis1": 1.483530,
    "axis2": 1.483530,
    "axis3": 1.745329,
    "axis4": 1.308997,
    "axis5": 2.268928,
    "axis6": 2.356194,
    "axis7": 2.356194,
}
_PD_GAINS = {
    "axis1": (720.0, 291.836835),
    "axis2": (989.2432250976562, 180.0),
    "axis3": (2149.39794921875, 339.548124),
    "axis4": (1980.458251953125, 268.8757021654006),
    "axis5": (3180.906005859375, 431.8516914836278),
    "axis6": (3125.637451171875, 424.34788293776584),
    "axis7": (730.6450805664062, 99.19616994355966),
}

_MF_FRAME_MARKER_CFG = FRAME_MARKER_CFG.copy()
_MF_FRAME_MARKER_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


def _actuator_cfg(joint_name: str) -> ImplicitActuatorCfg:
    stiffness, damping = _PD_GAINS[joint_name]
    return ImplicitActuatorCfg(
        joint_names_expr=[joint_name],
        effort_limit_sim=_JOINT_EFFORT_LIMITS_NM[joint_name],
        velocity_limit=None,
        velocity_limit_sim=_JOINT_VELOCITY_LIMITS_RAD_S[joint_name],
        stiffness=stiffness,
        damping=damping,
        friction=_JOINT_FRICTION,
        armature=_JOINT_ARMATURE,
    )


KUKA_LBR14_MED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KUKA_LBR14_MED_USD,
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
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=KUKA_HOME_JOINT_POS),
    actuators={joint_name: _actuator_cfg(joint_name) for joint_name in KUKA_JOINT_NAMES},
    soft_joint_pos_limit_factor=1.0,
)


def make_kuka_lbr14_med_cfg() -> ArticulationCfg:
    """Return a copy of the KUKA LBR14Med articulation config."""
    return KUKA_LBR14_MED_CFG.copy()


def make_kuka_lbr14_med_ik_action_cfg() -> mdp.DifferentialInverseKinematicsActionCfg:
    """Return the built-in IsaacLab absolute-pose IK action for the media flange."""
    return mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(KUKA_JOINT_NAMES),
        body_name="mf",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.20},
        ),
        scale=1.0,
    )


@configclass
class _RobotSceneCfg:
    robot = KUKA_LBR14_MED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    mf_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mf",
        debug_vis=True,
        visualizer_cfg=_MF_FRAME_MARKER_CFG.replace(prim_path="/Visuals/kuka_lbr14/mf_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/mf",
                name="mf",
            )
        ],
    )


@configclass
class _ActionsCfg:
    arm_action: mdp.DifferentialInverseKinematicsActionCfg = make_kuka_lbr14_med_ik_action_cfg()


@configclass
class _ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


class KukaLbr14MedEmbodiment(EmbodimentBase):
    """KUKA LBR14Med embodiment with media-flange IK control."""

    name: str = "kuka_lbr14"
    tags: ClassVar[list[str]] = ["embodiment", "kuka", "lbr"]

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Any | None = None,
    ) -> None:
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = _RobotSceneCfg()
        self.scene_config.robot = make_kuka_lbr14_med_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config = _ActionsCfg()
        self.observation_config = _ObservationsCfg()

    def modify_env_cfg(self, env_cfg):
        env_cfg.decimation = 4
        env_cfg.sim.dt = 1 / 200
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.sim.render.rendering_mode = "performance"
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.enable_translucency = False
        env_cfg.sim.render.enable_reflections = False
        env_cfg.sim.render.enable_global_illumination = False
        env_cfg.sim.render.enable_direct_lighting = True
        env_cfg.sim.render.samples_per_pixel = 1
        env_cfg.sim.render.enable_shadows = False
        env_cfg.sim.render.enable_ambient_occlusion = False
        env_cfg.sim.render.dome_light_upper_lower_strategy = 4
        env_cfg.sim.render.enable_dlssg = False
        env_cfg.sim.render.enable_dl_denoiser = False
        return env_cfg

    def get_recorder_term_cfg(self):
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

        return ActionStateRecorderManagerCfg()
