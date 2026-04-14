import tempfile

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik import FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode, XrCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

from .h2_spread_tablecloth_env_cfg import G1SpreadTableclothEnvCfg

INSPIRE_HAND_JOINT_NAMES = [
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]


@configclass
class TeleopActionsCfg:
    """38-D PINK IK action for Meta Quest teleoperation with Inspire hand."""

    pink_ik_cfg: ActionTermCfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_yaw_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        hand_joint_names=INSPIRE_HAND_JOINT_NAMES,
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=24,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="g1_29dof_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame="g1_29dof_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                NullSpacePostureTaskCfg(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "g1_29dof_rev_1_0_left_wrist_yaw_link",
                        "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
        ),
        enable_gravity_compensation=False,
    )


@configclass
class G1SpreadTableclothTeleopEnvCfg(G1SpreadTableclothEnvCfg):
    """Meta Quest teleoperation variant with G1 29DOF + Inspire hand.

    Uses PINK IK for arm control and direct trigger-based Inspire hand open/close.
    """

    actions: TeleopActionsCfg = TeleopActionsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.0, 0.0, 0.0, 1.0),
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_INSPIRE_FTP_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.95, 0.0, 0.80),
                rot=(0.0, 0.0, 0.0, 1.0),
                joint_pos={
                    ".*_hip_pitch_joint": -0.05,
                    ".*_knee_joint": 0.2,
                    ".*_ankle_pitch_joint": -0.15,
                    "waist_.*": 0.0,
                    ".*_shoulder_pitch_joint": 0.0,
                    ".*_shoulder_roll_joint": 0.0,
                    ".*_shoulder_yaw_joint": 0.0,
                    ".*_elbow_joint": -0.3,
                    ".*_wrist_.*_joint": 0.0,
                    ".*_thumb_.*": 0.0,
                    ".*_index_.*": 0.0,
                    ".*_middle_.*": 0.0,
                    ".*_ring_.*": 0.0,
                    ".*_pinky_.*": 0.0,
                },
                joint_vel={".*": 0.0},
            ),
        )

        # Lock waist with very high stiffness to prevent shaking
        self.scene.robot.actuators["waist"] = ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness=10000.0,
            damping=10000.0,
        )

        self.sim.render_interval = 2
        self.episode_length_s = 300.0

        self.actions.pink_ik_cfg.controller.usd_path = self.scene.robot.spawn.usd_path
        self.actions.pink_ik_cfg.controller.urdf_output_dir = tempfile.gettempdir()

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
