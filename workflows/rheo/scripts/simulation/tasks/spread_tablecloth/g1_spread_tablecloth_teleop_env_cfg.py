import os
import tempfile

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik import FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from .config import (
    DEFAULT_JOINT_POS,
    G129_CFG_WITH_INSPIRE_HAND,
    SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    SPREAD_TABLECLOTH_INIT_POS,
    SPREAD_TABLECLOTH_INIT_ROT,
)
from .g1_spread_tablecloth_env_cfg import G1SpreadTableclothEnvCfg

# IsaacTeleop-native cfgs. Follow the upstream IsaacLab G1 Inspire example
# (pickplace_unitree_g1_inspire_hand_env_cfg.py): use a single XrCfg from
# ``isaaclab_teleop.xr_cfg`` for both ``self.xr`` and ``isaac_teleop.xr_cfg``.
# The legacy ``isaaclab.devices.openxr.xr_cfg.XrCfg`` belongs to the deprecated
# native XR stack and is NOT what IsaacTeleop's XrAnchorManager reads.
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg

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


# ---------------------------------------------------------------------------
# IsaacTeleop action-tensor layout for the hand-tracking pipeline.  Retargeter
# outputs are flattened through a TensorReorderer so the downstream
# PinkInverseKinematicsActionCfg sees a single fixed 38-D action tensor.
# ---------------------------------------------------------------------------
_LEFT_EE_ELEMENTS = [
    "l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
]
_RIGHT_EE_ELEMENTS = [
    "r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
]

_LEFT_HAND_JOINT_NAMES = [
    "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint", "L_thumb_distal_joint",
    "L_index_proximal_joint", "L_index_intermediate_joint",
    "L_middle_proximal_joint", "L_middle_intermediate_joint",
    "L_ring_proximal_joint", "L_ring_intermediate_joint",
    "L_pinky_proximal_joint", "L_pinky_intermediate_joint",
]
_RIGHT_HAND_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
    "R_index_proximal_joint", "R_index_intermediate_joint",
    "R_middle_proximal_joint", "R_middle_intermediate_joint",
    "R_ring_proximal_joint", "R_ring_intermediate_joint",
    "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
]

_OUTPUT_ORDER = (
    _LEFT_EE_ELEMENTS + _RIGHT_EE_ELEMENTS + [
        "L_index_proximal_joint", "L_middle_proximal_joint",
        "L_pinky_proximal_joint", "L_ring_proximal_joint", "L_thumb_proximal_yaw_joint",
        "R_index_proximal_joint", "R_middle_proximal_joint",
        "R_pinky_proximal_joint", "R_ring_proximal_joint", "R_thumb_proximal_yaw_joint",
        "L_index_intermediate_joint", "L_middle_intermediate_joint",
        "L_pinky_intermediate_joint", "L_ring_intermediate_joint", "L_thumb_proximal_pitch_joint",
        "R_index_intermediate_joint", "R_middle_intermediate_joint",
        "R_pinky_intermediate_joint", "R_ring_intermediate_joint", "R_thumb_proximal_pitch_joint",
        "L_thumb_intermediate_joint", "R_thumb_intermediate_joint",
        "L_thumb_distal_joint", "R_thumb_distal_joint",
    ]
)


def _build_g1_inspire_tablecloth_pipeline():
    """IsaacTeleop retargeting pipeline for G1 Inspire Hand (hand tracking from AVP).

    Reuses the same architecture as the official PickPlace G1 Inspire example:
    Se3AbsRetargeter (wrists) + DexHandRetargeter (fingers) → TensorReorderer → 38D action.
    """
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    left_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=90.0,
        target_offset_yaw=0.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)}
    )

    right_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=180.0,
        target_offset_pitch=-90.0,
        target_offset_yaw=0.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)}
    )

    import isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_dex_retargeting_utils as _dex_utils

    _data_dir = os.path.abspath(os.path.join(os.path.dirname(_dex_utils.__file__), "data"))
    _config_dir = os.path.join(_data_dir, "configs", "dex-retargeting")
    left_yaml_path = os.path.join(_config_dir, "unitree_hand_left_dexpilot.yml")
    right_yaml_path = os.path.join(_config_dir, "unitree_hand_right_dexpilot.yml")

    local_left_urdf = retrieve_file_path(
        f"{ISAACLAB_NUCLEUS_DIR}/Mimic/G1_inspire_assets/retarget_inspire_white_left_hand.urdf"
    )
    local_right_urdf = retrieve_file_path(
        f"{ISAACLAB_NUCLEUS_DIR}/Mimic/G1_inspire_assets/retarget_inspire_white_right_hand.urdf"
    )

    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)

    left_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=left_yaml_path, hand_urdf=local_left_urdf,
            hand_joint_names=_LEFT_HAND_JOINT_NAMES, hand_side="left",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="left_hand",
    )
    connected_left_dex = left_dex.connect({HandsSource.LEFT: hands.output(HandsSource.LEFT)})

    right_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=right_yaml_path, hand_urdf=local_right_urdf,
            hand_joint_names=_RIGHT_HAND_JOINT_NAMES, hand_side="right",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="right_hand",
    )
    connected_right_dex = right_dex.connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": _LEFT_EE_ELEMENTS, "right_ee_pose": _RIGHT_EE_ELEMENTS,
            "left_hand_joints": _LEFT_HAND_JOINT_NAMES, "right_hand_joints": _RIGHT_HAND_JOINT_NAMES,
        },
        output_order=_OUTPUT_ORDER, name="action_reorderer",
        input_types={
            "left_ee_pose": "array", "right_ee_pose": "array",
            "left_hand_joints": "scalar", "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect({
        "left_ee_pose": connected_left_se3.output("ee_pose"),
        "right_ee_pose": connected_right_se3.output("ee_pose"),
        "left_hand_joints": connected_left_dex.output("hand_joints"),
        "right_hand_joints": connected_right_dex.output("hand_joints"),
    })

    return OutputCombiner({"action": connected_reorderer.output("output")})


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
                    frame="g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame="g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                NullSpacePostureTaskCfg(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                        "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
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

    # Single source of truth for the XR anchor. Mirrors the upstream
    # pickplace_unitree_g1_inspire_hand_env_cfg.py pattern so IsaacTeleop's
    # XrAnchorManager and the CloudXR bridge agree on one cfg instance.
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.0, 0.0, 0.0, 1.0),
    )

    def __post_init__(self):
        super().__post_init__()

        _joint_pos = DEFAULT_JOINT_POS.copy()
        _joint_pos.update(SPREAD_TABLECLOTH_CUSTOM_JOINT_POS)

        self.scene.robot = G129_CFG_WITH_INSPIRE_HAND.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=SPREAD_TABLECLOTH_INIT_POS,
                rot=SPREAD_TABLECLOTH_INIT_ROT,
                joint_pos=_joint_pos,
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

        self.sim.render_interval = 4
        self.episode_length_s = 300.0

        self.actions.pink_ik_cfg.controller.usd_path = self.scene.robot.spawn.usd_path
        self.actions.pink_ik_cfg.controller.urdf_output_dir = tempfile.gettempdir()

        # Attach the anchor to the robot pelvis so the operator's XR view
        # follows the robot; smoothed rotation reduces motion sickness.
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_g1_inspire_tablecloth_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
