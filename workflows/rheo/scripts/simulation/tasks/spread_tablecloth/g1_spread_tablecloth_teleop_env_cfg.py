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

"""G1 + Inspire teleop env: PinkIK + AVP hand tracking via IsaacTeleop."""

import os
import tempfile

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik import FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg

from .cloth_physics import PinkInverseKinematicsActionOrderedCfg as PinkInverseKinematicsActionCfg
from .config import (
    DEFAULT_JOINT_POS,
    G129_CFG_WITH_INSPIRE_HAND,
    SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    SPREAD_TABLECLOTH_INIT_POS,
    SPREAD_TABLECLOTH_INIT_ROT,
)
from .g1_spread_tablecloth_env_cfg import G1SpreadTableclothEnvCfg

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


# 38-D action: wrist SE3 (left/right) + Inspire finger joints (reordered).
_LEFT_EE_ELEMENTS = [
    "l_pos_x",
    "l_pos_y",
    "l_pos_z",
    "l_quat_x",
    "l_quat_y",
    "l_quat_z",
    "l_quat_w",
]
_RIGHT_EE_ELEMENTS = [
    "r_pos_x",
    "r_pos_y",
    "r_pos_z",
    "r_quat_x",
    "r_quat_y",
    "r_quat_z",
    "r_quat_w",
]

_LEFT_HAND_JOINT_NAMES = [
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
]
_RIGHT_HAND_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
]

_OUTPUT_ORDER = (
    _LEFT_EE_ELEMENTS
    + _RIGHT_EE_ELEMENTS
    + [
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
)


def _build_g1_inspire_tablecloth_pipeline():
    """Build the IsaacTeleop pipeline (called after Kit is up).

    Se3AbsRetargeter (wrists) + DexHandRetargeter (fingers) -> TensorReorderer -> 38D.
    """
    # Deferred imports: isaacteleop must not load until Kit is running.
    import isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_dex_retargeting_utils as _dex_utils
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
    connected_left_se3 = left_se3.connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})

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
    connected_right_se3 = right_se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

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
            hand_retargeting_config=left_yaml_path,
            hand_urdf=local_left_urdf,
            hand_joint_names=_LEFT_HAND_JOINT_NAMES,
            hand_side="left",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="left_hand",
    )
    connected_left_dex = left_dex.connect({HandsSource.LEFT: hands.output(HandsSource.LEFT)})

    right_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=right_yaml_path,
            hand_urdf=local_right_urdf,
            hand_joint_names=_RIGHT_HAND_JOINT_NAMES,
            hand_side="right",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="right_hand",
    )
    connected_right_dex = right_dex.connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": _LEFT_EE_ELEMENTS,
            "right_ee_pose": _RIGHT_EE_ELEMENTS,
            "left_hand_joints": _LEFT_HAND_JOINT_NAMES,
            "right_hand_joints": _RIGHT_HAND_JOINT_NAMES,
        },
        output_order=_OUTPUT_ORDER,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_hand_joints": connected_left_dex.output("hand_joints"),
            "right_hand_joints": connected_right_dex.output("hand_joints"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


@configclass
class TeleopActionsCfg:
    """38-D PinkIK action for AVP teleop with Inspire hand."""

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
        # See cloth_physics.PinkInverseKinematicsActionOrderedCfg: required so
        # Newton (per-finger DFS) doesn't silently reorder the hand-joint slots
        # away from the PhysX-BFS order that the action tensor is packed in.
        preserve_order=True,
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
                    frame="left_wrist_yaw_link",
                    position_cost=1.0,
                    orientation_cost=0.5,
                    gain=0.8,
                ),
                FrameTaskCfg(
                    frame="right_wrist_yaw_link",
                    position_cost=1.0,
                    orientation_cost=0.5,
                    gain=0.8,
                ),
                NullSpacePostureTaskCfg(
                    cost=0.01,
                    lm_damping=1.0,
                    controlled_frames=[
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
        ),
        enable_gravity_compensation=False,
    )


@configclass
class G1SpreadTableclothTeleopEnvCfg(G1SpreadTableclothEnvCfg):
    """AVP teleop variant: G1 29DOF + Inspire hand via PinkIK."""

    actions: TeleopActionsCfg = TeleopActionsCfg()

    # Single XR anchor shared by IsaacTeleop's XrAnchorManager and CloudXR bridge.
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

        # Lock waist to prevent shaking.
        self.scene.robot.actuators["waist"] = ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness=10000.0,
            damping=10000.0,
        )

        self.episode_length_s = 120.0
        self.actions.pink_ik_cfg.controller.usd_path = self.scene.robot.spawn.usd_path
        self.actions.pink_ik_cfg.controller.urdf_output_dir = tempfile.gettempdir()

        # XR anchor follows the robot pelvis; smoothed rotation reduces motion sickness.
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_g1_inspire_tablecloth_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
