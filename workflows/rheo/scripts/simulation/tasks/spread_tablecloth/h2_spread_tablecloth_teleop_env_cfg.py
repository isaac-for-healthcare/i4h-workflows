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

"""H2 + Sharpa Wave teleop env: 58D PinkIK (dual wrist SE3 + 44 hand joints).

OpenXR hand-tracking via IsaacTeleop. H2 lower body is fixed.
"""

import os

from isaaclab.controllers.pink_ik import FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg

from .cloth_physics import PinkInverseKinematicsActionOrderedCfg as PinkInverseKinematicsActionCfg
from .config import (
    H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER,
    H2_SHARPA_HAND_URDF_DIR,
    H2_SHARPA_TELEOP_CONFIG_DIR,
    H2_SHARPA_URDF_PATH,
    ensure_h2_sharpa_assets,
)
from .h2_spread_tablecloth_env_cfg import H2SpreadTableclothEnvCfg

_LEFT_HAND_JOINT_NAMES = [
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]

_RIGHT_HAND_JOINT_NAMES = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]

# 58D action: left_wrist_pos(3)+quat(4), right_wrist_pos(3)+quat(4), 44 hand joints.
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

_OUTPUT_ORDER = _LEFT_EE_ELEMENTS + _RIGHT_EE_ELEMENTS + H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER


def _build_h2_sharpa_tablecloth_pipeline():
    """Build the IsaacTeleop pipeline (called after Kit is up).

    Se3AbsRetargeter (wrists) + DexHandRetargeter (Sharpa fingers) -> TensorReorderer -> 58D.
    """
    # Deferred imports: isaacteleop must not load until Kit is running.
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

    # Wrist SE3 retargeters with H2-specific offsets (see isaaclab_arena_h2/h2_pink_pipeline).
    left_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=90.0,
        target_offset_pitch=90.0,
        target_offset_yaw=-90.0,
        target_offset_z=0.05,
        target_offset_x=0.0,
        target_offset_y=0.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})

    right_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=-90.0,
        target_offset_pitch=-90.0,
        target_offset_yaw=90.0,
        target_offset_z=0.05,
        target_offset_x=0.0,
        target_offset_y=0.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    # DexPilot retargeters for Sharpa Wave hands.
    left_yaml_path = os.path.join(H2_SHARPA_TELEOP_CONFIG_DIR, "sharpa_wave_left_dexpilot.yml")
    right_yaml_path = os.path.join(H2_SHARPA_TELEOP_CONFIG_DIR, "sharpa_wave_right_dexpilot.yml")

    left_hand_urdf = os.path.join(H2_SHARPA_HAND_URDF_DIR, "left_sharpa_wave.urdf")
    right_hand_urdf = os.path.join(H2_SHARPA_HAND_URDF_DIR, "right_sharpa_wave.urdf")

    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)

    left_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=left_yaml_path,
            hand_urdf=left_hand_urdf,
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
            hand_urdf=right_hand_urdf,
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
class H2TeleopActionsCfg:
    """58-D PinkIK action for OpenXR teleop with Sharpa Wave hands."""

    pink_ik_cfg: ActionTermCfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        hand_joint_names=H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER,
        preserve_order=True,
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=44,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="left_wrist_yaw_link",
                    position_cost=1.0,
                    orientation_cost=0.5,
                ),
                FrameTaskCfg(
                    frame="right_wrist_yaw_link",
                    position_cost=1.0,
                    orientation_cost=0.5,
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
class H2SpreadTableclothTeleopEnvCfg(H2SpreadTableclothEnvCfg):
    """OpenXR teleop variant: inherits H2 scene/MDP, swaps in 58D PinkIK action."""

    actions: H2TeleopActionsCfg = H2TeleopActionsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.0, 0.0, -0.70711, 0.70711),
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 300.0

        # Ensure PinkIK URDF + Sharpa hand URDFs + DexPilot yamls are on disk
        # (mirrored from Nucleus alongside H2_SHARPA_USD_PATH) before PinkIK /
        # the teleop pipeline tries to open them.
        ensure_h2_sharpa_assets()

        self.actions.pink_ik_cfg.controller.urdf_path = H2_SHARPA_URDF_PATH
        self.actions.pink_ik_cfg.controller.mesh_path = os.path.dirname(H2_SHARPA_URDF_PATH)

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_h2_sharpa_tablecloth_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
