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


"""Helpers for mapping assemble trocar HDF5 fields into canonical LeRobot 28-D tensors."""

from __future__ import annotations

import numpy as np

STATE_28_GROUP_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand")

# Canonical 28-D joint order for assemble trocar LeRobot data.
STATE_28_NAMES_ENV_ORDER = [
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
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

# Assemble trocar replay/training env uses JointPositionActionCfg(scale=1, offset=offset_dict),
# so LeRobot action should match the env raw action space, not the post-offset joint target.
STATE_28_RAW_ACTION_FROM_PROCESSED_DELTA = np.zeros(28, dtype=np.float64)
STATE_28_RAW_ACTION_FROM_PROCESSED_DELTA[3] = 0.3
STATE_28_RAW_ACTION_FROM_PROCESSED_DELTA[10] = 0.3

# Runtime-confirmed order of HDF5 processed_actions, i.e. env.scene["robot"].data.joint_names.
RECORDED_ACTION_43_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
)

BODY_JOINT_INDICES = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)
DEX3_JOINT_INDICES = (31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38)

_recorded_action_name_to_idx = {name: i for i, name in enumerate(RECORDED_ACTION_43_JOINT_NAMES)}
ACTION_HDF5_TO_ENV_28 = [_recorded_action_name_to_idx[name] for name in STATE_28_NAMES_ENV_ORDER]

_body_joint_to_col = {joint_id: i for i, joint_id in enumerate(BODY_JOINT_INDICES)}
STATE_28_BODY_COL_LEFT_ARM = [_body_joint_to_col[j] for j in range(15, 22)]
STATE_28_BODY_COL_RIGHT_ARM = [_body_joint_to_col[j] for j in range(22, 29)]

_dex3_col_for_joint = {joint_id: i for i, joint_id in enumerate(DEX3_JOINT_INDICES)}
STATE_28_DEX3_COL_LEFT_HAND = [_dex3_col_for_joint[j] for j in range(29, 36)]
STATE_28_DEX3_COL_RIGHT_HAND = [_dex3_col_for_joint[j] for j in range(36, 43)]


def convert_g1_state_action_to_lerobot_28d(
    state_body: np.ndarray,
    state_dex3: np.ndarray,
    action_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rheo HDF5 obs/action arrays into canonical assemble trocar 28-D state/action."""
    state_parts = [
        state_body[:-1, STATE_28_BODY_COL_LEFT_ARM],
        state_body[:-1, STATE_28_BODY_COL_RIGHT_ARM],
        state_dex3[:-1, STATE_28_DEX3_COL_LEFT_HAND],
        state_dex3[:-1, STATE_28_DEX3_COL_RIGHT_HAND],
    ]
    state = np.concatenate(state_parts, axis=1).astype(np.float64)

    action = action_full[:-1, ACTION_HDF5_TO_ENV_28].astype(np.float64)
    action += STATE_28_RAW_ACTION_FROM_PROCESSED_DELTA
    return state, action
