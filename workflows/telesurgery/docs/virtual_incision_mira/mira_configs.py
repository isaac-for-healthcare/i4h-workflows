# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


robot_usd_root = "/World/A5_GUI_MODEL/A5_GUI_MODEL_001"

left_arm_base = f"{robot_usd_root}/ASM_L654321"
lj_paths = [
    f"{left_arm_base}/LJ1/LJ1_joint",
    f"{left_arm_base}/ASM_L65432/LJ2/LJ2_joint",
    f"{left_arm_base}/ASM_L65432/ASM_L6543/LJ3/LJ3_joint",
    f"{left_arm_base}/ASM_L65432/ASM_L6543/ASM_L654/LJ4/LJ4_joint",
    f"{left_arm_base}/ASM_L65432/ASM_L6543/ASM_L654/ASM_L65/LJ5/LJ5_joint",
    f"{left_arm_base}/ASM_L65432/ASM_L6543/ASM_L654/ASM_L65/ASM_L61/LJ6/LJ6_1_joint",
]

right_arm_base = f"{robot_usd_root}/ASM_R654321"
rj_paths = [
    f"{right_arm_base}/RJ1/RJ1_joint",
    f"{right_arm_base}/ASM_R65432/RJ2/RJ2_joint",
    f"{right_arm_base}/ASM_R65432/ASM_R6543/RJ3/RJ3_joint",
    f"{right_arm_base}/ASM_R65432/ASM_R6543/ASM_R654/RJ4/RJ4_joint",
    f"{right_arm_base}/ASM_R65432/ASM_R6543/ASM_R654/ASM_R65/RJ5/RJ5_joint",
    f"{right_arm_base}/ASM_R65432/ASM_R6543/ASM_R654/ASM_R65/ASM_R6/RJ6/RJ6_joint",
]

camera_base = f"{robot_usd_root}/C_ASM_6543210"
camera_paths = [
    f"{camera_base}/C_ASM_654321",
    f"{camera_base}/C_ASM_654321/C_ASM_65432",
    f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543",
    f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654",
    f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65",
    f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65/C_ASM_6",
]
camera_prim_path = f"{camera_base}/C_ASM_654321/C_ASM_65432/C_ASM_6543/C_ASM_654/C_ASM_65/C_ASM_6/Camera_Tip/Camera"
max_camera_angle = 70

key_map = {
    "I": (0, 0.1),
    "K": (0, -0.1),
    "J": (1, 0.1),
    "L": (1, -0.1),
    "U": (2, 0.1),
    "O": (2, -0.1),
    # 3: wrist pitch
    "Z": (3, 0.1),
    "X": (3, -0.1),
    # 4: wrist rotation
    "C": ("rotation", 0.1),
    "V": ("rotation", -0.1),
    # 5: gripper
    "B": ("grasp", 10.0),
    "N": ("grasp", -10.0),
}

camera_key_map = {
    "UP": (1, 1.0),
    "DOWN": (1, -1.0),
    "LEFT": (0, -1.0),
    "RIGHT": (0, 1.0),
}

switch_key = "Y"
snapshot_key = "F12"
