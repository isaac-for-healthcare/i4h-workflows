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

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("organs"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """Reward the robot for reaching the drawer handle using inverse-square law."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    target_pos_w = object.data.root_pos_w
    # apply an offset to the object position.
    target_pos_w = target_pos_w + torch.tensor([0.0, -0.25, 1.0], device=target_pos_w.device)
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(target_pos_w - ee_w, dim=1)

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + object_ee_distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(object_ee_distance <= threshold, 2 * reward, reward)
    # return 1 - torch.tanh(object_ee_distance / std)


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """
    ee_frame_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    # organs_quat = env.scene["organs"].data.root_quat_w
    goal_frame_quat = env.scene["goal_frame"].data.target_quat_w[..., 0, :]

    ee_frame_rot_mat = matrix_from_quat(ee_frame_quat)
    # organ_mat = matrix_from_quat(organs_quat)
    goal_frame_rot_mat = matrix_from_quat(goal_frame_quat)

    # get current x and y direction of the organ
    # organ_x, organ_y = organ_mat[..., 0], organ_mat[..., 1]
    # get the current x and z direction of the goal frame
    goal_frame_x, goal_frame_z = goal_frame_rot_mat[..., 0], goal_frame_rot_mat[..., 2]
    # get current x and z direction of the gripper
    ee_frame_x, ee_frame_z = ee_frame_rot_mat[..., 0], ee_frame_rot_mat[..., 2]

    # make sure gripper aligns with the organ
    # in this case, the z direction of the gripper should be close to the -x direction of the organ
    # and the x direction of the gripper should be close to the -y direction of the organ
    # dot product of z and x should be large
    # align_z = torch.bmm(ee_frame_z.unsqueeze(1), -organ_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # align_x = torch.bmm(ee_frame_x.unsqueeze(1), -organ_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)

    # make sure gripper aligns with the goal frame. they should have the same orientation
    # in this case, the z direction of the gripper should be close to the z direction of the goal frame
    # and the x direction of the gripper should be close to the x direction of the goal frame
    align_z = torch.bmm(ee_frame_z.unsqueeze(1), goal_frame_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_frame_x.unsqueeze(1), goal_frame_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def approach_ee_patient(
    env: ManagerBasedRLEnv, ground_truth_pos_wrt_organ: torch.tensor, threshold: float
) -> torch.Tensor:
    r"""Reward the robot for reaching the patient body using inverse-square law.

    It uses a piecewise function to reward the robot for reaching the scan location on the patient body.

    .. math::

        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{otherwise}
        \end{cases}

    """
    # TODO: check if t the matrix multiplication is correct
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]

    organ_pos = env.scene["organs"].data.target_pos_w[..., 0, :]
    gt_pose_w = torch.mm(organ_pos, ground_truth_pos_wrt_organ)

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(gt_pose_w - ee_tcp_pos, dim=-1, p=2)

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def align_ee_patien(
    env: ManagerBasedRLEnv,
    ground_truth_rot_wrt_organ: torch.tensor,
) -> torch.Tensor:
    """Reward for aligning the end-effector with the patient.

    The reward is based on the alignment of the probe (robot flange) with the scan pose on the patient surface.
    It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the probe (flange) and the -x direction
    of the scan pose and :math:`align_x` is the dot product of the x direction of the probe and the -y
    direction of the scan pose.
    """
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)

    organ_quat = env.scene["organs"].data.target_quat_w[..., 0, :]
    organ_rot_mat = matrix_from_quat(organ_quat)

    gt_pose_mat_wrt_organ = matrix_from_quat(ground_truth_rot_wrt_organ)

    # TODO: check if the matrix multiplication is correct
    gt_pose_mat = torch.mm(organ_rot_mat, gt_pose_mat_wrt_organ)

    # get current x and y direction of the handle
    gt_x, gt_y, _ = gt_pose_mat[..., 0], gt_pose_mat[..., 1], gt_pose_mat[..., 2]
    # get current x and y direction of the gripper
    ee_tcp_x, ee_tcp_y, ee_tcp_z = (
        ee_tcp_rot_mat[..., 0],
        ee_tcp_rot_mat[..., 1],
        ee_tcp_rot_mat[..., 2],
    )

    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -gt_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_y = torch.bmm(ee_tcp_y.unsqueeze(1), -gt_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -gt_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    return 1 / 3 * (torch.sign(align_x) * align_x**2) + (
        torch.sign(align_y) * align_y**2 + (torch.sign(align_z) * align_z**2)
    )
