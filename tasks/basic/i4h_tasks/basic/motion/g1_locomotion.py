# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Unitree G1 whole-body locomotion command helpers."""

from __future__ import annotations

import numpy as np

from i4h_engine.task import TickContext

G1_JOINT_WIDTH = 43
G1_WBC_WIDTH = 50
G1_NAVIGATION_SLICE = slice(43, 46)
G1_BASE_HEIGHT_INDEX = 46
G1_TORSO_SLICE = slice(47, 50)


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Extract world yaw from ``wxyz`` quaternions."""
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def heading_error_to_position(
    robot_position: np.ndarray,
    robot_quat_wxyz: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """Signed shortest yaw error from G1's forward axis to a target."""
    delta = np.asarray(target_position)[..., :2] - np.asarray(robot_position)[..., :2]
    target_yaw = np.arctan2(delta[..., 1], delta[..., 0])
    return wrap_angle_rad(target_yaw - yaw_from_quat_wxyz(robot_quat_wxyz))


def set_g1_wbc_command(
    ctx: TickContext,
    *,
    navigation: np.ndarray,
    robot: str = "robot",
    base_height_m: float = 0.75,
) -> None:
    """Hold G1's measured posture while sending ``vx, vy, yaw_rate`` to its WBC."""
    if ctx.act.dof != G1_WBC_WIDTH:
        raise ValueError(f"G1 WBC locomotion requires {G1_WBC_WIDTH} actions, got {ctx.act.dof}")
    joints = ctx.scene.joints(robot)
    if joints.dof != G1_JOINT_WIDTH:
        raise ValueError(f"G1 WBC locomotion requires {G1_JOINT_WIDTH} measured joints, got {joints.dof}")
    navigation = np.asarray(navigation, dtype=np.float32)
    expected_shape = (ctx.num_envs, 3)
    if navigation.shape != expected_shape:
        raise ValueError(f"G1 navigation command must have shape {expected_shape}, got {navigation.shape}")

    command = np.zeros((ctx.num_envs, G1_WBC_WIDTH), dtype=np.float32)
    command[:, :G1_JOINT_WIDTH] = joints.pos
    command[:, G1_NAVIGATION_SLICE] = navigation
    command[:, G1_BASE_HEIGHT_INDEX] = base_height_m
    command[:, G1_TORSO_SLICE] = 0.0
    ctx.act.set_raw_action(command, robot)
