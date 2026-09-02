# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint-coordinate conversion shared by policy backends and dataset tools."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

JointLimitRange = Sequence[tuple[float, float]]


def remap_joint_range(
    joint_pos: np.ndarray,
    source_range: JointLimitRange,
    target_range: JointLimitRange,
) -> np.ndarray:
    """Linearly map the last dimension from one calibrated range to another."""
    values = np.asarray(joint_pos)
    source = np.asarray(source_range, dtype=np.float32)
    target = np.asarray(target_range, dtype=np.float32)
    if source.shape != target.shape:
        raise ValueError(f"source and target joint ranges must match, got {source.shape} and {target.shape}")
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"joint ranges must have shape (dof, 2), got {source.shape}")
    if values.shape[-1] != source.shape[0]:
        raise ValueError(f"expected {source.shape[0]} joint positions, got {values.shape[-1]}")

    source_min, source_max = source[:, 0], source[:, 1]
    target_min, target_max = target[:, 0], target[:, 1]
    if np.any(source_max == source_min):
        raise ValueError("source joint ranges must have non-zero width")
    return (values - source_min) / (source_max - source_min) * (target_max - target_min) + target_min


def isaaclab_rad_to_lerobot(
    joint_pos: np.ndarray,
    isaaclab_range: JointLimitRange,
    lerobot_range: JointLimitRange,
) -> np.ndarray:
    """Convert IsaacLab radians to calibrated LeRobot joint coordinates."""
    return remap_joint_range(np.asarray(joint_pos) / np.pi * 180.0, isaaclab_range, lerobot_range)


def lerobot_to_isaaclab_rad(
    joint_pos: np.ndarray,
    lerobot_range: JointLimitRange,
    isaaclab_range: JointLimitRange,
) -> np.ndarray:
    """Convert calibrated LeRobot joint coordinates to IsaacLab radians."""
    return remap_joint_range(joint_pos, lerobot_range, isaaclab_range) / 180.0 * np.pi
