# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np

JointLimitRange = Sequence[tuple[float, float]]


def remap_joint_range(
    joint_pos: np.ndarray,
    source_range: JointLimitRange,
    target_range: JointLimitRange,
) -> np.ndarray:
    values = np.asarray(joint_pos)
    source = np.asarray(source_range, dtype=np.float32)
    target = np.asarray(target_range, dtype=np.float32)
    if source.shape != target.shape:
        raise ValueError(f"source and target joint ranges must match, got {source.shape} and {target.shape}")
    if values.shape[-1] != source.shape[0]:
        raise ValueError(f"expected {source.shape[0]} joint positions, got {values.shape[-1]}")

    source_min, source_max = source[:, 0], source[:, 1]
    target_min, target_max = target[:, 0], target[:, 1]
    return (values - source_min) / (source_max - source_min) * (target_max - target_min) + target_min


def isaaclab_rad_to_lerobot(
    joint_pos: np.ndarray,
    isaaclab_range: JointLimitRange,
    lerobot_range: JointLimitRange,
) -> np.ndarray:
    return remap_joint_range(joint_pos / np.pi * 180.0, isaaclab_range, lerobot_range)


def lerobot_to_isaaclab_rad(
    joint_pos: np.ndarray,
    lerobot_range: JointLimitRange,
    isaaclab_range: JointLimitRange,
) -> np.ndarray:
    return remap_joint_range(joint_pos, lerobot_range, isaaclab_range) / 180.0 * np.pi
