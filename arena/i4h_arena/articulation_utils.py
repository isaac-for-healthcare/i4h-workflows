# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Articulation helpers."""

from __future__ import annotations

from collections.abc import Sequence


def joint_indices(robot, joint_names: Sequence[str]) -> list[int]:
    """Resolve every requested joint exactly once and fail on an incomplete match."""
    requested = list(joint_names)
    indices, matched_names = robot.find_joints(requested, preserve_order=True)
    if len(indices) != len(requested):
        missing = [name for name in requested if name not in matched_names]
        raise KeyError(f"joint resolution failed; missing={missing}, matched={matched_names}")
    return [int(index) for index in indices]
