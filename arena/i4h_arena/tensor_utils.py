# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor helpers for simulator data."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.utils.warp.proxy_array import ProxyArray


def to_torch(value: torch.Tensor | ProxyArray | wp.array) -> torch.Tensor:
    """Return a zero-copy torch view for a supported simulator tensor type."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, ProxyArray):
        return value.torch
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    raise TypeError(f"expected torch.Tensor, ProxyArray, or warp.array; got {type(value).__name__}")


def quat_from_euler_degrees(euler_xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles in degrees to an XYZW quaternion."""
    from isaaclab.utils.math import quat_from_euler_xyz

    radians = torch.deg2rad(euler_xyz)
    return quat_from_euler_xyz(radians[0], radians[1], radians[2])
