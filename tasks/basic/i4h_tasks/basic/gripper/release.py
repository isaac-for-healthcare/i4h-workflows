# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open the jaw."""

from __future__ import annotations

from i4h_tasks.basic.gripper.set_gripper import SetGripper


class Release(SetGripper):
    """Open the jaw."""

    requires = {"gripper": True}
    postcondition = {"holding": "none"}

    def __init__(self, *, width: float = 0.35, duration_s: float = 0.3, name: str | None = None) -> None:
        super().__init__(width, duration_s=duration_s, name=name)
