# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A single joint-space target — the common case, spelled directly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from i4h_tasks.basic.motion.keyframes import Frame, Keyframes


class MoveJoints(Keyframes):
    """A single joint-space target. The common case, spelled directly."""

    requires = {"action_space": "joint_position"}

    def __init__(
        self,
        target: Sequence[float],
        *,
        duration_s: float = 0.5,
        relative_to_home: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            [Frame(name or "move", tuple(target), duration_s)],
            relative_to_home=relative_to_home,
            name=name,
            **kwargs,
        )
