# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameBundle:
    index: int
    cameras: dict[str, np.ndarray]


@dataclass(frozen=True)
class EpisodeSample:
    episode_id: str
    frames: list[FrameBundle]
    total_frames: int
