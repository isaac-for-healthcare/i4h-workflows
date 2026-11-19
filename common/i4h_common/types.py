# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain value types shared across the projects.

Batched over environments: the leading axis of any array is ``num_envs``, so a
single-env value is shape ``(1, ...)``. They are numpy rather than torch, so the
projects that use them stay installable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class JointState:
    """Batched joint positions/velocities, shape ``(N, dof)``, plus joint names."""

    pos: np.ndarray
    vel: np.ndarray
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pos = np.atleast_2d(np.asarray(self.pos, dtype=np.float32))
        vel = np.atleast_2d(np.asarray(self.vel, dtype=np.float32))
        if pos.shape != vel.shape:
            raise ValueError(f"joint pos/vel shape mismatch: {pos.shape} vs {vel.shape}")
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "vel", vel)
        object.__setattr__(self, "names", tuple(self.names))

    @property
    def num_envs(self) -> int:
        return int(self.pos.shape[0])

    @property
    def dof(self) -> int:
        return int(self.pos.shape[1])

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown joint {name!r}; have {list(self.names)}") from exc


@dataclass(slots=True)
class CameraFrame:
    """One rendered image, as raw RGB bytes."""

    name: str
    width: int
    height: int
    data: bytes = b""
    encoding: str = "rgb8"

    def to_array(self) -> np.ndarray:
        if self.encoding != "rgb8":
            raise ValueError(f"cannot decode {self.encoding!r} without an image codec")
        return np.frombuffer(self.data, dtype=np.uint8).reshape(self.height, self.width, 3)
