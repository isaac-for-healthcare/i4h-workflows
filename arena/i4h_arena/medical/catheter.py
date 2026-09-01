# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime boundary between catheter mechanics and image formation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class CatheterState:
    """One batched catheter polyline in Isaac world coordinates.

    ``positions_world_m`` is shaped ``(num_envs, num_nodes, 3)``. ``valid_nodes``
    allows a fixed-width buffer to represent catheters with different active
    lengths without reallocating the renderer input.
    """

    positions_world_m: np.ndarray
    valid_nodes: np.ndarray
    radius_m: float = 0.0005

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_world_m, dtype=np.float32)
        valid = np.asarray(self.valid_nodes, dtype=np.int32)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions_world_m must have shape (num_envs, num_nodes, 3)")
        if valid.shape != (positions.shape[0],):
            raise ValueError("valid_nodes must contain one count per environment")
        if np.any(valid < 0) or np.any(valid > positions.shape[1]):
            raise ValueError("valid_nodes entries must be between zero and num_nodes")
        if not np.isfinite(positions).all():
            raise ValueError("positions_world_m must contain only finite values")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        object.__setattr__(self, "positions_world_m", positions)
        object.__setattr__(self, "valid_nodes", valid)

    @property
    def num_envs(self) -> int:
        return int(self.positions_world_m.shape[0])

    @classmethod
    def empty(cls, num_envs: int) -> CatheterState:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        return cls(
            positions_world_m=np.zeros((num_envs, 0, 3), dtype=np.float32),
            valid_nodes=np.zeros(num_envs, dtype=np.int32),
        )


@runtime_checkable
class CatheterStateProvider(Protocol):
    """Physics-independent source of the latest catheter geometry."""

    def snapshot(self, num_envs: int) -> CatheterState:
        """Return the latest world-space catheter state for every environment."""


class StaticCatheterStateProvider:
    """Immutable provider used by tests, phantoms, and fixed validation scenes."""

    def __init__(self, state: CatheterState):
        self._state = state

    def snapshot(self, num_envs: int) -> CatheterState:
        if self._state.num_envs == num_envs:
            return self._state
        if self._state.num_envs != 1:
            raise ValueError(f"provider has {self._state.num_envs} environments, requested {num_envs}")
        return CatheterState(
            positions_world_m=np.repeat(self._state.positions_world_m, num_envs, axis=0),
            valid_nodes=np.repeat(self._state.valid_nodes, num_envs),
            radius_m=self._state.radius_m,
        )
