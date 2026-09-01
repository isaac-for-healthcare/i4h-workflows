# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU phantom used to validate the fluoroscopy sensor contract."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .catheter import CatheterState
from .fluoroscopy_guidance import draw_catheter_guidance


class SyntheticFluoroscopyRenderer:
    """Render an analytic attenuation phantom and projected catheter polyline.

    This is deliberately not a clinical DRR model. It gives CI and an initial
    Isaac scene a deterministic image source while the Slang backend is adapted
    behind the same runtime contract.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        world_bounds_m: tuple[float, float, float, float] = (-0.35, 0.35, -0.30, 0.30),
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        x_min, x_max, y_min, y_max = world_bounds_m
        if not x_min < x_max or not y_min < y_max:
            raise ValueError("world_bounds_m must be (x_min, x_max, y_min, y_max)")
        self.width = int(width)
        self.height = int(height)
        self.world_bounds_m = tuple(float(value) for value in world_bounds_m)
        x = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        body = 0.65 * np.exp(-((xx / 0.86) ** 2 + (yy / 0.96) ** 2) * 2.4)
        spine = 0.85 * np.exp(-((xx / 0.13) ** 2 + ((yy + 0.04) / 0.58) ** 2) * 3.0)
        left_pelvis = 0.28 * np.exp(-(((xx + 0.37) / 0.26) ** 2 + ((yy - 0.30) / 0.22) ** 2) * 2.2)
        right_pelvis = 0.28 * np.exp(-(((xx - 0.37) / 0.26) ** 2 + ((yy - 0.30) / 0.22) ** 2) * 2.2)
        self._attenuation = (body + spine + left_pelvis + right_pelvis).astype(np.float32)

    def render(self, catheter: CatheterState, carm: object | None = None) -> dict[str, np.ndarray]:
        del carm
        attenuation = np.repeat(self._attenuation[None, ..., None], catheter.num_envs, axis=0)
        transmission = np.exp(-attenuation[..., 0])
        rgb = np.repeat(np.rint(255.0 * transmission)[..., None], 3, axis=-1).astype(np.uint8)
        guidance = rgb.copy()
        for env_index, valid_count in enumerate(catheter.valid_nodes):
            if valid_count >= 2:
                pixels = self._project_catheter(catheter.positions_world_m[env_index, :valid_count])
                self._draw_catheter(rgb[env_index], pixels)
                guidance[env_index] = draw_catheter_guidance(rgb[env_index], pixels)
        return {
            "rgb": rgb,
            "guidance": guidance,
            "dsa": rgb.copy(),
            "dsa_guidance": guidance.copy(),
            "attenuation": attenuation,
        }

    def _project_catheter(self, points_world_m: np.ndarray) -> np.ndarray:
        # The zero-angle AP C-arm projects along world Z, so its detector plane
        # is world X/Y. The patient-backed renderer derives this plane from CArmState.
        x_min, x_max, y_min, y_max = self.world_bounds_m
        u = (points_world_m[:, 0] - x_min) / (x_max - x_min) * (self.width - 1)
        v = (1.0 - (points_world_m[:, 1] - y_min) / (y_max - y_min)) * (self.height - 1)
        return np.stack((u, v), axis=-1)

    def _draw_catheter(self, image: np.ndarray, pixels: np.ndarray) -> None:
        for start, end in pairwise(pixels):
            samples = max(2, int(np.ceil(np.linalg.norm(end - start))) + 1)
            segment = np.linspace(start, end, samples)
            xs = np.clip(np.rint(segment[:, 0]).astype(np.int32), 0, self.width - 1)
            ys = np.clip(np.rint(segment[:, 1]).astype(np.int32), 0, self.height - 1)
            for offset_x, offset_y in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                image[np.clip(ys + offset_y, 0, self.height - 1), np.clip(xs + offset_x, 0, self.width - 1)] = 12
