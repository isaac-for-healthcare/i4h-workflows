# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen pre-optimization implementations for identical-input regression checks."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import torch

from i4h_arena.medical.catheter import CatheterState
from i4h_arena.medical.fluoroscopy_guidance import draw_catheter_guidance
from i4h_arena.medical.slang_fluoroscopy import ProjectionGeometry
from i4h_common.types import CameraFrame


def frame_rgba(frame: CameraFrame) -> np.ndarray:
    """Decode an RGB8 scene frame into the contiguous RGBA layout Kit expects."""
    if frame.encoding != "rgb8":
        raise ValueError(f"sensor image window requires rgb8, got {frame.encoding!r}")
    expected = frame.height * frame.width * 3
    rgb = np.frombuffer(frame.data, dtype=np.uint8)
    if rgb.size != expected:
        raise ValueError(f"sensor frame contains {rgb.size} bytes; expected {expected}")
    rgb = rgb.reshape(frame.height, frame.width, 3)
    rgba = np.empty((frame.height, frame.width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    return rgba


def _draw_catheter(self, image: np.ndarray, pixels: np.ndarray) -> None:
    for start, end in pairwise(pixels):
        samples = max(2, int(np.ceil(np.linalg.norm(end - start))) + 1)
        segment = np.linspace(start, end, samples)
        xs = np.clip(np.rint(segment[:, 0]).astype(np.int32), 0, self.width - 1)
        ys = np.clip(np.rint(segment[:, 1]).astype(np.int32), 0, self.height - 1)
        for offset_x, offset_y in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            image[np.clip(ys + offset_y, 0, self.height - 1), np.clip(xs + offset_x, 0, self.width - 1)] = 12


def _draw_catheter_overlays(
    self,
    catheter: CatheterState,
    projections: list[ProjectionGeometry],
    *,
    annotate: bool,
    frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the guidance overlays, and the catheter itself when it is not in the beam, on the host.

    These are vector drawings with no batched GPU equivalent, so the frames that carry them make
    one round trip together. ``rgb`` only makes it when the catheter is an annotation.
    """
    drawn = [index for index in range(catheter.num_envs) if int(catheter.valid_nodes[index]) >= 2]
    if not drawn:
        return frames
    rgb, guidance, dsa_rgb, dsa_guidance = frames
    host_rgb = rgb.cpu().numpy() if annotate else None
    host_guidance = guidance.cpu().numpy()
    host_dsa = dsa_rgb.cpu().numpy()
    host_dsa_guidance = dsa_guidance.cpu().numpy()
    for env_index in drawn:
        valid = int(catheter.valid_nodes[env_index])
        pixels, visible = self._project_catheter(catheter.positions_world_m[env_index, :valid], projections[env_index])
        if annotate:
            # Without beam compositing the catheter exists only as an annotation.
            self._draw_catheter(host_rgb[env_index], pixels, visible)
            self._draw_catheter(host_guidance[env_index], pixels, visible)
            self._draw_catheter(host_dsa[env_index], pixels, visible)
        host_guidance[env_index] = draw_catheter_guidance(host_guidance[env_index], pixels, visible)
        host_dsa_guidance[env_index] = draw_catheter_guidance(host_dsa[env_index], pixels, visible)

    device = rgb.device
    return (
        torch.from_numpy(host_rgb).to(device) if annotate else rgb,
        torch.from_numpy(host_guidance).to(device),
        torch.from_numpy(host_dsa).to(device) if annotate else dsa_rgb,
        torch.from_numpy(host_dsa_guidance).to(device),
    )
