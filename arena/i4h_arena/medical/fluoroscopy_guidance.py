# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Presentation-only catheter guidance overlays for fluoroscopy frames."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def draw_catheter_guidance(image: np.ndarray, pixels: np.ndarray, visible: np.ndarray | None = None) -> np.ndarray:
    """Return an RGB frame with a green catheter path and emphasized distal tip.

    The source image is not modified. This is deliberately separate from the
    raw fluoroscopy output so recordings can retain the unannotated sensor
    image while the operator uses a more legible presentation.
    """
    frame = np.asarray(image)
    points = np.asarray(pixels, dtype=np.float64)
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("pixels must have shape (N, 2)")
    if visible is None:
        valid = np.ones(points.shape[0], dtype=bool)
    else:
        valid = np.asarray(visible, dtype=bool)
        if valid.shape != (points.shape[0],):
            raise ValueError("visible must contain one value per pixel point")
    valid &= np.isfinite(points).all(axis=1)

    canvas = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(canvas)
    height, width = frame.shape[:2]
    extent = float(max(height, width))
    line_width = max(3, round(extent / 160.0))
    node_radius = max(3, round(extent / 128.0))
    tip_radius = max(6, round(extent / 85.0))
    green = (96, 255, 64)

    for index in range(1, points.shape[0]):
        if not valid[index - 1] or not valid[index]:
            continue
        start, end = points[index - 1], points[index]
        # Reject numerically exploded projections instead of painting a line
        # across the complete detector when a node passes behind the source.
        if np.linalg.norm(end - start) > 4.0 * extent:
            continue
        draw.line((tuple(start), tuple(end)), fill=green, width=line_width)

    for point, keep in zip(points, valid, strict=True):
        if keep and 0.0 <= point[0] < width and 0.0 <= point[1] < height:
            x, y = float(point[0]), float(point[1])
            draw.ellipse((x - node_radius, y - node_radius, x + node_radius, y + node_radius), fill=green)

    tip_indices = np.flatnonzero(valid)
    if tip_indices.size:
        tip = points[int(tip_indices[-1])]
        if 0.0 <= tip[0] < width and 0.0 <= tip[1] < height:
            x, y = float(tip[0]), float(tip[1])
            draw.ellipse(
                (x - tip_radius - 2, y - tip_radius - 2, x + tip_radius + 2, y + tip_radius + 2),
                fill=(16, 32, 16),
            )
            draw.ellipse((x - tip_radius, y - tip_radius, x + tip_radius, y + tip_radius), fill=green)
            center_radius = max(1, tip_radius // 3)
            draw.ellipse(
                (x - center_radius, y - center_radius, x + center_radius, y + center_radius),
                fill=(245, 255, 245),
            )

    return np.asarray(canvas)
