# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def write_video(path: str | Path, frames: np.ndarray, fps: int = 30) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, list(_to_uint8_rgb(frames)), fps=fps, macro_block_size=1)


def read_video(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    reader = imageio.get_reader(path)
    frames = []
    try:
        for frame in reader:
            frames.append(_to_uint8_rgb(np.asarray(frame)))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"video has no frames: {path}")
    return np.stack(frames, axis=0)


def _to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    single_frame = frames.ndim == 3
    if single_frame:
        frames = frames[None, ...]
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(f"expected RGB/RGBA frames, got shape {frames.shape}")
    frames = frames[..., :3]
    if frames.dtype != np.uint8:
        max_value = 1.0 if np.issubdtype(frames.dtype, np.floating) and float(np.nanmax(frames)) <= 1.0 else 255.0
        frames = np.clip(frames / max_value * 255.0, 0.0, 255.0).astype(np.uint8)
    return frames[0] if single_frame else frames
