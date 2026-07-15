# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main Fluoroscopy Simulator class.

This module provides the FluoroSimulator class - the primary interface for
generating simulated fluoroscopy (X-ray) images from preprocessed CT volumes.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Sequence

import numpy as np

from .config import SimulatorConfig
from .volume import PreprocessedVolume

if TYPE_CHECKING:
    from .catheter_provider import CatheterProvider
    from .rendering.diffdrr_slang_renderer import CatheterSegmentData


@dataclass
class Pose:
    """6-DOF pose for C-arm positioning.

    Attributes:
        rotation: Euler angles (rx, ry, rz) in radians.
        translation: Translation (tx, ty, tz) in mm.
    """

    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        """Create from dictionary."""
        return cls(
            rotation=tuple(d.get("rotation", (0.0, 0.0, 0.0))),
            translation=tuple(d.get("translation", (0.0, 0.0, 0.0))),
        )


@dataclass
class Frame:
    """A rendered fluoroscopy frame.

    Attributes:
        image: 2D numpy array of pixel values (H, W), float32 in [0, 1].
        pose: The pose at which this frame was rendered.
        frame_idx: Frame index in a sequence (0 for single frames).
        timestamp_ms: Render timestamp in milliseconds.
    """

    image: np.ndarray
    pose: Pose
    frame_idx: int = 0
    timestamp_ms: float = 0.0

    def save(self, path: str | Path) -> None:
        """Save frame to disk as PNG or NPY."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".npy":
            np.save(path, self.image)
        else:
            # Save as PNG
            self._save_png(path)

    def _save_png(self, path: Path) -> None:
        """Save as 8-bit grayscale PNG."""
        img = np.clip(self.image, 0.0, 1.0)
        u8 = (img * 255.0 + 0.5).astype(np.uint8)
        try:
            import matplotlib.pyplot as plt

            plt.imsave(str(path), u8, cmap="gray", vmin=0, vmax=255)
        except ImportError:
            # Fallback to PGM format
            pgm_path = path.with_suffix(".pgm")
            header = f"P5\n{u8.shape[1]} {u8.shape[0]}\n255\n".encode("ascii")
            with pgm_path.open("wb") as f:
                f.write(header)
                f.write(u8.tobytes())


@dataclass
class CineSequence:
    """A sequence of rendered fluoroscopy frames.

    Attributes:
        frames: List of Frame objects.
        fps: Frames per second.
    """

    frames: list[Frame]
    fps: float = 30.0

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Frame:
        return self.frames[idx]

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def save_all(self, output_dir: str | Path, format: str = "png") -> list[Path]:
        """Save all frames to disk.

        Args:
            output_dir: Directory to save frames.
            format: Output format ("png" or "npy").

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for frame in self.frames:
            path = output_dir / f"frame_{frame.frame_idx:04d}.{format}"
            frame.save(path)
            paths.append(path)

        return paths

    def to_numpy(self) -> np.ndarray:
        """Return all frames as a numpy array (N, H, W)."""
        return np.stack([f.image for f in self.frames], axis=0)


@dataclass
class SimulatorMetrics:
    """Performance metrics from the simulator.

    Attributes:
        fps: Average frames per second.
        frame_times_ms: List of individual frame render times.
        gpu_memory_mb: GPU memory usage in MB (if available).
        gpu_utilization: GPU utilization percentage (if available).
        jitter_ms: Frame timing jitter (std dev of frame times).
    """

    fps: float = 0.0
    frame_times_ms: list[float] | None = None
    gpu_memory_mb: float | None = None
    gpu_utilization: float | None = None
    jitter_ms: float = 0.0

    def __post_init__(self):
        if self.frame_times_ms is None:
            self.frame_times_ms = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fps": self.fps,
            "avg_frame_time_ms": np.mean(self.frame_times_ms) if self.frame_times_ms else 0.0,
            "jitter_ms": self.jitter_ms,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_utilization": self.gpu_utilization,
        }


class FluoroSimulator:
    """Main fluoroscopy simulator class.

    This class provides the primary interface for generating simulated X-ray
    images from preprocessed CT volumes using GPU-accelerated Slang DiffDRR rendering.

    Features:
    - Single frame rendering
    - Cine sequence generation
    - Real-time streaming
    - Performance metrics collection
    - Realism post-processing (noise, blur)
    - Differentiable rendering (gradients via Slang autodiff)

    Example:
        >>> from fluorosim import FluoroSimulator, SimulatorConfig, PreprocessedVolume
        >>>
        >>> # Load preprocessed volume
        >>> volume = PreprocessedVolume.load("/tmp/fluoro_cache")
        >>>
        >>> # Create simulator
        >>> config = SimulatorConfig()
        >>> simulator = FluoroSimulator(volume, config)
        >>>
        >>> # Render single frame
        >>> frame = simulator.render_frame(rotation=(0, 0, 0), translation=(0, 0, 0))
        >>>
        >>> # Render cine sequence
        >>> poses = [Pose(rotation=(i * 0.01, 0, 0)) for i in range(100)]
        >>> cine = simulator.render_cine(poses, fps=30)
    """

    def __init__(
        self,
        volume: PreprocessedVolume,
        config: SimulatorConfig | None = None,
        catheter_provider: Optional["CatheterProvider"] = None,
    ):
        """Initialize the fluoroscopy simulator.

        Args:
            volume:            Preprocessed CT volume (μ values).
            config:            Simulator configuration. Uses defaults if not provided.
            catheter_provider: Optional :class:`~fluorosim.catheter_provider.CatheterProvider`
                               that supplies catheter geometry each frame.  Can also be
                               set later with :meth:`set_catheter_provider`.
        """
        self._volume = volume
        self._config = config or SimulatorConfig()
        self._renderer = None
        self._metrics = SimulatorMetrics()
        self._frame_times = deque(maxlen=100)  # Rolling window for FPS
        self._catheter_provider: Optional["CatheterProvider"] = catheter_provider

        # Initialize the renderer
        self._init_renderer()

    def _init_renderer(self) -> None:
        """Initialize the Slang DiffDRR rendering backend."""
        cfg = self._config

        try:
            from fluorosim.rendering.diffdrr_slang_renderer import SlangDiffDRRConfig, SlangDiffDRRRenderer

            slang_cfg = SlangDiffDRRConfig(
                det_height_px=cfg.geometry.detector_height_px,
                det_width_px=cfg.geometry.detector_width_px,
                pixel_spacing_mm=cfg.geometry.pixel_spacing_mm,
                source_to_detector_mm=cfg.geometry.source_to_detector_mm,
                source_to_isocenter_mm=cfg.geometry.source_to_isocenter_mm,
                step_mm=cfg.physics.step_mm,
                i0=cfg.physics.i0,
                normalize=cfg.physics.normalize,
                invert=cfg.physics.invert,
            )

            self._renderer = SlangDiffDRRRenderer(
                mu_volume=self._volume.mu_volume,
                spacing_zyx_mm=self._volume.spacing_zyx_mm,
                cfg=slang_cfg,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Slang DiffDRR renderer: {e}\n"
                "Make sure slangpy is installed: pip install slangpy"
            ) from e

    def set_catheter_provider(
        self,
        provider: Optional["CatheterProvider"],
    ) -> None:
        """Attach (or detach) a catheter provider.

        The provider is called once per rendered frame (or per environment in
        batched mode) to obtain up-to-date :class:`CatheterSegmentData`.
        Passing ``None`` disables catheter compositing.

        Args:
            provider: Any object implementing
                :class:`~fluorosim.catheter_provider.CatheterProvider`, or
                ``None`` to clear.

        Example::

            from fluorosim.catheter_provider import SolverCatheterAdapter
            adapter = SolverCatheterAdapter(solver, radii=0.5, mu_values=1.5, scale=1000.0)
            sim.set_catheter_provider(adapter)
        """
        self._catheter_provider = provider

    def _resolve_catheter(
        self,
        catheter: Optional["CatheterSegmentData"],
        env_idx: int = 0,
    ) -> Optional["CatheterSegmentData"]:
        """Return explicit catheter data, falling back to the registered provider."""
        if catheter is not None:
            return catheter
        if self._catheter_provider is not None:
            return self._catheter_provider.get_catheter_segments(env_idx)
        return None

    def render_frame(
        self,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        pose: Pose | None = None,
        catheter: Optional["CatheterSegmentData"] = None,
    ) -> Frame:
        """Render a single fluoroscopy frame.

        Args:
            rotation:  Euler angles (rx, ry, rz) in radians.
            translation: Translation (tx, ty, tz) in mm.
            pose:      Alternative to rotation/translation. If provided, overrides them.
            catheter:  Optional explicit :class:`~fluorosim.rendering.diffdrr_slang_renderer.CatheterSegmentData`.
                       If ``None`` and a catheter provider has been registered via
                       :meth:`set_catheter_provider`, the provider is queried automatically.

        Returns:
            Rendered :class:`Frame` object.
        """
        if pose is not None:
            rotation = pose.rotation
            translation = pose.translation

        start_time = time.perf_counter()

        # Resolve catheter: explicit arg → registered provider → None
        active_catheter = self._resolve_catheter(catheter)

        # Render using Slang DiffDRR (fused Beer-Lambert if catheter is present)
        if active_catheter is not None:
            image = self._renderer.render_with_catheter(rotation, translation, active_catheter)
        else:
            image = self._renderer.render(rotation, translation)

        # Apply realism if enabled
        if self._config.realism.enabled:
            image = self._apply_realism(image)

        # Record timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._frame_times.append(elapsed_ms)

        # Create frame
        frame = Frame(
            image=image,
            pose=Pose(rotation=rotation, translation=translation),
            frame_idx=0,
            timestamp_ms=elapsed_ms,
        )

        # Save if configured
        if self._config.output.save_to_disk and self._config.output.output_dir:
            output_dir = Path(self._config.output.output_dir)
            frame.save(output_dir / f"frame_0000.{self._config.output.format}")

        return frame

    def update_volume(self, mu_volume: np.ndarray) -> None:
        """Replace the μ volume on the GPU without re-creating the renderer.

        Useful for dynamic simulations where the volume changes between frames
        (e.g., contrast propagation, catheter advancement).

        Args:
            mu_volume: 3D float32 array (Z, Y, X), must match the original shape.
        """
        self._renderer.update_volume(mu_volume)

    def render_cine(
        self,
        poses: Sequence[Pose] | Sequence[dict],
        fps: float = 30.0,
        progress: bool = True,
        volume_callback: Callable[[int], np.ndarray | None] | None = None,
        catheter_callback: Callable[[int], Optional["CatheterSegmentData"]] | None = None,
    ) -> CineSequence:
        """Render a cine (movie) sequence of fluoroscopy frames.

        Args:
            poses:             Sequence of Pose objects or dicts with rotation/translation.
            fps:               Target frames per second (for metadata).
            progress:          If True, print progress.
            volume_callback:   Optional callable ``f(frame_idx) -> mu_volume | None``.
                               Called before each frame. If it returns a 3D array, the
                               renderer's μ volume is updated. Useful for contrast
                               propagation or per-frame volume mutation.
            catheter_callback: Optional callable ``f(frame_idx) -> CatheterSegmentData | None``.
                               If provided it takes precedence over the registered provider.
                               Return ``None`` to skip catheter compositing for that frame.

        Returns:
            :class:`CineSequence` containing all rendered frames.
        """
        frames = []
        n_frames = len(poses)

        for i, pose_data in enumerate(poses):
            if isinstance(pose_data, Pose):
                pose = pose_data
            else:
                pose = Pose.from_dict(pose_data)

            start_time = time.perf_counter()

            # Per-frame volume update
            if volume_callback is not None:
                new_volume = volume_callback(i)
                if new_volume is not None:
                    self._renderer.update_volume(new_volume)

            # Resolve catheter: explicit callback → registered provider → None
            if catheter_callback is not None:
                active_catheter = catheter_callback(i)
            else:
                active_catheter = self._resolve_catheter(None)

            # Render frame using Slang DiffDRR (fused Beer-Lambert if catheter is present)
            if active_catheter is not None:
                image = self._renderer.render_with_catheter(pose.rotation, pose.translation, active_catheter)
            else:
                image = self._renderer.render(pose.rotation, pose.translation)

            # Apply realism
            if self._config.realism.enabled:
                image = self._apply_realism(image, seed_offset=i)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._frame_times.append(elapsed_ms)

            frames.append(
                Frame(
                    image=image,
                    pose=pose,
                    frame_idx=i,
                    timestamp_ms=elapsed_ms,
                )
            )

            if progress and (i + 1) % 10 == 0:
                current_fps = 1000.0 / np.mean(list(self._frame_times)[-10:])
                print(f"[FluoroSimulator] Frame {i + 1}/{n_frames} ({current_fps:.1f} FPS)")

        cine = CineSequence(frames=frames, fps=fps)

        # Save if configured
        if self._config.output.save_to_disk and self._config.output.output_dir:
            cine.save_all(self._config.output.output_dir, self._config.output.format)

        return cine

    def stream(
        self,
        pose_generator: Iterator[Pose | dict],
        max_frames: int | None = None,
    ) -> Iterator[Frame]:
        """Stream frames in real-time from a pose generator.

        This is useful for interactive applications or RL environments.

        Args:
            pose_generator: Iterator yielding Pose objects or dicts.
            max_frames: Maximum number of frames to generate (None = unlimited).

        Yields:
            Frame objects as they are rendered.

        Example:
            >>> def pose_gen():
            ...     angle = 0.0
            ...     while True:
            ...         yield Pose(rotation=(angle, 0, 0))
            ...         angle += 0.01
            >>>
            >>> for frame in simulator.stream(pose_gen(), max_frames=100):
            ...     process_frame(frame)
        """
        frame_idx = 0

        for pose_data in pose_generator:
            if max_frames is not None and frame_idx >= max_frames:
                break

            if isinstance(pose_data, Pose):
                pose = pose_data
            else:
                pose = Pose.from_dict(pose_data)

            frame = self.render_frame(pose=pose)
            frame.frame_idx = frame_idx

            yield frame
            frame_idx += 1

    def get_metrics(self) -> SimulatorMetrics:
        """Get current performance metrics.

        Returns:
            SimulatorMetrics with FPS, frame times, GPU info.
        """
        frame_times = list(self._frame_times)

        if frame_times:
            avg_time = np.mean(frame_times)
            fps = 1000.0 / avg_time if avg_time > 0 else 0.0
            jitter = np.std(frame_times)
        else:
            fps = 0.0
            jitter = 0.0

        # Try to get GPU metrics
        gpu_memory = None
        gpu_util = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                # torch.cuda.utilization() needs pynvml; degrade gracefully if absent.
                try:
                    gpu_util = float(torch.cuda.utilization())
                except Exception:
                    gpu_util = None
        except ImportError:
            pass

        return SimulatorMetrics(
            fps=fps,
            frame_times_ms=frame_times,
            gpu_memory_mb=gpu_memory,
            gpu_utilization=gpu_util,
            jitter_ms=jitter,
        )

    def _apply_realism(
        self,
        image: np.ndarray,
        seed_offset: int = 0,
    ) -> np.ndarray:
        """Apply realism post-processing to a frame."""
        from fluorosim.rendering.realism import RealismConfig, apply_realism

        cfg = self._config.realism
        base_seed = cfg.seed

        realism_cfg = RealismConfig(
            gain=cfg.gain,
            bias=cfg.bias,
            poisson_photons=cfg.poisson_photons,
            gaussian_sigma=cfg.gaussian_sigma,
            blur_sigma_px=cfg.blur_sigma_px,
            scatter_sigma_px=cfg.scatter_sigma_px,
            gamma=cfg.gamma,
            normalize_output=True,
            seed=None if base_seed is None else base_seed + seed_offset,
        )

        return apply_realism(image, realism_cfg)

    @property
    def config(self) -> SimulatorConfig:
        """Return current configuration."""
        return self._config

    @property
    def volume(self) -> PreprocessedVolume:
        """Return the preprocessed volume."""
        return self._volume

    @property
    def renderer(self):
        """Return the underlying SlangDiffDRRRenderer."""
        return self._renderer

    @property
    def catheter_provider(self) -> Optional["CatheterProvider"]:
        """The currently registered catheter provider, or ``None``."""
        return self._catheter_provider

    def __repr__(self) -> str:
        provider_name = type(self._catheter_provider).__name__ if self._catheter_provider is not None else "None"
        return (
            f"FluoroSimulator(\n"
            f"  volume_shape={self._volume.shape},\n"
            f"  detector=({self._config.geometry.detector_height_px}×"
            f"{self._config.geometry.detector_width_px}),\n"
            f"  catheter_provider={provider_name}\n"
            f")"
        )
