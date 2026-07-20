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

"""Differentiable DRR Renderer using Slang Automatic Differentiation.

This module implements a truly differentiable Digitally Reconstructed Radiograph
(DRR) renderer using Slang's compiler-level automatic differentiation. Unlike
finite-difference approaches, this provides:

1. **Exact Gradients**: Slang's autodiff computes analytical derivatives
2. **GPU Acceleration**: All computation runs on CUDA
3. **PyTorch Integration**: Seamless `torch.autograd.Function` wrapper
4. **Memory Efficient**: No need to store intermediate buffers for finite differences

Architecture:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DiffDRR with Slang Autodiff                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌───────────────────────────────────────────────┐    │
│  │ PyTorch      │     │               Slang Shader                    │    │
│  │ Autograd     │◄────┤  [Differentiable] computePixelIntensity()     │    │
│  │              │     │                                               │    │
│  │ rotation ────┼────►│  ┌─────────────┐   ┌──────────────────────┐  │    │
│  │ translation ─┼────►│  │ Euler→R     │──►│ Ray Generation       │  │    │
│  │              │     │  └─────────────┘   └──────────────────────┘  │    │
│  │              │     │                            │                  │    │
│  │              │     │                            ▼                  │    │
│  │              │     │  ┌──────────────────────────────────────────┐│    │
│  │              │     │  │ Siddon's Ray-March with Trilinear Sample ││    │
│  │              │     │  │    ∫ μ(s) ds = Σ μᵢ · Δs                 ││    │
│  │              │     │  └──────────────────────────────────────────┘│    │
│  │              │     │                            │                  │    │
│  │              │     │                            ▼                  │    │
│  │   ∂L/∂θ  ◄───┼─────┤  ┌──────────────────────────────────────────┐│    │
│  │   ∂L/∂t  ◄───┼─────┤  │ Beer-Lambert: I = I₀·exp(-∫μds)         ││    │
│  │              │     │  │                                          ││    │
│  │   Image  ◄───┼─────┤  │ bwd_diff() → ∂I/∂θ, ∂I/∂t               ││    │
│  └──────────────┘     │  └──────────────────────────────────────────┘│    │
│                       └───────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

Usage:
    >>> from diffdrr_slang_renderer import SlangDiffDRRRenderer, SlangDiffDRRConfig
    >>>
    >>> # Initialize renderer
    >>> renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm)
    >>>
    >>> # Forward pass only
    >>> image = renderer.render(rotation=[0, 0, 0], translation=[0, 0, 0])
    >>>
    >>> # With gradients (PyTorch integration)
    >>> torch_renderer = TorchSlangDiffDRR(mu_volume, spacing_zyx_mm)
    >>> rot = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    >>> trans = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    >>> image = torch_renderer(rot, trans)
    >>> loss = (image - target).pow(2).mean()
    >>> loss.backward()  # Gradients computed via Slang autodiff!
    >>> print(rot.grad, trans.grad)

Requirements:
    - slangpy >= 0.40
    - CUDA-capable GPU
    - PyTorch (optional, for autograd integration)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import slangpy

    SLANG_AVAILABLE = True
except ImportError:
    slangpy = None  # type: ignore
    SLANG_AVAILABLE = False


def _suppress_rhi_output():
    """Context manager to suppress Slang RHI stdout/stderr output.

    The Slang RHI layer logs to both stdout and stderr, and these messages
    (e.g., OptixCache, shader model warnings) cannot be disabled via API.
    This context manager redirects both at the OS level.
    """
    import contextlib
    import io
    import os
    import sys

    @contextlib.contextmanager
    def suppress():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdout_fd = None
        old_stderr_fd = None
        devnull_fd = None
        try:
            # Redirect OS-level file descriptors (for C++ libraries)
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            # Redirect Python-level streams
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            yield
        finally:
            # Restore everything
            if old_stdout_fd is not None:
                os.dup2(old_stdout_fd, 1)
                os.close(old_stdout_fd)
            if old_stderr_fd is not None:
                os.dup2(old_stderr_fd, 2)
                os.close(old_stderr_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return suppress()


# Path to the Slang shader file
_SLANG_SHADER_PATH = Path(__file__).parent / "diffdrr_slang.slang"


@dataclass(frozen=True)
class SlangDiffDRRConfig:
    """Configuration for Slang-based differentiable DRR renderer.

    Attributes:
        det_height_px: Detector height in pixels.
        det_width_px: Detector width in pixels.
        pixel_spacing_mm: Pixel pitch on detector (mm).
        source_to_detector_mm: Source-to-detector distance (mm).
        source_to_isocenter_mm: Source-to-isocenter distance (mm).
        step_mm: Ray-marching step size (mm). Smaller = more accurate but slower.
        i0: Unattenuated X-ray intensity.
        normalize: If True, normalize output to [0, 1].
        invert: If True, invert so bone=white, air=black (clinical X-ray convention).
        eps: Numerical stability constant.
    """

    det_height_px: int = 512
    det_width_px: int = 512
    pixel_spacing_mm: float = 0.5
    source_to_detector_mm: float = 1020.0
    source_to_isocenter_mm: float = 510.0
    step_mm: float = 0.5
    i0: float = 1.0
    normalize: bool = True
    invert: bool = True  # Clinical convention: bone=white, air=black
    eps: float = 1e-8


@dataclass
class CatheterSegmentData:
    """Catheter segment geometry for Beer-Lambert compositing in the Slang shader.

    Each segment is a capped cylinder defined by two 3D endpoints, a radius,
    and an effective linear attenuation coefficient (mu in mm^-1).

    Typical mu values:
        - Platinum markers: ~5.0 mm^-1
        - Tungsten markers:  ~3.0 mm^-1
        - Nitinol shaft:     ~0.8 mm^-1
        - Polymer tip:       ~0.15 mm^-1

    Attributes:
        positions: (N, 3) float32 array of node positions in mm (world frame).
        radii: (N-1,) or scalar — cylinder radius per segment in mm.
        mu_values: (N-1,) or scalar — attenuation coefficient per segment in mm^-1.
    """

    positions: np.ndarray
    radii: Union[np.ndarray, float] = 0.5
    mu_values: Union[np.ndarray, float] = 1.0

    def to_structured_array(self) -> np.ndarray:
        """Convert to the flat structured array expected by the Slang shader.

        Returns:
            Structured numpy array with dtype matching CatheterSegment:
            each element has (p0[3], p1[3], radius, mu) = 8 floats.
        """
        n_nodes = self.positions.shape[0]
        n_segs = n_nodes - 1
        if n_segs <= 0:
            return np.zeros((0, 8), dtype=np.float32)

        pos = self.positions.astype(np.float32)
        radii = np.broadcast_to(np.atleast_1d(np.asarray(self.radii, dtype=np.float32)), (n_segs,))
        mus = np.broadcast_to(np.atleast_1d(np.asarray(self.mu_values, dtype=np.float32)), (n_segs,))

        buf = np.zeros((n_segs, 8), dtype=np.float32)
        buf[:, 0:3] = pos[:-1]  # p0
        buf[:, 3:6] = pos[1:]  # p1
        buf[:, 6] = radii  # radius
        buf[:, 7] = mus  # mu
        return buf


class SlangDiffDRRRenderer:
    """Differentiable DRR Renderer using Slang's Automatic Differentiation.

    Supports both single-environment (legacy) and multi-environment (batched) rendering.

    Single-env usage (backward-compatible):
        >>> renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm)
        >>> image = renderer.render(rotation=[0, 0, 0], translation=[0, 0, 0])

    Multi-env usage:
        >>> renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm, num_envs=4)
        >>> rotations    = np.zeros((4, 3), dtype=np.float32)
        >>> translations = np.zeros((4, 3), dtype=np.float32)
        >>> images = renderer.render_batch(rotations, translations)  # (4, H, W)

    Multi-env with per-env catheters:
        >>> images = renderer.render_batch_with_catheter(rotations, translations, catheters)
        where catheters is a list of CatheterSegmentData (one per env, or None).

    Key Features:
    - True autodiff (not finite differences)
    - GPU-accelerated via CUDA
    - Siddon-style ray marching for accurate line integrals
    - Multi-environment: one dispatch (W, H, N) instead of N serial dispatches
    - Compatible with PyTorch's autograd system
    """

    def __init__(
        self,
        mu_volume: np.ndarray,
        spacing_zyx_mm: tuple[float, float, float],
        cfg: SlangDiffDRRConfig = SlangDiffDRRConfig(),
        num_envs: int = 1,
    ):
        """Initialize the Slang differentiable DRR renderer.

        Args:
            mu_volume: 3D numpy array (Z, Y, X) of linear attenuation coefficients (mm^-1).
            spacing_zyx_mm: Voxel spacing in (Z, Y, X) order, in mm.
            cfg: Renderer configuration.
            num_envs: Number of parallel environments. 1 = single-env (default, backward-
                compatible). >1 enables batched rendering via render_batch() / render_batch_with_catheter().

        Raises:
            RuntimeError: If Slang/slangpy is not available.
            ValueError: If mu_volume is not 3D or num_envs < 1.
        """
        if not SLANG_AVAILABLE:
            raise RuntimeError("Slang is not available. Install with: pip install slangpy\n" "Requires slangpy >= 0.40")

        if mu_volume.ndim != 3:
            raise ValueError(f"Expected mu_volume to be 3D; got shape={mu_volume.shape}")
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        self._cfg = cfg
        self._num_envs = num_envs
        self._spacing_zyx = spacing_zyx_mm
        self._vol_shape_zyx = mu_volume.shape

        # Convert spacing from ZYX to XYZ
        sz, sy, sx = spacing_zyx_mm
        self._spacing_xyz = (sx, sy, sz)

        # Store volume as contiguous float32
        self._mu_volume = np.ascontiguousarray(mu_volume.astype(np.float32))

        # Initialize Slang
        self._init_slang()

    def _init_slang(self):
        """Initialize Slang device and load the differentiable shader."""
        print("[SlangDiffDRR] Initializing...")

        # Suppress harmless Slang/RHI warnings during initialization
        # These include: warning 41012 (profile auto-detection), sm_6_0, OptixCache info
        with _suppress_rhi_output():
            # Create CUDA device
            self._device = slangpy.create_device(slangpy.DeviceType.cuda, enable_hot_reload=False)

            # Load the Slang module with autodiff enabled
            if not _SLANG_SHADER_PATH.exists():
                raise FileNotFoundError(f"Slang shader not found: {_SLANG_SHADER_PATH}")

            self._module = slangpy.Module.load_from_file(
                self._device,
                str(_SLANG_SHADER_PATH),
                options={"defines": {"ENABLE_AUTODIFF": "1"}},
            )

        # API changed: adapter_info.name -> info.adapter_name in slangpy 0.40+
        device_name = getattr(self._device.info, "adapter_name", "Unknown GPU")
        print(f"  Device: {device_name}")
        print(f"  Module: {_SLANG_SHADER_PATH.name}")

        # Get entry points for differentiable rendering
        self._forward_fn = self._module.find_function("renderDRR_forward")
        self._backward_fn = self._module.find_function("renderDRR_backward")
        self._forward_catheter_fn = self._module.find_function("renderDRR_withCatheter_forward")
        self._forward_batched_fn = self._module.find_function("renderDRR_forward_batched")
        print("  Functions loaded: forward, backward, withCatheter, forward_batched")

        # Create GPU resources
        self._create_resources()
        self._catheter_buffer = None
        self._catheter_num_segments = 0
        # Batched-mode GPU buffers (lazily allocated on first render_batch call)
        self._batch_output_buf = None
        self._batch_poses_buf = None
        self._batch_catheter_flat_buf = None
        self._batch_catheter_offsets_buf = None
        self._batch_catheter_counts_buf = None
        self._batch_empty_catheter_buf = None  # 1-element placeholder
        self._batch_empty_offsets_buf = None
        self._batch_empty_counts_buf = None
        print(f"[SlangDiffDRR] Ready (num_envs={self._num_envs}, Differentiable mode enabled)")

    def _create_resources(self):
        """Create GPU textures and buffers."""
        z, y, x = self._vol_shape_zyx
        cfg = self._cfg
        self._num_voxels = x * y * z

        # Volume texture (3D)
        self._mu_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_3d,
            format=slangpy.Format.r32_float,
            width=x,
            height=y,
            depth=z,
            usage=slangpy.TextureUsage.shader_resource,
            data=self._mu_volume,
        )
        print(f"  Volume texture: {x}x{y}x{z}")

        # Sampler with trilinear interpolation
        sampler_desc = slangpy.SamplerDesc(
            {
                "min_filter": slangpy.TextureFilteringMode.linear,
                "mag_filter": slangpy.TextureFilteringMode.linear,
                "mip_filter": slangpy.TextureFilteringMode.linear,
                "address_u": slangpy.TextureAddressingMode.clamp_to_edge,
                "address_v": slangpy.TextureAddressingMode.clamp_to_edge,
                "address_w": slangpy.TextureAddressingMode.clamp_to_edge,
            }
        )
        self._sampler = self._device.create_sampler(sampler_desc)

        # Output image texture (2D)
        self._output_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.r32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
        )

        # Gradient output texture (for upstream gradient ∂L/∂I)
        self._grad_output_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.r32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
        )

        print(f"  Output: {cfg.det_width_px}x{cfg.det_height_px}")

        # Per-pixel gradient textures (kept for compatibility)
        self._grad_rotation_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.rgba32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
        )

        self._grad_translation_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.rgba32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource | slangpy.TextureUsage.unordered_access,
        )
        print(f"  Gradient textures: {cfg.det_width_px}x{cfg.det_height_px}")

    # ------------------------------------------------------------------
    # Batched (multi-env) resource helpers
    # ------------------------------------------------------------------

    def _ensure_batch_buffers(self) -> None:
        """Lazily allocate GPU buffers shared across all render_batch calls."""
        cfg = self._cfg
        N = self._num_envs
        W = cfg.det_width_px
        H = cfg.det_height_px

        if self._batch_output_buf is None:
            # Flat output: N * H * W float32 values, unordered-access for shader writes
            self._batch_output_buf = self._device.create_buffer(
                element_count=N * H * W,
                struct_size=4,
                usage=slangpy.BufferUsage.unordered_access,
            )

        # Placeholder catheter buffers (used when an env has no catheter).
        # One dummy CatheterSegment (32 bytes = 8 floats) and zero offset/count.
        if self._batch_empty_catheter_buf is None:
            dummy_seg = np.zeros((1, 8), dtype=np.float32)
            self._batch_empty_catheter_buf = self._device.create_buffer(
                element_count=1,
                struct_size=32,
                usage=slangpy.BufferUsage.shader_resource,
                data=dummy_seg,
            )
            zero_int = np.zeros(N, dtype=np.int32)
            self._batch_empty_offsets_buf = self._device.create_buffer(
                element_count=N,
                struct_size=4,
                usage=slangpy.BufferUsage.shader_resource,
                data=zero_int,
            )
            self._batch_empty_counts_buf = self._device.create_buffer(
                element_count=N,
                struct_size=4,
                usage=slangpy.BufferUsage.shader_resource,
                data=zero_int,
            )

    def _upload_poses(self, rotations: np.ndarray, translations: np.ndarray):
        """Upload (N, 3) rotation and translation arrays to a persistent StructuredBuffer<Pose>.

        Pose layout in Slang structured buffer: float3 rotation (12 bytes) +
        float3 translation (12 bytes) = 24 bytes, tightly packed.

        The buffer is allocated once and reused; data is copied in-place via
        a staging numpy array to avoid per-frame CUDA allocation overhead.
        """
        N = rotations.shape[0]
        poses_np = np.empty((N, 6), dtype=np.float32)
        poses_np[:, :3] = rotations.astype(np.float32)
        poses_np[:, 3:] = translations.astype(np.float32)

        # Allocate once; copy in-place every subsequent call.
        # Buffer.size is in bytes; struct_size=24 → element count = size // 24.
        buf_n = (self._batch_poses_buf.size // 24) if self._batch_poses_buf is not None else -1
        if buf_n != N:
            self._batch_poses_buf = self._device.create_buffer(
                element_count=N,
                struct_size=24,
                usage=slangpy.BufferUsage.shader_resource,
                data=poses_np,
            )
        else:
            self._batch_poses_buf.copy_from_numpy(poses_np)

    def _upload_catheters(self, catheters: list) -> tuple:
        """Pack per-env catheter data into a flat StructuredBuffer<CatheterSegment>
        plus offset and count buffers.

        Args:
            catheters: List of CatheterSegmentData (or None) of length num_envs.

        Returns:
            (catheter_buf, offsets_buf, counts_buf)
        """
        N = self._num_envs
        segments_list = []
        offsets = np.zeros(N, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        cursor = 0

        for i, cat in enumerate(catheters):
            if cat is None:
                offsets[i] = 0
                counts[i] = 0
            else:
                seg_arr = cat.to_structured_array()  # (S, 8) float32
                n_segs = seg_arr.shape[0]
                offsets[i] = cursor
                counts[i] = n_segs
                if n_segs > 0:
                    segments_list.append(seg_arr)
                cursor += n_segs

        total_segs = cursor
        if total_segs == 0:
            # Reuse the persistent empty counts buf or update it in-place
            cnt_n = (self._batch_empty_counts_buf.size // 4) if self._batch_empty_counts_buf is not None else -1
            if cnt_n == N:
                self._batch_empty_counts_buf.copy_from_numpy(counts)
            else:
                self._batch_empty_counts_buf = self._device.create_buffer(
                    element_count=N,
                    struct_size=4,
                    usage=slangpy.BufferUsage.shader_resource,
                    data=counts,
                )
            return (
                self._batch_empty_catheter_buf,
                self._batch_empty_offsets_buf,
                self._batch_empty_counts_buf,
            )

        flat_segs = np.vstack(segments_list).astype(np.float32)

        # Segment buffer: reallocate only when total count changes
        seg_n = (self._batch_catheter_flat_buf.size // 32) if self._batch_catheter_flat_buf is not None else -1
        if seg_n != total_segs:
            self._batch_catheter_flat_buf = self._device.create_buffer(
                element_count=total_segs,
                struct_size=32,
                usage=slangpy.BufferUsage.shader_resource,
                data=flat_segs,
            )
        else:
            self._batch_catheter_flat_buf.copy_from_numpy(flat_segs)

        # Offset buffer: reallocate only when N changes
        off_n = (self._batch_catheter_offsets_buf.size // 4) if self._batch_catheter_offsets_buf is not None else -1
        if off_n != N:
            self._batch_catheter_offsets_buf = self._device.create_buffer(
                element_count=N,
                struct_size=4,
                usage=slangpy.BufferUsage.shader_resource,
                data=offsets,
            )
        else:
            self._batch_catheter_offsets_buf.copy_from_numpy(offsets)

        cnt_n = (self._batch_catheter_counts_buf.size // 4) if self._batch_catheter_counts_buf is not None else -1
        if cnt_n != N:
            self._batch_catheter_counts_buf = self._device.create_buffer(
                element_count=N,
                struct_size=4,
                usage=slangpy.BufferUsage.shader_resource,
                data=counts,
            )
        else:
            self._batch_catheter_counts_buf.copy_from_numpy(counts)

        return self._batch_catheter_flat_buf, self._batch_catheter_offsets_buf, self._batch_catheter_counts_buf

    def _readback_batch(self) -> np.ndarray:
        """Read the flat output buffer back and reshape to (N, H, W).

        slangpy StructuredBuffer.to_numpy() returns raw bytes (uint8).
        We reinterpret them as float32 before reshaping.
        """
        cfg = self._cfg
        N, H, W = self._num_envs, cfg.det_height_px, cfg.det_width_px
        raw = self._batch_output_buf.to_numpy()
        # raw may be uint8 bytes — reinterpret as float32
        floats = raw.view(np.float32) if raw.dtype != np.float32 else raw
        return floats.reshape(N, H, W).copy()

    def _normalize_batch(self, images: np.ndarray) -> np.ndarray:
        """Per-env normalize to [0,1] and optionally invert."""
        cfg = self._cfg
        if cfg.normalize:
            # Normalize each env independently
            mn = images.min(axis=(1, 2), keepdims=True)
            mx = images.max(axis=(1, 2), keepdims=True)
            images = (images - mn) / (mx - mn + cfg.eps)
        if cfg.invert:
            images = 1.0 - images
        return images.astype(np.float32)

    def update_volume(self, mu_volume: np.ndarray) -> None:
        """Replace the attenuation volume used by subsequent renders.

        This updates the renderer's 3D mu texture in-place from the caller's
        perspective (a new GPU texture handle is created under the hood).

        Args:
            mu_volume: New 3D attenuation volume in (Z, Y, X) order.

        Raises:
            ValueError: If the input is not 3D or shape-mismatched.
        """
        if mu_volume.ndim != 3:
            raise ValueError(f"Expected mu_volume to be 3D; got shape={mu_volume.shape}")
        if tuple(mu_volume.shape) != tuple(self._vol_shape_zyx):
            raise ValueError(f"Volume shape mismatch: expected {self._vol_shape_zyx}, got {tuple(mu_volume.shape)}")

        self._mu_volume = np.ascontiguousarray(mu_volume.astype(np.float32))
        z, y, x = self._vol_shape_zyx
        self._mu_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_3d,
            format=slangpy.Format.r32_float,
            width=x,
            height=y,
            depth=z,
            usage=slangpy.TextureUsage.shader_resource,
            data=self._mu_volume,
        )

    # ------------------------------------------------------------------
    # Public multi-env API
    # ------------------------------------------------------------------

    def render_batch(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
    ) -> np.ndarray:
        """Render N DRR images in a single GPU dispatch (no catheter).

        Args:
            rotations:    (N, 3) float32 — Euler angles (rx, ry, rz) in radians per env.
            translations: (N, 3) float32 — Translation (tx, ty, tz) in mm per env.

        Returns:
            (N, H, W) float32 numpy array of rendered images.
        """
        rotations = np.asarray(rotations, dtype=np.float32).reshape(-1, 3)
        translations = np.asarray(translations, dtype=np.float32).reshape(-1, 3)
        N = rotations.shape[0]
        if N != self._num_envs:
            raise ValueError(f"Expected {self._num_envs} poses, got {N}")

        self._ensure_batch_buffers()
        self._upload_poses(rotations, translations)

        cfg = self._cfg
        _, carm_params, _ = self._build_params((0, 0, 0), (0, 0, 0))
        sz, sy, sx = self._spacing_zyx
        z, y, x = self._vol_shape_zyx
        vol_info = {
            "spacing": slangpy.float3(sx, sy, sz),
            "dimensions": slangpy.int3(x, y, z),
            "origin": slangpy.float3(0.0, 0.0, 0.0),
        }

        self._forward_batched_fn.dispatch(
            thread_count=slangpy.uint3(cfg.det_width_px, cfg.det_height_px, N),
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            outputBuffer=self._batch_output_buf,
            volInfo=vol_info,
            carm=carm_params,
            poses=self._batch_poses_buf,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
            numEnvs=int(N),
            catheterSegments=self._batch_empty_catheter_buf,
            catheterOffsets=self._batch_empty_offsets_buf,
            catheterCounts=self._batch_empty_counts_buf,
        )

        images = self._readback_batch()
        return self._normalize_batch(images)

    def render_batch_with_catheter(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        catheters: list,
    ) -> np.ndarray:
        """Render N DRR images with per-env catheter compositing in a single dispatch.

        Args:
            rotations:    (N, 3) float32 — Euler angles per env.
            translations: (N, 3) float32 — Translation per env.
            catheters:    list of N CatheterSegmentData (or None for no catheter).

        Returns:
            (N, H, W) float32 numpy array of rendered images.
        """
        rotations = np.asarray(rotations, dtype=np.float32).reshape(-1, 3)
        translations = np.asarray(translations, dtype=np.float32).reshape(-1, 3)
        N = rotations.shape[0]
        if N != self._num_envs:
            raise ValueError(f"Expected {self._num_envs} poses, got {N}")
        if len(catheters) != N:
            raise ValueError(f"Expected {N} catheters, got {len(catheters)}")

        self._ensure_batch_buffers()
        self._upload_poses(rotations, translations)
        cat_buf, off_buf, cnt_buf = self._upload_catheters(catheters)

        cfg = self._cfg
        _, carm_params, _ = self._build_params((0, 0, 0), (0, 0, 0))
        sz, sy, sx = self._spacing_zyx
        z, y, x = self._vol_shape_zyx
        vol_info = {
            "spacing": slangpy.float3(sx, sy, sz),
            "dimensions": slangpy.int3(x, y, z),
            "origin": slangpy.float3(0.0, 0.0, 0.0),
        }

        self._forward_batched_fn.dispatch(
            thread_count=slangpy.uint3(cfg.det_width_px, cfg.det_height_px, N),
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            outputBuffer=self._batch_output_buf,
            volInfo=vol_info,
            carm=carm_params,
            poses=self._batch_poses_buf,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
            numEnvs=int(N),
            catheterSegments=cat_buf,
            catheterOffsets=off_buf,
            catheterCounts=cnt_buf,
        )

        images = self._readback_batch()
        return self._normalize_batch(images)

    @property
    def num_envs(self) -> int:
        """Number of parallel environments this renderer supports."""
        return self._num_envs

    def _build_params(
        self,
        rotation: tuple[float, float, float],
        translation: tuple[float, float, float],
    ) -> tuple[dict, dict, dict]:
        """Build parameter dictionaries for shader dispatch."""
        sz, sy, sx = self._spacing_zyx
        z, y, x = self._vol_shape_zyx
        cfg = self._cfg

        vol_info = {
            "spacing": slangpy.float3(sx, sy, sz),
            "dimensions": slangpy.int3(x, y, z),
            "origin": slangpy.float3(0.0, 0.0, 0.0),
        }

        carm = {
            "sdd": float(cfg.source_to_detector_mm),
            "sid": float(cfg.source_to_isocenter_mm),
            "detectorSize": slangpy.float2(
                cfg.det_width_px * cfg.pixel_spacing_mm,
                cfg.det_height_px * cfg.pixel_spacing_mm,
            ),
            "detectorPixels": slangpy.int2(cfg.det_width_px, cfg.det_height_px),
            "pixelSpacing": float(cfg.pixel_spacing_mm),
        }

        pose = {
            "rotation": slangpy.float3(*rotation),
            "translation": slangpy.float3(*translation),
        }

        return vol_info, carm, pose

    def render(
        self,
        rotation: Union[tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
        translation: Union[tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Render a DRR image at the specified pose (forward pass only).

        Args:
            rotation: Euler angles (rx, ry, rz) in radians.
            translation: Translation (tx, ty, tz) in mm.

        Returns:
            2D float32 numpy array of shape (H, W).
        """
        # Convert to tuples
        if isinstance(rotation, np.ndarray):
            rotation = tuple(rotation.flatten()[:3].tolist())
        if isinstance(translation, np.ndarray):
            translation = tuple(translation.flatten()[:3].tolist())

        vol_info, carm, pose = self._build_params(rotation, translation)
        cfg = self._cfg

        # Dispatch forward kernel
        thread_count = slangpy.uint3(cfg.det_width_px, cfg.det_height_px, 1)

        self._forward_fn.dispatch(
            thread_count=thread_count,
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            outputImage=self._output_texture,
            volInfo=vol_info,
            carm=carm,
            pose=pose,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
        )

        # Read back result
        image = self._output_texture.to_numpy()
        image = image.reshape(cfg.det_height_px, cfg.det_width_px)

        # Normalize if requested
        if cfg.normalize:
            vmin = float(np.min(image))
            vmax = float(np.max(image))
            image = (image - vmin) / (vmax - vmin + cfg.eps)

        # Invert for clinical X-ray convention (bone=white, air=black)
        if cfg.invert:
            image = 1.0 - image

        return image.astype(np.float32)

    def set_catheter_segments(self, catheter: CatheterSegmentData) -> None:
        """Upload catheter segment geometry to GPU for fused Beer-Lambert compositing.

        After calling this, use ``render_with_catheter()`` to render a DRR with
        the catheter composited depth-correctly in a single ray march.

        Args:
            catheter: Catheter geometry. Call with ``None`` to clear.
        """
        if catheter is None:
            self._catheter_buffer = None
            self._catheter_num_segments = 0
            return

        buf = catheter.to_structured_array()
        n_segs = buf.shape[0]
        if n_segs == 0:
            self._catheter_buffer = None
            self._catheter_num_segments = 0
            return

        self._catheter_buffer = self._device.create_buffer(
            element_count=n_segs,
            struct_size=8 * 4,  # 8 floats × 4 bytes
            usage=slangpy.BufferUsage.shader_resource,
            data=buf,
        )
        self._catheter_num_segments = n_segs

    def render_with_catheter(
        self,
        rotation: Union[tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
        translation: Union[tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
        catheter: Optional[CatheterSegmentData] = None,
    ) -> np.ndarray:
        """Render DRR with catheter Beer-Lambert compositing in a single fused ray march.

        If catheter is provided it is uploaded; otherwise the previously set
        segments are reused. If no segments are available, falls back to the
        standard volume-only DRR.

        Args:
            rotation: Euler angles (rx, ry, rz) in radians.
            translation: Translation (tx, ty, tz) in mm.
            catheter: Optional catheter geometry to upload before rendering.

        Returns:
            2D float32 numpy array of shape (H, W).
        """
        if catheter is not None:
            self.set_catheter_segments(catheter)

        if self._catheter_buffer is None or self._catheter_num_segments == 0:
            return self.render(rotation, translation)

        if isinstance(rotation, np.ndarray):
            rotation = tuple(rotation.flatten()[:3].tolist())
        if isinstance(translation, np.ndarray):
            translation = tuple(translation.flatten()[:3].tolist())

        vol_info, carm, pose = self._build_params(rotation, translation)
        cfg = self._cfg

        thread_count = slangpy.uint3(cfg.det_width_px, cfg.det_height_px, 1)

        self._forward_catheter_fn.dispatch(
            thread_count=thread_count,
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            outputImage=self._output_texture,
            volInfo=vol_info,
            carm=carm,
            pose=pose,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
            catheterSegments=self._catheter_buffer,
            numCatheterSegments=int(self._catheter_num_segments),
        )

        image = self._output_texture.to_numpy()
        image = image.reshape(cfg.det_height_px, cfg.det_width_px)

        if cfg.normalize:
            vmin = float(np.min(image))
            vmax = float(np.max(image))
            image = (image - vmin) / (vmax - vmin + cfg.eps)

        if cfg.invert:
            image = 1.0 - image

        return image.astype(np.float32)

    def render_with_gradients(
        self,
        rotation: Union[tuple[float, float, float], np.ndarray],
        translation: Union[tuple[float, float, float], np.ndarray],
        grad_output: Optional[np.ndarray] = None,
        max_steps: int = 2048,
    ) -> tuple[np.ndarray, dict]:
        """Render DRR and compute gradients via Slang autodiff.

        This uses Slang's automatic differentiation with custom backward derivatives
        for texture sampling to compute exact gradients of the rendered image
        with respect to pose parameters and optionally the volume.

        The implementation follows Slang's autodiff-texture example pattern:
        - Hardware texture sampling for fast forward pass
        - Software trilinear interpolation for backward pass (gradient flow)
        - Atomic fixed-point accumulation for thread-safe gradient updates

        Args:
            rotation: Euler angles (rx, ry, rz) in radians.
            translation: Translation (tx, ty, tz) in mm.
            grad_output: Upstream gradient ∂L/∂I. If None, uses ones (gradient of sum).
            max_steps: Maximum ray-march steps (for differentiable path).

        Returns:
            Tuple of (image, gradients_dict) where:
            - image: Rendered DRR as numpy array (H, W)
            - gradients_dict: {
                'rotation': np.ndarray of shape (3,),
                'translation': np.ndarray of shape (3,),
                'volume': np.ndarray of shape (Z, Y, X) - gradients w.r.t. mu volume
              }

        Example:
            >>> # Compute gradients for 2D/3D registration
            >>> img, grads = renderer.render_with_gradients(
            ...     rotation=[5, 0, 0],
            ...     translation=[10, 0, 0],
            ...     grad_output=2 * (synthetic - target),  # MSE gradient
            ... )
            >>> # Update pose
            >>> rotation -= lr * grads['rotation']
            >>> translation -= lr * grads['translation']
        """
        # Convert to tuples
        if isinstance(rotation, np.ndarray):
            rotation = tuple(rotation.flatten()[:3].tolist())
        if isinstance(translation, np.ndarray):
            translation = tuple(translation.flatten()[:3].tolist())

        cfg = self._cfg
        vol_info, carm, pose = self._build_params(rotation, translation)

        # Prepare upstream gradient
        if grad_output is None:
            grad_output = np.ones((cfg.det_height_px, cfg.det_width_px), dtype=np.float32)
        else:
            grad_output = np.ascontiguousarray(grad_output.astype(np.float32))

        # Upload gradient to texture
        grad_output_texture = self._device.create_texture(
            type=slangpy.TextureType.texture_2d,
            format=slangpy.Format.r32_float,
            width=cfg.det_width_px,
            height=cfg.det_height_px,
            usage=slangpy.TextureUsage.shader_resource,
            data=grad_output,
        )

        thread_count = slangpy.uint3(cfg.det_width_px, cfg.det_height_px, 1)

        # Step 1: Forward pass
        self._forward_fn.dispatch(
            thread_count=thread_count,
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            outputImage=self._output_texture,
            volInfo=vol_info,
            carm=carm,
            pose=pose,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
        )

        # Step 2: Backward pass - compute per-pixel gradients
        self._backward_fn.dispatch(
            thread_count=thread_count,
            muVolume=self._mu_texture,
            volumeSampler=self._sampler,
            gradOutput=grad_output_texture,
            gradRotation=self._grad_rotation_texture,
            gradTranslation=self._grad_translation_texture,
            volInfo=vol_info,
            carm=carm,
            pose=pose,
            stepMM=float(cfg.step_mm),
            i0=float(cfg.i0),
        )

        # Read back results
        image = self._output_texture.to_numpy().reshape(cfg.det_height_px, cfg.det_width_px)

        # Read per-pixel gradients and reduce to total gradients
        grad_rot_pixels = self._grad_rotation_texture.to_numpy()
        grad_rot_pixels = grad_rot_pixels.reshape(cfg.det_height_px, cfg.det_width_px, 4)
        grad_rotation = grad_rot_pixels[:, :, :3].sum(axis=(0, 1))  # Sum over all pixels

        grad_trans_pixels = self._grad_translation_texture.to_numpy()
        grad_trans_pixels = grad_trans_pixels.reshape(cfg.det_height_px, cfg.det_width_px, 4)
        grad_translation = grad_trans_pixels[:, :, :3].sum(axis=(0, 1))  # Sum over all pixels

        # Normalize if requested
        if cfg.normalize:
            vmin = float(np.min(image))
            vmax = float(np.max(image))
            image = (image - vmin) / (vmax - vmin + cfg.eps)

        # Invert for clinical X-ray convention (bone=white, air=black)
        if cfg.invert:
            image = 1.0 - image

        return image.astype(np.float32), {
            "rotation": grad_rotation.astype(np.float32),
            "translation": grad_translation.astype(np.float32),
        }

    @property
    def config(self) -> SlangDiffDRRConfig:
        """Return renderer configuration."""
        return self._cfg

    @property
    def volume_shape_zyx(self) -> tuple[int, int, int]:
        """Return volume shape in ZYX order."""
        return self._vol_shape_zyx

    @property
    def spacing_xyz(self) -> tuple[float, float, float]:
        """Return voxel spacing in XYZ order."""
        return self._spacing_xyz

    def __repr__(self) -> str:
        return (
            f"SlangDiffDRRRenderer(\n"
            f"  volume_shape={self._vol_shape_zyx},\n"
            f"  detector=({self._cfg.det_height_px}×{self._cfg.det_width_px}),\n"
            f"  num_envs={self._num_envs},\n"
            f"  step_mm={self._cfg.step_mm},\n"
            f"  sdd={self._cfg.source_to_detector_mm}mm\n"
            f")"
        )


# =============================================================================
# PyTorch Integration
# =============================================================================

if TORCH_AVAILABLE:

    class SlangDiffDRRFunction(torch.autograd.Function):
        """PyTorch autograd Function for Slang DiffDRR.

        This integrates Slang's automatic differentiation with PyTorch's autograd
        system, enabling end-to-end gradient flow through the DRR renderer.

        Usage:
            >>> renderer = SlangDiffDRRRenderer(volume, spacing)
            >>> rot = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> trans = torch.tensor([0., 0., 0.], requires_grad=True)
            >>>
            >>> image = SlangDiffDRRFunction.apply(renderer, rot, trans)
            >>> loss = (image - target).pow(2).mean()
            >>> loss.backward()  # Gradients computed via Slang autodiff!
            >>>
            >>> print(rot.grad)    # ∂L/∂rotation
            >>> print(trans.grad)  # ∂L/∂translation
        """

        @staticmethod
        def forward(
            ctx,
            renderer: SlangDiffDRRRenderer,
            rotation: torch.Tensor,
            translation: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass: render DRR image.

            Args:
                ctx: Autograd context.
                renderer: SlangDiffDRRRenderer instance.
                rotation: Tensor of shape (3,) with Euler angles in radians.
                translation: Tensor of shape (3,) with translation in mm.

            Returns:
                Rendered image as tensor of shape (H, W).
            """
            ctx.renderer = renderer
            ctx.save_for_backward(rotation, translation)

            # Convert to numpy and render
            rot_np = rotation.detach().cpu().numpy()
            trans_np = translation.detach().cpu().numpy()

            image_np = renderer.render(rot_np, trans_np)

            device = rotation.device
            return torch.from_numpy(image_np).to(device)

        @staticmethod
        def backward(
            ctx,
            grad_output: torch.Tensor,
        ) -> tuple[None, torch.Tensor, torch.Tensor]:
            """Backward pass: compute gradients via Slang autodiff.

            Args:
                ctx: Autograd context with saved tensors.
                grad_output: Gradient of loss w.r.t. output image, shape (H, W).

            Returns:
                Tuple of (None, grad_rotation, grad_translation).
                First None is for the renderer argument.
            """
            rotation, translation = ctx.saved_tensors
            renderer = ctx.renderer

            rot_np = rotation.detach().cpu().numpy()
            trans_np = translation.detach().cpu().numpy()
            grad_out_np = grad_output.detach().cpu().numpy()

            # Use Slang's autodiff for gradient computation
            _, grads = renderer.render_with_gradients(rot_np, trans_np, grad_output=grad_out_np)

            grad_rotation = torch.from_numpy(grads["rotation"]).to(rotation.device)
            grad_translation = torch.from_numpy(grads["translation"]).to(translation.device)

            return None, grad_rotation, grad_translation

    class TorchSlangDiffDRR(torch.nn.Module):
        """PyTorch Module for Slang-based differentiable DRR.

        This wraps SlangDiffDRRRenderer as a torch.nn.Module, enabling:
        - Integration with PyTorch neural network pipelines
        - Compatibility with torch.compile()
        - Easy parameter management

        Example:
            >>> # As part of a registration network
            >>> class RegistrationNet(torch.nn.Module):
            ...     def __init__(self, volume, spacing):
            ...         super().__init__()
            ...         self.drr = TorchSlangDiffDRR(volume, spacing)
            ...         self.pose_net = PoseEstimator()
            ...
            ...     def forward(self, x):
            ...         pose = self.pose_net(x)
            ...         rotation, translation = pose[:, :3], pose[:, 3:]
            ...         return self.drr(rotation[0], translation[0])

            >>> # For direct pose optimization
            >>> drr = TorchSlangDiffDRR(volume, spacing)
            >>> rot = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> trans = torch.tensor([0., 850., 0.], requires_grad=True)
            >>> optimizer = torch.optim.Adam([rot, trans], lr=0.01)
            >>>
            >>> for step in range(100):
            ...     optimizer.zero_grad()
            ...     synthetic = drr(rot, trans)
            ...     loss = -ncc(synthetic, target)  # Negative NCC
            ...     loss.backward()
            ...     optimizer.step()
        """

        def __init__(
            self,
            mu_volume: np.ndarray,
            spacing_zyx_mm: tuple[float, float, float],
            cfg: SlangDiffDRRConfig = SlangDiffDRRConfig(),
            num_envs: int = 1,
        ):
            """Initialize the PyTorch Slang DRR module.

            Args:
                mu_volume: 3D attenuation volume (Z, Y, X).
                spacing_zyx_mm: Voxel spacing in mm.
                cfg: Renderer configuration.
                num_envs: Number of parallel environments (1 = single-env, default).
            """
            super().__init__()
            self._renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm, cfg, num_envs=num_envs)
            self._cfg = cfg

        def forward(
            self,
            rotation: torch.Tensor,
            translation: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass through the differentiable renderer.

            Args:
                rotation: Tensor of shape (3,) or (N, 3) with Euler angles in radians.
                translation: Tensor of shape (3,) or (N, 3) with translation in mm.

            Returns:
                Rendered image(s) as tensor of shape (H, W) or (N, H, W).
            """
            if rotation.dim() == 1:
                return SlangDiffDRRFunction.apply(self._renderer, rotation, translation)
            else:
                # Batch rendering
                batch_size = rotation.shape[0]
                results = []
                for i in range(batch_size):
                    img = SlangDiffDRRFunction.apply(
                        self._renderer,
                        rotation[i],
                        translation[i],
                    )
                    results.append(img)
                return torch.stack(results)

        def render_numpy(
            self,
            rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
            translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ) -> np.ndarray:
            """Render directly to numpy (no gradients).

            Args:
                rotation: Euler angles in radians.
                translation: Translation in mm.

            Returns:
                Rendered image as numpy array.
            """
            return self._renderer.render(rotation, translation)

        def render_batch(
            self,
            rotations: np.ndarray,
            translations: np.ndarray,
        ) -> np.ndarray:
            """Batched multi-env render (numpy, no gradients).

            Args:
                rotations:    (N, 3) float32 Euler angles per env.
                translations: (N, 3) float32 translations per env.

            Returns:
                (N, H, W) float32 numpy array.
            """
            return self._renderer.render_batch(rotations, translations)

        def render_batch_with_catheter(
            self,
            rotations: np.ndarray,
            translations: np.ndarray,
            catheters: list,
        ) -> np.ndarray:
            """Batched multi-env render with per-env catheters (numpy, no gradients).

            Args:
                rotations:    (N, 3) float32 Euler angles per env.
                translations: (N, 3) float32 translations per env.
                catheters:    list of N CatheterSegmentData (or None).

            Returns:
                (N, H, W) float32 numpy array.
            """
            return self._renderer.render_batch_with_catheter(rotations, translations, catheters)

        @property
        def num_envs(self) -> int:
            """Number of parallel environments."""
            return self._renderer.num_envs

        @property
        def config(self) -> SlangDiffDRRConfig:
            """Return renderer configuration."""
            return self._cfg

        @property
        def renderer(self) -> SlangDiffDRRRenderer:
            """Access the underlying SlangDiffDRRRenderer."""
            return self._renderer

        def __repr__(self) -> str:
            return f"TorchSlangDiffDRR({self._renderer})"


# =============================================================================
# Convenience Functions
# =============================================================================


def render_diffdrr_slang(
    mu_volume: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cfg: SlangDiffDRRConfig = SlangDiffDRRConfig(),
) -> np.ndarray:
    """One-shot differentiable DRR rendering using Slang.

    This is a convenience function for quick rendering without explicitly
    creating a renderer instance.

    Args:
        mu_volume: 3D attenuation volume (Z, Y, X) in mm^-1.
        spacing_zyx_mm: Voxel spacing in mm.
        rotation: Euler angles in radians.
        translation: Translation in mm.
        cfg: Renderer configuration.

    Returns:
        Rendered DRR image as numpy array of shape (H, W).
    """
    renderer = SlangDiffDRRRenderer(mu_volume, spacing_zyx_mm, cfg)
    return renderer.render(rotation, translation)


def create_slang_diffdrr_optimizer(
    mu_volume: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    cfg: SlangDiffDRRConfig = SlangDiffDRRConfig(),
    lr: float = 1e-2,
) -> tuple["TorchSlangDiffDRR", torch.Tensor, torch.Tensor, torch.optim.Optimizer]:
    """Create a Slang DiffDRR setup for gradient-based pose optimization.

    This helper function sets up everything needed for 2D/3D registration:
    - Slang DiffDRR module
    - Learnable rotation and translation parameters
    - Adam optimizer

    Args:
        mu_volume: 3D attenuation volume (Z, Y, X).
        spacing_zyx_mm: Voxel spacing in mm.
        initial_rotation: Initial rotation in radians, shape (3,).
        initial_translation: Initial translation in mm, shape (3,).
        cfg: Renderer configuration.
        lr: Learning rate for Adam optimizer.

    Returns:
        Tuple of (drr_module, rotation, translation, optimizer)

    Example:
        >>> drr, rot, trans, opt = create_slang_diffdrr_optimizer(
        ...     volume, spacing,
        ...     initial_rotation=[0, 0, 0],
        ...     initial_translation=[0, 850, 0],
        ...     lr=0.01,
        ... )
        >>>
        >>> for step in range(100):
        ...     opt.zero_grad()
        ...     synthetic = drr(rot, trans)
        ...     loss = mse_loss(synthetic, target)
        ...     loss.backward()
        ...     opt.step()
        ...     print(f"Step {step}: loss={loss.item():.4f}")
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for optimizer creation")

    drr = TorchSlangDiffDRR(mu_volume, spacing_zyx_mm, cfg)

    rotation = torch.tensor(
        np.asarray(initial_rotation, dtype=np.float32),
        requires_grad=True,
    )
    translation = torch.tensor(
        np.asarray(initial_translation, dtype=np.float32),
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([rotation, translation], lr=lr)

    return drr, rotation, translation, optimizer
