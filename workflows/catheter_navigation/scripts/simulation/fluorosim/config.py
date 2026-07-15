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

"""Configuration dataclasses for the Fluoroscopy Simulator.

This module defines all configuration objects used by the simulator API.
All configs are frozen dataclasses with sensible defaults for clinical C-arm geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class CarmGeometry:
    """C-arm geometry configuration.

    Defines the physical geometry of the X-ray imaging system including
    source-detector distances and detector specifications.

    Attributes:
        source_to_detector_mm: Distance from X-ray source to detector plane (mm).
            Also called SDD (Source-to-Detector Distance). Typical range: 990–1250 mm.
        source_to_isocenter_mm: Distance from X-ray source to isocenter (mm).
            Also called SID (Source-to-Isocenter Distance). The isocenter is the
            rotation center, typically at patient table level. Typical range: 495–780 mm.
        detector_width_px: Detector width in pixels.
        detector_height_px: Detector height in pixels.
        pixel_spacing_mm: Physical size of each detector pixel (mm).

    Vendor-Specific Configuration:
        Different C-arm vendors (GE, Siemens, Philips, Ziehm, etc.) have distinct
        geometry specifications. To configure for a specific vendor system:

        1. **SDD and SID**: These define the X-ray cone geometry and magnification.
           - Mobile C-arms typically have SDD ~1000mm with SID ~500mm (2x magnification)
           - Fixed interventional systems have larger SDD ~1200-1250mm with SID ~750-780mm

        2. **Detector size**: Varies by detector type and generation.
           - Image intensifiers (legacy): 1024×1024 pixels
           - Flat-panel detectors: 1536×1536 to 2480×1920 pixels

        3. **Pixel spacing**: Determines field of view (FOV = pixels × spacing).
           - Typical range: 0.15–0.20 mm/pixel
           - FOV = detector_px × pixel_spacing_mm (e.g., 1024 × 0.194 ≈ 200mm = 20cm)

        4. **Magnification**: Calculated as SDD / SID.
           - At isocenter, objects appear magnified by this factor on the detector.
           - Typical range: 1.5x to 2.0x

        Example vendor-specific configurations::

            # GE OEC 9900 (mobile C-arm, 12" II equivalent)
            geometry = CarmGeometry(
                source_to_detector_mm=1020.0,
                source_to_isocenter_mm=510.0,
                detector_width_px=1024,
                detector_height_px=1024,
                pixel_spacing_mm=0.194,  # ~20cm FOV
            )

            # Siemens Artis zee (fixed biplane angiography)
            geometry = CarmGeometry(
                source_to_detector_mm=1250.0,
                source_to_isocenter_mm=780.0,
                detector_width_px=2480,
                detector_height_px=1920,
                pixel_spacing_mm=0.154,  # ~38×30cm FOV
            )

            # Philips Azurion 7 (premium interventional)
            geometry = CarmGeometry(
                source_to_detector_mm=1240.0,
                source_to_isocenter_mm=780.0,
                detector_width_px=2480,
                detector_height_px=1920,
                pixel_spacing_mm=0.154,
            )

        Reference specifications (approximate values, verify with actual system docs):

        | Vendor/Model              | SDD (mm) | SID (mm) | Detector   | Pixel (mm) |
        |---------------------------|----------|----------|------------|------------|
        | GE OEC 9900               | 1020     | 510      | 1024×1024  | 0.194      |
        | GE OEC Elite CFD          | 1150     | 575      | 1920×1920  | 0.154      |
        | GE Innova IGS 540         | 1200     | 750      | 2048×2048  | 0.200      |
        | Siemens Arcadis Avantic   | 1000     | 500      | 1024×1024  | 0.195      |
        | Siemens Cios Alpha        | 1100     | 550      | 1536×1536  | 0.178      |
        | Siemens Artis zee         | 1250     | 780      | 2480×1920  | 0.154      |
        | Philips BV Pulsera        | 990      | 495      | 1024×1024  | 0.200      |
        | Philips Azurion 7         | 1240     | 780      | 2480×1920  | 0.154      |
        | Ziehm Vision RFD 3D       | 1000     | 500      | 1024×1024  | 0.194      |
    """

    source_to_detector_mm: float = 1020.0
    source_to_isocenter_mm: float = 510.0
    detector_width_px: int = 512
    detector_height_px: int = 512
    pixel_spacing_mm: float = 0.5

    @property
    def detector_size_mm(self) -> tuple[float, float]:
        """Physical detector size (width, height) in mm."""
        return (
            self.detector_width_px * self.pixel_spacing_mm,
            self.detector_height_px * self.pixel_spacing_mm,
        )

    @classmethod
    def ge_oec_9900(cls) -> "CarmGeometry":
        """GE OEC 9900 — mobile C-arm, 12-inch image intensifier equivalent."""
        return cls(
            source_to_detector_mm=1020.0,
            source_to_isocenter_mm=510.0,
            detector_width_px=1024,
            detector_height_px=1024,
            pixel_spacing_mm=0.194,
        )

    @classmethod
    def ge_oec_elite_cfd(cls) -> "CarmGeometry":
        """GE OEC Elite CFD — mobile C-arm with flat-panel detector."""
        return cls(
            source_to_detector_mm=1150.0,
            source_to_isocenter_mm=575.0,
            detector_width_px=1920,
            detector_height_px=1920,
            pixel_spacing_mm=0.154,
        )

    @classmethod
    def ge_innova_igs_540(cls) -> "CarmGeometry":
        """GE Innova IGS 540 — fixed interventional angiography suite."""
        return cls(
            source_to_detector_mm=1200.0,
            source_to_isocenter_mm=750.0,
            detector_width_px=2048,
            detector_height_px=2048,
            pixel_spacing_mm=0.200,
        )

    @classmethod
    def siemens_arcadis_avantic(cls) -> "CarmGeometry":
        """Siemens Arcadis Avantic — mobile surgical C-arm."""
        return cls(
            source_to_detector_mm=1000.0,
            source_to_isocenter_mm=500.0,
            detector_width_px=1024,
            detector_height_px=1024,
            pixel_spacing_mm=0.195,
        )

    @classmethod
    def siemens_cios_alpha(cls) -> "CarmGeometry":
        """Siemens Cios Alpha — compact mobile C-arm with flat panel."""
        return cls(
            source_to_detector_mm=1100.0,
            source_to_isocenter_mm=550.0,
            detector_width_px=1536,
            detector_height_px=1536,
            pixel_spacing_mm=0.178,
        )

    @classmethod
    def siemens_artis_zee(cls) -> "CarmGeometry":
        """Siemens Artis zee — fixed biplane angiography system."""
        return cls(
            source_to_detector_mm=1250.0,
            source_to_isocenter_mm=780.0,
            detector_width_px=2480,
            detector_height_px=1920,
            pixel_spacing_mm=0.154,
        )

    @classmethod
    def philips_bv_pulsera(cls) -> "CarmGeometry":
        """Philips BV Pulsera — mobile C-arm with image intensifier."""
        return cls(
            source_to_detector_mm=990.0,
            source_to_isocenter_mm=495.0,
            detector_width_px=1024,
            detector_height_px=1024,
            pixel_spacing_mm=0.200,
        )

    @classmethod
    def philips_azurion_7(cls) -> "CarmGeometry":
        """Philips Azurion 7 — premium interventional flat-panel system."""
        return cls(
            source_to_detector_mm=1240.0,
            source_to_isocenter_mm=780.0,
            detector_width_px=2480,
            detector_height_px=1920,
            pixel_spacing_mm=0.154,
        )

    @classmethod
    def ziehm_vision_rfd_3d(cls) -> "CarmGeometry":
        """Ziehm Vision RFD 3D — motorised mobile C-arm with CBCT."""
        return cls(
            source_to_detector_mm=1000.0,
            source_to_isocenter_mm=500.0,
            detector_width_px=1024,
            detector_height_px=1024,
            pixel_spacing_mm=0.194,
        )


@dataclass(frozen=True)
class XrayPhysics:
    """X-ray physics configuration.

    Controls the physical parameters of the X-ray simulation including
    ray-marching resolution and intensity settings.

    Attributes:
        step_mm: Ray-marching step size (mm). Smaller = more accurate but slower.
        i0: Unattenuated X-ray intensity (incident beam intensity).
        normalize: If True, normalize output image to [0, 1] range.
        invert: If True, invert image so bone=white, air=black (clinical convention).
    """

    step_mm: float = 0.5
    i0: float = 1.0
    normalize: bool = True
    invert: bool = True


@dataclass(frozen=True)
class HuToMuMapping:
    """Hounsfield Unit to linear attenuation coefficient mapping.

    Defines the linear mapping from HU values to μ (mm⁻¹) for X-ray simulation.

    Attributes:
        hu_min: Minimum HU value (maps to mu_min). Default: -1000 (air).
        hu_max: Maximum HU value (maps to mu_max). Default: 3000 (dense bone).
        mu_min: Minimum μ value (mm⁻¹). Default: 0.0.
        mu_max: Maximum μ value (mm⁻¹). Default: 0.02.
    """

    hu_min: float = -1000.0
    hu_max: float = 3000.0
    mu_min: float = 0.0
    mu_max: float = 0.02


@dataclass(frozen=True)
class RealismSettings:
    """Realism post-processing settings.

    Controls noise, blur, and other post-processing effects to make
    simulated fluoroscopy more realistic.

    Attributes:
        enabled: If True, apply realism effects.
        gain: Linear intensity scaling factor.
        bias: Intensity offset (added after gain).
        poisson_photons: Photon count for Poisson noise (0 = disabled).
        gaussian_sigma: Gaussian noise sigma (0 = disabled).
        blur_sigma_px: Gaussian blur sigma in pixels (0 = disabled).
        scatter_sigma_px: Scatter convolution kernel sigma in pixels (0 = disabled).
            Models the physical X-ray scatter field S = G(sigma_s) * I.
        gamma: Display gamma correction exponent (1.0 = disabled).
            Clinical displays typically use gamma ~0.8 (brightens midtones).
            Applied as image = image^(1/gamma) after all other effects.
        seed: Random seed for reproducibility (None = random).
    """

    enabled: bool = False
    gain: float = 1.0
    bias: float = 0.0
    poisson_photons: float = 0.0
    gaussian_sigma: float = 0.0
    blur_sigma_px: float = 0.0
    scatter_sigma_px: float = 0.0
    gamma: float = 1.0
    seed: int | None = 0


@dataclass(frozen=True)
class DSASettings:
    """Digital Subtraction Angiography pipeline settings.

    Controls the DSA rendering mode: contrast DRR - mask DRR → subtraction
    → post-processing. This is the rendering mode clinicians use during
    catheter procedures.

    Attributes:
        enabled: If True, render in DSA mode.
        contrast_boost: Multiplicative scaling on the subtraction image (k).
        gamma: Display gamma for the DSA output.
        scatter_sigma_px: Scatter convolution sigma for both mask and contrast DRRs.
        misregistration_px: Sub-pixel mask shift (dx, dy) for realistic DSA artifacts.
            Simulates patient motion between mask and contrast acquisitions.
        poisson_photons: Photon noise applied independently to mask and contrast DRRs.
        gaussian_sigma: Additive Gaussian noise sigma.
    """

    enabled: bool = False
    contrast_boost: float = 20.0
    gamma: float = 0.8
    scatter_sigma_px: float = 0.0
    misregistration_px: tuple[float, float] = (0.0, 0.0)
    poisson_photons: float = 0.0
    gaussian_sigma: float = 0.0


@dataclass(frozen=True)
class OutputSettings:
    """Output configuration for rendered frames.

    Attributes:
        save_to_disk: If True, save rendered frames to disk.
        output_dir: Directory for saved frames (used if save_to_disk=True).
        format: Output format for saved frames.
        keep_in_gpu: If True, keep frames in GPU memory (for streaming).
    """

    save_to_disk: bool = False
    output_dir: str | Path | None = None
    format: Literal["png", "npy", "npz"] = "png"
    keep_in_gpu: bool = False


@dataclass(frozen=True)
class MetricsSettings:
    """Performance metrics collection settings.

    Attributes:
        enabled: If True, collect performance metrics.
        track_fps: Track frames per second.
        track_gpu_usage: Track GPU memory and utilization.
        track_jitter: Track frame timing jitter.
    """

    enabled: bool = False
    track_fps: bool = True
    track_gpu_usage: bool = True
    track_jitter: bool = True


@dataclass(frozen=True)
class PreprocessingSettings:
    """CT preprocessing settings.

    Attributes:
        hu_clip_min: Minimum HU value for clipping.
        hu_clip_max: Maximum HU value for clipping.
        clip_hu: If True, clip HU values to [hu_clip_min, hu_clip_max].
        hu_to_mu: HU to μ mapping configuration.
    """

    hu_clip_min: float = -1024.0
    hu_clip_max: float = 3071.0
    clip_hu: bool = True
    hu_to_mu: HuToMuMapping = field(default_factory=HuToMuMapping)


@dataclass(frozen=True)
class SimulatorConfig:
    """Unified configuration for the Fluoroscopy Simulator.

    This is the main configuration object that bundles all settings for
    the simulator. Pass this to FluoroSimulator to control rendering behavior.

    Attributes:
        geometry: C-arm geometry settings.
        physics: X-ray physics settings.
        realism: Realism post-processing settings.
        output: Output settings for rendered frames.
        metrics: Performance metrics settings.
        backend: Rendering backend (currently only "slang" is supported).

    Example:
        >>> config = SimulatorConfig(
        ...     geometry=CarmGeometry(detector_width_px=1024, detector_height_px=1024),
        ...     realism=RealismSettings(enabled=True, gaussian_sigma=0.01),
        ...     output=OutputSettings(save_to_disk=True, output_dir="/tmp/frames"),
        ... )
    """

    geometry: CarmGeometry = field(default_factory=CarmGeometry)
    physics: XrayPhysics = field(default_factory=XrayPhysics)
    realism: RealismSettings = field(default_factory=RealismSettings)
    dsa: DSASettings = field(default_factory=DSASettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    metrics: MetricsSettings = field(default_factory=MetricsSettings)
    backend: Literal["slang"] = "slang"

    def with_geometry(self, **kwargs) -> "SimulatorConfig":
        """Return a new config with updated geometry settings."""
        new_geometry = CarmGeometry(**{**self.geometry.__dict__, **kwargs})
        return SimulatorConfig(
            geometry=new_geometry,
            physics=self.physics,
            realism=self.realism,
            dsa=self.dsa,
            output=self.output,
            metrics=self.metrics,
            backend=self.backend,
        )

    def with_realism(self, **kwargs) -> "SimulatorConfig":
        """Return a new config with updated realism settings."""
        new_realism = RealismSettings(**{**self.realism.__dict__, **kwargs})
        return SimulatorConfig(
            geometry=self.geometry,
            physics=self.physics,
            realism=new_realism,
            dsa=self.dsa,
            output=self.output,
            metrics=self.metrics,
            backend=self.backend,
        )

    def with_dsa(self, **kwargs) -> "SimulatorConfig":
        """Return a new config with updated DSA settings."""
        new_dsa = DSASettings(**{**self.dsa.__dict__, **kwargs})
        return SimulatorConfig(
            geometry=self.geometry,
            physics=self.physics,
            realism=self.realism,
            dsa=new_dsa,
            output=self.output,
            metrics=self.metrics,
            backend=self.backend,
        )

    def with_output(self, **kwargs) -> "SimulatorConfig":
        """Return a new config with updated output settings."""
        new_output = OutputSettings(**{**self.output.__dict__, **kwargs})
        return SimulatorConfig(
            geometry=self.geometry,
            physics=self.physics,
            realism=self.realism,
            dsa=self.dsa,
            output=new_output,
            metrics=self.metrics,
            backend=self.backend,
        )
