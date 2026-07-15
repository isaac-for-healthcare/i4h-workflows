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

"""Fluoroscopy Simulator - High-Level API for X-ray Image Simulation.

This package provides a simple, object-oriented API for generating simulated
fluoroscopy (X-ray) images from CT volumes. It wraps the underlying Slang-based
GPU rendering pipeline with a clean interface.

Example:
    >>> from fluorosim import VolumePreprocessor, FluoroSimulator, SimulatorConfig
    >>>
    >>> # Step 1: Preprocess CT volume
    >>> preprocessor = VolumePreprocessor.from_dicom("/path/to/dicom/")
    >>> volume = preprocessor.preprocess(output_dir="/tmp/fluoro_cache")
    >>>
    >>> # Step 2: Generate fluoroscopy frames
    >>> config = SimulatorConfig()
    >>> simulator = FluoroSimulator(volume, config)
    >>> frame = simulator.render_frame(rotation=(0, 0, 0), translation=(0, 0, 0))
"""

from .catheter_provider import CatheterProvider, SolverCatheterAdapter, StaticCatheterProvider
from .config import (
    CarmGeometry,
    DSASettings,
    HuToMuMapping,
    MetricsSettings,
    OutputSettings,
    PreprocessingSettings,
    RealismSettings,
    SimulatorConfig,
    XrayPhysics,
)
from .dsa import DSAFrame, DSAPipeline
from .preprocessor import VolumePreprocessor
from .simulator import CineSequence, FluoroSimulator, Frame, Pose, SimulatorMetrics
from .vasculature import (
    TOTALSEG_CORONARY_LABEL,
    TOTALSEG_VESSEL_TERRITORY_MAP,
    CenterlineGraph,
    VesselSegmentationResult,
    apply_vessel_boost,
    build_contrast_volume,
    compute_arrival_map,
    extract_centerlines,
    extract_vessel_mesh,
    gamma_variate,
    get_vessel_mask,
    vessel_mask_from_hu,
    vessel_mask_from_totalsegmentator,
)
from .volume import PreprocessedVolume, VolumeMetadata

__all__ = [
    # Configuration
    "SimulatorConfig",
    "CarmGeometry",
    "XrayPhysics",
    "RealismSettings",
    "DSASettings",
    "OutputSettings",
    "MetricsSettings",
    "PreprocessingSettings",
    "HuToMuMapping",
    # Volume
    "PreprocessedVolume",
    "VolumePreprocessor",
    # Simulator
    "FluoroSimulator",
    "Pose",
    "Frame",
    "CineSequence",
    "SimulatorMetrics",
    # Catheter provider
    "CatheterProvider",
    "SolverCatheterAdapter",
    "StaticCatheterProvider",
    # DSA pipeline
    "DSAPipeline",
    "DSAFrame",
    # Vasculature utilities
    "apply_vessel_boost",
    "CenterlineGraph",
    "extract_centerlines",
    "compute_arrival_map",
    "gamma_variate",
    "build_contrast_volume",
    "TOTALSEG_CORONARY_LABEL",
    "TOTALSEG_VESSEL_TERRITORY_MAP",
    "VesselSegmentationResult",
    "extract_vessel_mesh",
    "get_vessel_mask",
    "vessel_mask_from_hu",
    "vessel_mask_from_totalsegmentator",
    # Volume metadata
    "VolumeMetadata",
]

__version__ = "0.1.0"
