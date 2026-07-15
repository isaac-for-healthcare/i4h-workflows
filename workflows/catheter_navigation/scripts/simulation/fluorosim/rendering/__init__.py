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

"""Fluoroscopy rendering using Slang DiffDRR.

This package provides a GPU-accelerated DRR (Digitally Reconstructed Radiograph) renderer
using NVIDIA Slang with automatic differentiation support.

**Slang DiffDRR** (`SlangDiffDRRConfig`, `SlangDiffDRRRenderer`, `TorchSlangDiffDRR`)
- Compiler-level automatic differentiation via Slang autodiff
- Exact analytical gradients (not finite differences)
- Native CUDA backend
- Optimal performance for 2D/3D registration
- Requires: `pip install slangpy`
"""

from .diffdrr_slang_renderer import (
    CatheterSegmentData,
    SlangDiffDRRConfig,
    SlangDiffDRRFunction,
    SlangDiffDRRRenderer,
    TorchSlangDiffDRR,
    create_slang_diffdrr_optimizer,
    render_diffdrr_slang,
)
from .realism import RealismConfig, apply_misregistration, apply_realism, apply_scatter

__all__ = [
    # Slang DiffDRR (differentiable with autodiff)
    "CatheterSegmentData",
    "SlangDiffDRRConfig",
    "SlangDiffDRRRenderer",
    "TorchSlangDiffDRR",
    "SlangDiffDRRFunction",
    "render_diffdrr_slang",
    "create_slang_diffdrr_optimizer",
    # Realism post-processing
    "RealismConfig",
    "apply_realism",
    "apply_scatter",
    "apply_misregistration",
]
