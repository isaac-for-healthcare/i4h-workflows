# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Head-camera G1 locomanip modality config for GR00T N1.6 fine-tuning.

The default. The G1 scenes record only the ``head`` camera, which
``i4h-dataset convert --robot g1`` maps to the ``ego_view`` video key, and the
base checkpoints (``nvidia/GR00T-N1.6-Rheo-*``) were trained on ``ego_view``
alone.
"""

from i4h_tasks.gr00t_n16.locomanip.modality import register_g1_locomanip

register_g1_locomanip(["ego_view"])
