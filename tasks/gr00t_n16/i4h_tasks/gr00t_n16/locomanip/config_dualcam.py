# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dual-camera G1 locomanip modality config for GR00T N1.6 fine-tuning.

Only for recordings from a G1 scene that adds a ``room`` camera beside
``head``. The shipped G1 scenes record ``head`` alone; use ``config_head.py``
(the default) for them.
"""

from i4h_tasks.gr00t_n16.locomanip.modality import register_g1_locomanip

register_g1_locomanip(["ego_view", "room"])
