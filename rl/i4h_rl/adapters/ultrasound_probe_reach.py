# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation metrics for the compact ultrasound probe reach example."""

from __future__ import annotations

from typing import Any


def evaluation_metrics(env: Any) -> dict[str, Any]:
    """Return per-environment error metrics; the evaluator tracks minima."""
    from i4h_arena.envcfg.ultrasound_probe_reach import orientation_error, position_error

    return {
        "position_error_m": position_error(env),
        "orientation_error_rad": orientation_error(env),
    }
