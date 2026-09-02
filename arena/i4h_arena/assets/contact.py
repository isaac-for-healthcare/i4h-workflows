# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable contact-sensor authoring helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def filtered_contact_sensor_family(
    *,
    family_name: str,
    sensing_prim_root: str,
    body_names: Iterable[str],
    filter_prim_path: str,
) -> dict[str, Any]:
    """Build one filtered sensor per rigid body for an aggregate contact family."""
    from isaaclab.sensors import ContactSensorCfg

    if not family_name or "__" in family_name:
        raise ValueError("family_name must be non-empty and must not contain '__'")
    bodies = tuple(body_names)
    if not bodies or any(not body for body in bodies):
        raise ValueError("body_names must contain at least one non-empty name")
    if len(set(bodies)) != len(bodies):
        raise ValueError("body_names must be unique")
    if not sensing_prim_root or not filter_prim_path:
        raise ValueError("sensing_prim_root and filter_prim_path must be non-empty")

    root = sensing_prim_root.rstrip("/")
    return {
        f"{family_name}__{body}": ContactSensorCfg(
            prim_path=f"{root}/{body}",
            filter_prim_paths_expr=[filter_prim_path],
        )
        for body in bodies
    }
