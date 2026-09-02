# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable success conditions.

These are the geometric checks that legacy workflow implementations buried inside each
env's task module, where they cannot be reused or tested. Here they are plain
functions over a :class:`~i4h_common.world.SceneView`, so they compose into
``WaitUntil``, into a remote task's ``until=``, and into a workflow's own success
criterion — and they unit-test against a fake scene.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from i4h_common.world import SceneView

Predicate = Callable[[Any], np.ndarray]


def object_within(scene: SceneView, name: str, target: str, radius: float) -> np.ndarray:
    """Per-env mask: is ``name`` within ``radius`` metres of ``target``?"""
    return scene.object(name).pose.distance_to(scene.object(target).pose) < radius


def object_above(scene: SceneView, name: str, height: float) -> np.ndarray:
    """Per-env mask: is ``name``'s world z above ``height``?"""
    return scene.object(name).pose.pos[:, 2] > height


def inside_box(
    scene: SceneView,
    name: str,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> np.ndarray:
    """Per-env mask: is ``name`` inside an axis-aligned world-frame box?"""
    pos = scene.object(name).pose.pos
    low = np.asarray(lower, dtype=np.float32)
    high = np.asarray(upper, dtype=np.float32)
    return np.all((pos >= low) & (pos <= high), axis=-1)


def is_settled(scene: SceneView, name: str) -> np.ndarray:
    """Per-env mask: has ``name`` stopped moving?"""
    return scene.object(name).is_settled


def near_object(scene: SceneView, name: str, radius: float, robot: str = "robot") -> np.ndarray:
    """Per-env mask: is the TCP within ``radius`` of ``name``?"""
    return scene.tcp(robot).distance_to(scene.object(name).pose) < radius


def at_home(scene: SceneView, tolerance_rad: float = 0.1, robot: str = "robot") -> np.ndarray:
    """Per-env mask: is every joint within ``tolerance_rad`` of home?"""
    error = np.abs(scene.joints(robot).pos - scene.home_joints(robot)).max(axis=-1)
    return error < tolerance_rad


def all_of(*predicates: Predicate) -> Predicate:
    """Conjunction, element-wise across envs."""

    def combined(ctx: Any) -> np.ndarray:
        result = np.asarray(predicates[0](ctx))
        for predicate in predicates[1:]:
            result = result & np.asarray(predicate(ctx))
        return result

    return combined


def any_of(*predicates: Predicate) -> Predicate:
    """Disjunction, element-wise across envs."""

    def combined(ctx: Any) -> np.ndarray:
        result = np.asarray(predicates[0](ctx))
        for predicate in predicates[1:]:
            result = result | np.asarray(predicate(ctx))
        return result

    return combined


def negate(predicate: Predicate) -> Predicate:
    def inverted(ctx: Any) -> np.ndarray:
        return ~np.asarray(predicate(ctx))

    return inverted
