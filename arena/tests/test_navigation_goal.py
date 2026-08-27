# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the catheter navigation goal.

The hold requirement is the valuable part. A tip that sweeps through the target
on one step has not navigated anywhere, so these check the counter against a
fake env rather than needing a live Newton model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from i4h_arena.medical.navigation_goal import (
    ARRIVAL_HOLD_STEPS,
    ARRIVAL_TOLERANCE_M,
    catheter_tip_world_m,
    hold_counter,
    reached_navigation_target,
    reset_arrival_progress,
    tip_distance_to_target_m,
)

TARGET = (1.0, 0.0, 0.0)


class _FakeEnv:
    """Minimum an IsaacLab term reads: a catheter polyline and env sizing."""

    def __init__(self, positions: torch.Tensor | None, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = {"catheter": SimpleNamespace(data=SimpleNamespace(positions_world_m=positions))}

    def place_tip(self, *tips: tuple[float, float, float]) -> None:
        """Rewrite each environment's polyline so its last point is ``tips[i]``."""
        positions = torch.zeros((self.num_envs, 3, 3))
        for index, tip in enumerate(tips):
            positions[index, -1] = torch.tensor(tip)
        self.scene["catheter"].data.positions_world_m = positions


def _env_at(*tips: tuple[float, float, float]) -> _FakeEnv:
    env = _FakeEnv(None, num_envs=len(tips))
    env.place_tip(*tips)
    return env


# --------------------------------------------------------------------------- #
# Tip and distance
# --------------------------------------------------------------------------- #
def test_the_tip_is_the_distal_end_of_the_polyline():
    env = _FakeEnv(torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.9, 0.0, 0.0]]]))

    assert catheter_tip_world_m(env)[0].tolist() == pytest.approx([0.9, 0.0, 0.0])


def test_distance_is_measured_from_the_tip_to_the_target():
    env = _env_at((0.9, 0.0, 0.0))

    assert tip_distance_to_target_m(env, TARGET).item() == pytest.approx(0.1)


def test_a_missing_polyline_never_counts_as_arrival():
    """Newton has no particles to read until its model is finalized."""
    env = _FakeEnv(None)

    assert torch.isinf(tip_distance_to_target_m(env, TARGET)).all()
    assert not reached_navigation_target(env, TARGET, hold_steps=1).any()


def test_the_target_is_not_offset_per_environment():
    """Every rod is seeded from the same absolute centerline positions."""
    env = _env_at((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    env.scene["env_origins"] = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    assert tip_distance_to_target_m(env, TARGET).tolist() == pytest.approx([0.0, 0.0])


# --------------------------------------------------------------------------- #
# Hold requirement
# --------------------------------------------------------------------------- #
def test_arrival_requires_the_tolerance_to_hold():
    env = _env_at(TARGET)

    for _ in range(ARRIVAL_HOLD_STEPS - 1):
        assert not reached_navigation_target(env, TARGET).any()
    assert reached_navigation_target(env, TARGET).all()


def test_leaving_the_tolerance_restarts_the_hold():
    env = _env_at(TARGET)
    for _ in range(ARRIVAL_HOLD_STEPS - 1):
        reached_navigation_target(env, TARGET)

    env.place_tip((0.5, 0.0, 0.0))
    assert not reached_navigation_target(env, TARGET).any()
    assert hold_counter(env).item() == 0

    env.place_tip(TARGET)
    assert not reached_navigation_target(env, TARGET).any()


def test_a_tip_outside_the_tolerance_never_arrives():
    env = _env_at((1.0 + 2.0 * ARRIVAL_TOLERANCE_M, 0.0, 0.0))

    for _ in range(2 * ARRIVAL_HOLD_STEPS):
        assert not reached_navigation_target(env, TARGET).any()


def test_environments_hold_independently():
    env = _env_at(TARGET, (0.5, 0.0, 0.0))

    for _ in range(ARRIVAL_HOLD_STEPS):
        arrived = reached_navigation_target(env, TARGET)

    assert arrived.tolist() == [True, False]


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
def test_reset_clears_only_the_listed_environments():
    env = _env_at(TARGET, TARGET)
    for _ in range(ARRIVAL_HOLD_STEPS - 1):
        reached_navigation_target(env, TARGET)

    reset_arrival_progress(env, torch.tensor([0]))

    assert hold_counter(env).tolist() == [0, ARRIVAL_HOLD_STEPS - 1]
    assert reached_navigation_target(env, TARGET).tolist() == [False, True]


def test_reset_without_ids_clears_every_environment():
    env = _env_at(TARGET, TARGET)
    reached_navigation_target(env, TARGET)

    reset_arrival_progress(env)

    assert hold_counter(env).tolist() == [0, 0]


def test_the_counter_is_reallocated_when_the_environment_count_changes():
    env = _env_at(TARGET)
    reached_navigation_target(env, TARGET)

    env.num_envs = 3
    assert hold_counter(env).tolist() == [0, 0, 0]
