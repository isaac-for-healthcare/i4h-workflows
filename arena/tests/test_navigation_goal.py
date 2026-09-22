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
    DRIFT_LOG_ENV_VAR,
    arrival_progress,
    arrival_readout,
    arrival_status,
    catheter_tip_world_m,
    drift_log_seconds,
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
def test_the_drift_log_is_off_until_an_interval_is_asked_for():
    """A line per UI frame would bury a teleop log, and the creep being measured
    is a slope in mm/s, so the caller picks the interval."""
    assert drift_log_seconds({}) == 0.0
    assert drift_log_seconds({DRIFT_LOG_ENV_VAR: "0.5"}) == 0.5
    assert drift_log_seconds({DRIFT_LOG_ENV_VAR: "0"}) == 0.0
    assert drift_log_seconds({DRIFT_LOG_ENV_VAR: "-2"}) == 0.0
    # A mistyped diagnostic must not take a teleop session down.
    assert drift_log_seconds({DRIFT_LOG_ENV_VAR: "every second"}) == 0.0


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


# --------------------------------------------------------------------------- #
# Operator readout
# --------------------------------------------------------------------------- #
class _FakeTerminationManager:
    """The slice of IsaacLab's manager the readout reads its thresholds from."""

    def __init__(self, **params) -> None:
        self.active_terms = ["success"]
        self._cfg = SimpleNamespace(params={"target_world_m": TARGET, **params})

    def get_term_cfg(self, name: str) -> SimpleNamespace:
        assert name == "success"
        return self._cfg


def _env_with_term(*tips, **params) -> _FakeEnv:
    env = _env_at(*tips)
    env.termination_manager = _FakeTerminationManager(**params)
    return env


def test_a_distant_tip_reads_out_the_gap_and_the_tolerance():
    assert arrival_readout(0.0274) == "Target: 27.4 mm away (arrive within 5 mm)"


def test_a_tip_inside_the_tolerance_reads_out_the_hold():
    """Being close is not arrival, so the operator needs to see the hold too."""
    assert arrival_readout(0.0031, 7) == "Target: 3.1 mm away -- holding 7/15"


def test_an_unknown_tip_says_so_rather_than_reading_out_infinity():
    assert arrival_readout(float("inf")) == "Target: waiting for the catheter"


def test_the_readout_never_promises_more_hold_than_the_criterion_wants():
    assert arrival_readout(0.0, ARRIVAL_HOLD_STEPS + 4).endswith(f"{ARRIVAL_HOLD_STEPS}/{ARRIVAL_HOLD_STEPS}")


def test_the_readout_follows_the_terms_own_thresholds():
    """A readout derived from anything else could disagree with what ends the episode."""
    env = _env_with_term((0.98, 0.0, 0.0), tolerance_m=0.05, hold_steps=3)

    assert arrival_status(env) == "Target: 20.0 mm away -- holding 0/3"


def test_progress_reports_the_live_distance_and_hold():
    env = _env_with_term(TARGET)
    for _ in range(4):
        reached_navigation_target(env, TARGET)

    progress = arrival_progress(env)

    assert progress.distance_m == pytest.approx(0.0)
    assert progress.held_steps == 4
    assert (progress.tolerance_m, progress.hold_steps) == (ARRIVAL_TOLERANCE_M, ARRIVAL_HOLD_STEPS)
    # No route configured, so the readout has only the straight line to go on.
    assert progress.route is None


def test_reading_the_progress_does_not_spend_the_hold():
    """The operator still has to earn every step of the hold on their own."""
    env = _env_with_term(TARGET)
    reached_navigation_target(env, TARGET)

    for _ in range(5):
        arrival_status(env)

    assert hold_counter(env).item() == 1


def test_a_scene_without_the_success_term_gets_no_readout():
    env = _env_at(TARGET)

    assert arrival_progress(env) is None
    assert arrival_status(env) == ""


# --------------------------------------------------------------------------- #
# Remaining vessel
#
# Straight-line distance to the far end of the centerline is what ends the
# episode, but on a vessel that doubles back it grows while the tip advances
# correctly. The readout leads with remaining arc so the operator has a number
# they can steer on, and keeps the straight line so they can see arrival come.
# --------------------------------------------------------------------------- #
def _route(remaining_m, lateral_m=0.0, on_route=True):
    from i4h_arena.medical.route_progress import RouteProgress

    return RouteProgress(arc_m=0.5 - remaining_m, remaining_m=remaining_m, lateral_m=lateral_m, on_route=on_route)


def test_the_readout_leads_with_the_vessel_still_ahead():
    line = arrival_readout(0.046, route=_route(0.3431))

    assert line == "Target: 343 mm of vessel ahead -- 46.0 mm direct (arrive within 5 mm)"


def test_the_straight_line_survives_alongside_it():
    """It is the quantity that ends the episode, so hiding it would leave the
    operator unable to see arrival coming."""
    assert "46.0 mm direct" in arrival_readout(0.046, route=_route(0.3431))


def test_an_off_route_tip_does_not_get_an_arc_figure():
    """Off route the projection may be measuring a stretch of vessel the tip is
    not in, and a confident wrong number is worse than none."""
    line = arrival_readout(0.046, route=_route(0.3431, lateral_m=0.05, on_route=False))

    assert line == "Target: 46.0 mm direct, tip off route (arrive within 5 mm)"


def test_inside_the_tolerance_the_hold_takes_over():
    """The arc is spent by then, and the hold is the only thing left to earn."""
    line = arrival_readout(0.0031, 7, route=_route(0.0))

    assert line == "Target: 3.1 mm away -- holding 7/15"


def test_a_scene_with_no_route_reads_out_exactly_as_before():
    assert arrival_readout(0.0274, route=None) == "Target: 27.4 mm away (arrive within 5 mm)"


def test_an_unknown_tip_still_says_so_even_with_a_route():
    assert arrival_readout(float("inf"), route=_route(0.3)) == "Target: waiting for the catheter"


def test_the_live_readout_measures_the_route_off_the_success_term():
    """Same place the criterion reads its target, so the two cannot drift."""
    route = tuple((x / 100.0, 0.0, 0.0) for x in range(31))  # 0.30 m straight

    env = _env_with_term((0.10, 0.0, 0.0), target_world_m=(0.30, 0.0, 0.0), route_world_m=route)

    progress = arrival_progress(env)
    assert progress.route is not None
    assert progress.route.remaining_m == pytest.approx(0.20)
    assert arrival_status(env).startswith("Target: 200 mm of vessel ahead")


def test_an_unusable_route_falls_back_to_the_straight_line():
    """A bad caption must not take down a scene that still terminates correctly."""
    env = _env_with_term((0.10, 0.0, 0.0), target_world_m=(0.30, 0.0, 0.0), route_world_m=((0.0, 0.0, 0.0),))

    assert arrival_progress(env).route is None
    assert arrival_status(env) == "Target: 200.0 mm away (arrive within 5 mm)"
