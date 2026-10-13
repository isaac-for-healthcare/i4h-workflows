# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feed measurements must remain correct when rendering slows or input stops."""

from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.medical.catheter_diagnostics import (
    RECORDED_CONTAINMENT_FIELDS,
    CatheterEpisodeDiagnostics,
    InsertionSample,
    insertion_report,
    route_coordinate_m,
)
from i4h_arena.medical.newton_catheter_physics import containment_report


def sample(simulation_s, wall_s, feed, offset=0.0):
    points = np.array([[offset, 0.0, 0.0], [offset + 0.1, 0.0, 0.0]])
    return InsertionSample(simulation_s, wall_s, feed, points)


@pytest.mark.parametrize("wall_seconds", [0.1, 1.0, 10.0])
def test_slow_rendering_changes_real_time_factor_but_not_feed_speed(wall_seconds):
    report = insertion_report(sample(0, 0, 0), sample(1, wall_seconds, 0.009, 0.009))
    assert report["commanded_mps"] == pytest.approx(0.009)
    assert report["root_mps"] == pytest.approx(0.009)
    assert report["tip_mps"] == pytest.approx(0.009)
    assert report["real_time_factor"] == pytest.approx(1 / wall_seconds)


def test_command_integral_includes_a_stop_within_the_measurement_window():
    # 9 mm/s for half a second, followed by zero for half a second.
    report = insertion_report(sample(0, 0, 0), sample(1, 10, 0.0045, 0.0045))
    assert report["commanded_mps"] == pytest.approx(0.0045)
    assert report["root_m"] == pytest.approx(report["commanded_m"])


def test_retraction_remains_signed():
    report = insertion_report(sample(1, 1, 0.01, 0.01), sample(2, 2, 0.006, 0.006))
    assert report["commanded_mps"] == pytest.approx(-0.004)
    assert report["tip_mps"] == pytest.approx(-0.004)


def test_tip_progress_follows_a_bend_in_the_route():
    route = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    old = InsertionSample(0, 0, 0, np.array([[0, 0, 0], [0.9, 0, 0]], dtype=float))
    new = InsertionSample(1, 3, 0.3, np.array([[0, 0, 0], [1, 0.2, 0]], dtype=float))
    report = insertion_report(old, new, route)
    assert report["tip_m"] == pytest.approx(0.3)
    assert report["root_m"] == 0.0


def test_lateral_tip_movement_does_not_count_as_route_progress():
    route = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    assert route_coordinate_m(np.array([0.4, 0.1, 0]), route) == pytest.approx(0.4)


@pytest.mark.parametrize("sim, wall", [(0, 1), (1, 0), (-1, 1)])
def test_measurements_cannot_cross_a_clock_reset(sim, wall):
    with pytest.raises(ValueError, match="time must both advance"):
        insertion_report(sample(0, 0, 0), sample(sim, wall, 0))


# --------------------------------------------------------------------------- #
# Episode recording columns
#
# A prolapsed catheter still reports plausible insertion and rotation, and the
# projection hides the fold behind the anatomy, so an episode can reach the
# target and be a demonstration of the wire coiling. These columns are what
# make that difference readable back off the recording.
# --------------------------------------------------------------------------- #
class _Catheter:
    """Stands in for the scene entity, whose data holds (num_envs, N, 3)."""

    def __init__(self, positions):
        self.data = SimpleNamespace(positions_world_m=None if positions is None else np.asarray(positions))


def _straight_rod(num_envs=1, num_points=5):
    single = np.stack([np.linspace(0.0, 0.04, num_points), np.zeros(num_points), np.zeros(num_points)], axis=1)
    return np.stack([single + index for index in range(num_envs)])


def test_the_tip_and_root_are_recorded_as_world_positions():
    provider = CatheterEpisodeDiagnostics(_Catheter(_straight_rod()))

    values = provider.diagnostics()

    np.testing.assert_allclose(values["root_world_m"], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(values["tip_world_m"], [0.04, 0.0, 0.0], atol=1e-6)


def test_the_target_distance_matches_the_operator_readout():
    """Same quantity arrival is judged on, so a recording can be checked against
    the number the operator was watching."""
    provider = CatheterEpisodeDiagnostics(_Catheter(_straight_rod()), target_world_m=(0.04, 0.03, 0.0))

    assert provider.diagnostics()["tip_target_distance_m"] == pytest.approx(0.03)


def test_only_the_first_environment_is_measured():
    """The schema is one trajectory per demo. Flattening the batch would splice
    every rod into one polyline and put the tip between two of them."""
    provider = CatheterEpisodeDiagnostics(_Catheter(_straight_rod(num_envs=3)))

    np.testing.assert_allclose(provider.diagnostics()["tip_world_m"], [0.04, 0.0, 0.0], atol=1e-6)


def test_a_scene_without_a_target_still_records_the_tip():
    """A phantom scene has no centerline and so no goal, which is not a failure."""
    values = CatheterEpisodeDiagnostics(_Catheter(_straight_rod())).diagnostics()

    assert "tip_world_m" in values
    assert "tip_target_distance_m" not in values


def test_nothing_is_recorded_before_there_are_particles():
    """Newton has no particles until the model is finalized, and a frame of
    zeros would read as a rod parked at the origin."""
    assert CatheterEpisodeDiagnostics(_Catheter(None)).diagnostics() == {}


def test_a_torch_tensor_is_brought_to_the_host():
    torch = pytest.importorskip("torch")
    provider = CatheterEpisodeDiagnostics(_Catheter(torch.as_tensor(_straight_rod(), dtype=torch.float32)))

    np.testing.assert_allclose(provider.diagnostics()["tip_world_m"], [0.04, 0.0, 0.0], atol=1e-6)


def test_the_containment_fields_are_the_ones_the_live_probe_prints():
    """Recorded names have to match the probe's, or a run's log and its
    recording describe the same rod in two vocabularies."""
    report = containment_report(
        _straight_rod()[0],
        path_world_m=np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]]),
        lumen_radii_m=np.array([0.008, 0.008]),
        segment_length_m=0.01,
    )

    recorded = set(RECORDED_CONTAINMENT_FIELDS)
    # The live-vessel fields only exist once a vessel has been built.
    assert recorded - {"live_worst_penetration_mm", "live_samples_outside"} <= set(report)


def test_the_constant_fields_are_left_out():
    """Recording num_particles or rest_length_mm would repeat one number on
    every frame of every episode; they belong to the spec, not the trajectory."""
    for constant in ("num_particles", "rest_length_mm", "num_bend_nodes"):
        assert constant not in RECORDED_CONTAINMENT_FIELDS
