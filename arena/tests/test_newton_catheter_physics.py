# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for placing the catheter rod under Isaac Lab's Newton manager.

The ordering assertions are the valuable ones. Particles added after the model
is finalized are invisible to it, and a rod registered before its particles
exist has nothing to drive, so this checks that registration happens on
MODEL_INIT and in the right order against stubs rather than a live stack.
"""

from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.medical.newton_catheter_physics import (
    BEND_STIFFNESS_ENV_VAR,
    CLEANUP_ROUNDS_ENV_VAR,
    CLEANUP_SWEEPS_ENV_VAR,
    CONTAINMENT_STAGE_ENV_VAR,
    DAMPING_ENV_VAR,
    DEFAULT_NUM_SEGMENTS,
    GRAVITY_NEUTRAL_BUOYANCY,
    GRAVITY_WORLD_Z_UP,
    INTERIOR_CONTAINMENT_ENV_VAR,
    REST_CURVATURE_ENV_VAR,
    SEGMENT_COUNT_ENV_VAR,
    TIP_EDGES_ENV_VAR,
    VESSEL_COMPLIANCE_ENV_VAR,
    CatheterRodHandle,
    CatheterRodSpec,
    bend_radii_m,
    bend_stiffness_override,
    cleanup_rounds_override,
    cleanup_sweeps_override,
    containment_report,
    containment_stage_override,
    interior_containment_override,
    max_faithful_tip_bend_rad,
    nearest_on_polyline,
    rest_curvature_override,
    rod_damping_override,
    seeded_rest_curvature_scale,
    segment_count_override,
    segment_inverse_inertia,
    tip_bend_polyline_turn_rad,
    tip_bend_rest_component,
    tip_bend_stiffness_profile,
    tip_edge_count_override,
    vessel_compliance_override,
)


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
def test_segment_length_divides_the_catheter():
    spec = CatheterRodSpec(length_m=0.4, num_segments=40)

    assert spec.num_points == 41
    assert spec.segment_length_m == pytest.approx(0.01)


def test_track_direction_is_normalized():
    spec = CatheterRodSpec(track_direction_world=(0.0, 3.0, 4.0))

    assert np.linalg.norm(spec.track_direction_world) == pytest.approx(1.0)
    assert spec.track_direction_world[1] == pytest.approx(0.6)


def test_gravity_defaults_to_neutral_buoyancy():
    """A guidewire in blood does not carry its weight in air, and the wall would
    otherwise have to resist a load the real device never applies."""
    assert CatheterRodSpec().gravity_world == GRAVITY_NEUTRAL_BUOYANCY
    assert not any(GRAVITY_NEUTRAL_BUOYANCY)


def test_a_scene_that_wants_weight_gets_the_z_up_world():
    """Isaac is Z-up; the rod config's own default points along -Y, so the
    opt-in value still has to be named rather than left to the solver."""
    spec = CatheterRodSpec(gravity_world=GRAVITY_WORLD_Z_UP)

    assert spec.gravity_world == GRAVITY_WORLD_Z_UP
    assert GRAVITY_WORLD_Z_UP[2] < 0.0


# --------------------------------------------------------------------------- #
# Vessel wall compliance
#
# These four match omniendo's xcath/scenes/s0065.yaml, the reference scene for
# this same centerline vessel and containment. Before them the wall took every
# contact correction in full, undamped, with its distal end free, and the
# catheter dragged the aorta about 15 mm off the anatomy.
# --------------------------------------------------------------------------- #
def test_the_wall_only_takes_half_of_a_contact_correction():
    """At 1.0 the wall yields completely, so nothing actually contains the wire."""
    assert CatheterRodSpec().vessel_response == pytest.approx(0.5)


def test_both_wall_ends_are_anchored():
    """Held at the root alone, a wall this stiff swings rather than bends."""
    assert CatheterRodSpec().vessel_endpoints_locked is True


def test_the_wall_is_damped():
    """Undamped, the energy a contact puts into the wall stays there."""
    spec = CatheterRodSpec()

    assert spec.vessel_linear_damping == pytest.approx(0.01)
    assert spec.vessel_angular_damping == pytest.approx(0.01)


@pytest.mark.parametrize("name", ["vessel_response", "vessel_linear_damping", "vessel_angular_damping"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_wall_fractions_outside_the_unit_range_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        CatheterRodSpec(**{name: value})


def test_the_wall_override_reads_all_three_together():
    override = vessel_compliance_override({VESSEL_COMPLIANCE_ENV_VAR: "0.25,0.05,0.02"})

    assert override == pytest.approx((0.25, 0.05, 0.02))


def test_an_unset_wall_override_leaves_the_spec_alone():
    assert vessel_compliance_override({}) is None


@pytest.mark.parametrize(
    "raw",
    [
        "0.5,0.01",  # a pair cannot say which of the three it omits
        "0.5,0.01,0.01,0.01",
        "0.5,soft,0.01",
        "1.5,0.01,0.01",  # a response above one over-corrects the wall
        "0.5,-0.01,0.01",
        "",
    ],
)
def test_an_unusable_wall_override_is_ignored_rather_than_raising(raw):
    """A mistyped diagnostic must not decide how the physics runs."""
    assert vessel_compliance_override({VESSEL_COMPLIANCE_ENV_VAR: raw}) is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_envs": 0}, "num_envs"),
        ({"num_segments": 0}, "num_segments"),
        ({"length_m": 0.0}, "length_m"),
        ({"radius_m": -1.0}, "radius_m"),
        ({"track_direction_world": (0.0, 0.0, 0.0)}, "non-zero"),
        ({"containment_stage": "both"}, "containment_stage"),
        ({"tip_bend_fraction": 0.0}, "tip_bend_fraction"),
        ({"tip_bend_fraction": 1.5}, "tip_bend_fraction"),
    ],
)
def test_invalid_specs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CatheterRodSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Tip bend-stiffness taper
# --------------------------------------------------------------------------- #
def test_a_uniform_rod_is_the_default():
    """Nothing tapers until a scene asks for it, so today's runs are unchanged."""
    assert CatheterRodSpec().tip_bend_fraction == 1.0
    assert CatheterRodSpec().containment_stage == "post"


def test_the_shaft_keeps_full_stiffness_and_the_tip_is_relieved():
    profile = tip_bend_stiffness_profile(2.0, num_edges=40, num_tip_edges=8, tip_fraction=0.3)

    assert profile.shape == (40,)
    assert profile[:32] == pytest.approx(2.0)
    assert profile[-1] == pytest.approx(0.6)


def test_the_taper_has_no_stiffness_step_for_the_solve_to_ring_on():
    """A jump in stiffness between neighbouring edges is what excites the rod."""
    profile = tip_bend_stiffness_profile(1.0, num_edges=40, num_tip_edges=10, tip_fraction=0.3)

    assert np.all(np.diff(profile) <= 1e-6), "stiffness must never rise toward the tip"
    # The taper covers the last ten edges, so the junction sits between 29 and 30.
    assert profile[29] == pytest.approx(profile[30]), "no step where the taper begins"
    # A raised cosine also flattens at both ends, so the junction has no kink.
    steps = np.abs(np.diff(profile[29:]))
    assert steps[0] < steps.max() / 5.0, "the taper must ease in rather than break away"


def test_a_fraction_of_one_leaves_the_rod_alone():
    profile = tip_bend_stiffness_profile(3.0, num_edges=12, num_tip_edges=5, tip_fraction=1.0)

    assert profile == pytest.approx(np.full(12, 3.0))


def test_the_taper_cannot_run_past_the_rod():
    profile = tip_bend_stiffness_profile(1.0, num_edges=4, num_tip_edges=99, tip_fraction=0.5)

    assert profile.shape == (4,)
    assert profile[-1] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Rotational inertia
# --------------------------------------------------------------------------- #
def test_segment_inverse_inertia_is_the_solid_cylinder_value():
    """``12 / (m (3r^2 + L^2))`` about a diameter through the centre of mass."""
    mass, radius, length = 2.0e-6, 5.0e-4, 1.0e-2

    expected = 12.0 / (mass * (3.0 * radius**2 + length**2))

    assert segment_inverse_inertia(mass, radius, length) == pytest.approx(expected)


def test_a_slender_segment_is_dominated_by_its_length():
    """At 20x the radius the length term carries the inertia, so ``12 / (m L^2)``."""
    mass, radius, length = 2.0e-6, 5.0e-4, 1.0e-2

    slender = 12.0 / (mass * length**2)

    assert segment_inverse_inertia(mass, radius, length) == pytest.approx(slender, rel=0.01)


def test_a_lighter_segment_turns_more_easily():
    light = segment_inverse_inertia(1.0e-6, 5.0e-4, 1.0e-2)
    heavy = segment_inverse_inertia(2.0e-6, 5.0e-4, 1.0e-2)

    assert light > heavy


def test_the_identity_default_is_far_stiffer_than_the_real_wire():
    """Why the override matters: the shipped ``1.0`` is orders off for a guidewire.

    A 0.5 mm wire's segment barely resists turning, so its inverse inertia is
    enormous next to unity -- the identity default is a near-immovable frame.
    """
    assert segment_inverse_inertia(2.0e-6, 5.0e-4, 1.0e-2) > 1.0e6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_kg": 0.0, "radius_m": 5.0e-4, "segment_length_m": 1.0e-2},
        {"mass_kg": 1.0e-6, "radius_m": 0.0, "segment_length_m": 1.0e-2},
        {"mass_kg": 1.0e-6, "radius_m": 5.0e-4, "segment_length_m": -1.0},
    ],
)
def test_nonphysical_segments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        segment_inverse_inertia(**kwargs)


def test_coupled_solver_uses_device_inertia_by_default():
    assert CatheterRodSpec().physical_rotational_inertia is True


# --------------------------------------------------------------------------- #
# Containment probe
# --------------------------------------------------------------------------- #
def _straight_vessel(radius_m=0.004, samples=11, length_m=1.0):
    path = np.zeros((samples, 3))
    path[:, 0] = np.linspace(0.0, length_m, samples)
    return path, np.full(samples, radius_m)


def test_a_rod_on_the_axis_reports_its_clearance_as_negative_penetration():
    """Signed, so the probe distinguishes 'just inside' from 'comfortably inside'."""
    path, radii = _straight_vessel(radius_m=0.004)
    points = np.stack([np.linspace(0.1, 0.5, 5), np.zeros(5), np.zeros(5)], axis=1)

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert report["worst_penetration_mm"] == pytest.approx(-4.0)
    assert report["particles_outside"] == 0
    assert report["num_particles"] == 5


def test_a_rod_outside_the_wall_is_counted_and_measured():
    path, radii = _straight_vessel(radius_m=0.004)
    points = np.stack([np.linspace(0.1, 0.5, 5), np.full(5, 0.010), np.zeros(5)], axis=1)

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert report["worst_penetration_mm"] == pytest.approx(6.0)
    assert report["particles_outside"] == 5


def test_only_the_particles_past_the_wall_are_counted():
    path, radii = _straight_vessel(radius_m=0.004)
    offsets = np.array([0.0, 0.001, 0.002, 0.008, 0.020])
    points = np.stack([np.linspace(0.1, 0.5, 5), offsets, np.zeros(5)], axis=1)

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert report["particles_outside"] == 2
    assert report["worst_penetration_mm"] == pytest.approx(16.0)


def test_the_wall_is_interpolated_along_a_tapering_vessel():
    """Halfway down a 8 mm -> 2 mm taper the wall is at 5 mm, not at either end."""
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = np.array([0.008, 0.002])
    points = np.array([[0.5, 0.005, 0.0], [0.5, 0.006, 0.0]])

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.5)

    assert report["worst_penetration_mm"] == pytest.approx(1.0)
    assert report["particles_outside"] == 1


def test_a_rod_at_rest_length_reports_even_chords():
    path, radii = _straight_vessel()
    points = np.stack([np.linspace(0.0, 0.4, 5), np.zeros(5), np.zeros(5)], axis=1)

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert report["chord_min_pct"] == pytest.approx(100.0)
    assert report["chord_max_pct"] == pytest.approx(100.0)


def test_clustered_particles_show_up_as_a_chord_spread():
    """The damage containment-after-solve does: overlaps beside stretched gaps."""
    path, radii = _straight_vessel()
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.3, 0.0, 0.0]])

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert report["chord_min_pct"] == pytest.approx(10.0)
    assert report["chord_max_pct"] == pytest.approx(280.0)


def test_a_rod_at_rest_length_carries_no_excess_arc():
    path, radii = _straight_vessel()
    points = np.stack([np.linspace(0.0, 0.4, 5), np.zeros(5), np.zeros(5)], axis=1)

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert report["arc_length_mm"] == pytest.approx(400.0)
    assert report["rest_length_mm"] == pytest.approx(400.0)
    assert report["arc_excess_mm"] == pytest.approx(0.0)


def test_the_excess_is_the_arc_length_the_rod_should_not_have():
    """A 4-chord rod laid out over 500 mm of a 400 mm rest length."""
    path, radii = _straight_vessel()
    points = np.stack([np.linspace(0.0, 0.5, 5), np.zeros(5), np.zeros(5)], axis=1)

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert report["arc_length_mm"] == pytest.approx(500.0)
    assert report["arc_excess_mm"] == pytest.approx(100.0)


def test_redistributing_the_excess_leaves_the_total_alone():
    """Why the total is reported next to the percentages rather than instead.

    Both rods hold the same 100 mm of excess. The sweeps move it between chords,
    which moves the percentages a long way and the total not at all, so reading
    only the percentages can show an improvement that did not happen.
    """
    path, radii = _straight_vessel()
    even = np.stack([np.linspace(0.0, 0.5, 5), np.zeros(5), np.zeros(5)], axis=1)
    piled = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.5, 0.0, 0.0]])

    spread = containment_report(even, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)
    clumped = containment_report(piled, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert spread["arc_excess_mm"] == pytest.approx(clumped["arc_excess_mm"])
    assert spread["chord_max_pct"] == pytest.approx(125.0)
    assert clumped["chord_max_pct"] == pytest.approx(300.0)


def test_the_excess_is_comparable_across_segment_counts():
    """The percentages are not, which is the trap this figure exists to avoid.

    Two rods of the same 400 mm rest length, each holding the same 100 mm of
    excess piled into its last chord. Refining from 4 chords to 12 doubles the
    percentage that describes it, because the percentage is measured against a
    segment length that got three times shorter. The millimetres do not move.
    This is the misreading that made a finer rod look like a stretch regression.
    """
    path, radii = _straight_vessel()
    coarse = np.zeros((5, 3))
    coarse[:, 0] = np.cumsum([0.0, 0.1, 0.1, 0.1, 0.2])
    fine = np.zeros((13, 3))
    fine[:, 0] = np.cumsum([0.0] + [0.4 / 12.0] * 11 + [0.5 - 11.0 * 0.4 / 12.0])

    at_4 = containment_report(coarse, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)
    at_12 = containment_report(fine, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.4 / 12.0)

    assert at_4["arc_excess_mm"] == pytest.approx(100.0)
    assert at_12["arc_excess_mm"] == pytest.approx(100.0)
    assert at_4["chord_max_pct"] == pytest.approx(200.0)
    assert at_12["chord_max_pct"] == pytest.approx(400.0)


def test_a_duplicated_centerline_sample_does_not_divide_by_zero():
    """Extracted centerlines do carry repeated points."""
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = np.array([0.004, 0.004, 0.004])
    points = np.array([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.1)

    assert np.isfinite(report["worst_penetration_mm"])
    assert report["particles_outside"] == 0


@pytest.mark.parametrize(
    "path, radii, segment_length_m, message",
    [
        (np.zeros((1, 3)), np.zeros(1), 0.1, "centerline needs at least two"),
        (np.zeros((3, 3)), np.zeros(2), 0.1, "radii for 3 centerline samples"),
        (np.zeros((3, 3)), np.zeros(3), 0.0, "segment_length_m must be positive"),
    ],
)
def test_a_malformed_probe_request_is_rejected(path, radii, segment_length_m, message):
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    with pytest.raises(ValueError, match=message):
        containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=segment_length_m)


def test_the_containment_stage_is_the_scene_s_choice_by_default():
    assert containment_stage_override({}) is None


@pytest.mark.parametrize("value, expected", [("pre", "pre"), ("post", "post"), (" PRE ", "pre")])
def test_the_containment_stage_can_be_switched_for_a_measurement(value, expected):
    assert containment_stage_override({CONTAINMENT_STAGE_ENV_VAR: value}) == expected


@pytest.mark.parametrize("value", ["", "both", "pre-solve", "1"])
def test_a_mistyped_stage_does_not_decide_how_the_physics_runs(value):
    assert containment_stage_override({CONTAINMENT_STAGE_ENV_VAR: value}) is None


def test_damping_is_the_solver_cfg_s_choice_by_default():
    assert rod_damping_override({}) is None


@pytest.mark.parametrize("value, expected", [("1.0", 1.0), ("0", 0.0), (" 0.35 ", 0.35)])
def test_damping_can_be_switched_for_a_measurement(value, expected):
    assert rod_damping_override({DAMPING_ENV_VAR: value}) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "quasi-static", "1.5", "-0.1"])
def test_damping_outside_the_unit_range_is_ignored(value):
    """Above one the factor ``1 - damping`` would invert the velocity it scales."""
    assert rod_damping_override({DAMPING_ENV_VAR: value}) is None


def test_the_shipped_sweep_count_is_the_best_measured():
    """32, 64 and 128 were measured on the s0011 route; 128 won.

    At 32 the chords reach 100-112% over a short insertion and 121-122% over a
    driven route. Driven to a matched depth, 64 gave 100-114% and 128 gave
    100-109% at the same containment, so shipping less was leaving the better
    setting on the table.
    """
    assert CatheterRodSpec().containment_cleanup_iterations == 128


def test_the_sweep_count_is_the_specs_choice_by_default():
    assert cleanup_sweeps_override({}) is None


@pytest.mark.parametrize("value, expected", [("64", 64), ("128", 128), (" 96 ", 96), ("0", 0)])
def test_the_sweep_count_can_be_walked_for_a_measurement(value, expected):
    """``0`` disables the pass, so it is a real choice rather than an unset."""
    assert cleanup_sweeps_override({CLEANUP_SWEEPS_ENV_VAR: value}) == expected


@pytest.mark.parametrize("value", ["", "lots", "64.5", "-1"])
def test_a_sweep_count_that_is_not_a_count_is_ignored(value):
    assert cleanup_sweeps_override({CLEANUP_SWEEPS_ENV_VAR: value}) is None


def test_containment_and_cleanup_alternate_by_default():
    """Sequencing them let the sweeps overrule containment outright.

    One containment pass followed by 128 sweeps measured chords at a best-ever
    100-104% with 13 of 41 particles outside the lumen -- the sweeps equalize
    spacing by pushing particles through the wall, and they had the last word.
    """
    assert CatheterRodSpec().containment_cleanup_rounds > 1


def test_the_round_count_is_the_specs_choice_by_default():
    assert cleanup_rounds_override({}) is None


@pytest.mark.parametrize("value, expected", [("8", 8), ("1", 1), (" 16 ", 16)])
def test_the_round_count_can_be_walked_for_a_measurement(value, expected):
    """``1`` is a real choice: it restores the original sequencing for an A/B."""
    assert cleanup_rounds_override({CLEANUP_ROUNDS_ENV_VAR: value}) == expected


@pytest.mark.parametrize("value", ["", "lots", "8.5", "0", "-1"])
def test_a_round_count_below_one_is_ignored(value):
    """Zero rounds would skip the cleanup entirely rather than mean anything."""
    assert cleanup_rounds_override({CLEANUP_ROUNDS_ENV_VAR: value}) is None


def test_the_segment_count_is_the_specs_choice_by_default():
    assert segment_count_override({}) is None
    assert CatheterRodSpec().num_segments == DEFAULT_NUM_SEGMENTS


@pytest.mark.parametrize("value, expected", [("80", 80), ("120", 120), (" 60 ", 60), ("2", 2)])
def test_the_segment_count_can_be_walked_for_a_measurement(value, expected):
    assert segment_count_override({SEGMENT_COUNT_ENV_VAR: value}) == expected


@pytest.mark.parametrize("value", ["", "lots", "40.5", "1", "0", "-40"])
def test_a_segment_count_below_two_is_ignored(value):
    """One segment has no interior joint, so no bend constraint to refine."""
    assert segment_count_override({SEGMENT_COUNT_ENV_VAR: value}) is None


def test_the_rest_shape_is_straight_until_a_scale_is_asked_for():
    """The default the fold was measured under, pinned so the sweep has a zero.

    A straight rest shape in curved anatomy is what buckles the free distal
    end, so this is the configuration under test rather than a neutral one.
    """
    assert rest_curvature_override({}) is None
    assert CatheterRodSpec().rest_curvature_from_path is False


@pytest.mark.parametrize("value, expected", [("0.25", 0.25), ("0.5", 0.5), (" 0.75 ", 0.75), ("1", 1.0)])
def test_the_rest_curvature_can_be_walked_between_straight_and_the_path(value, expected):
    assert rest_curvature_override({REST_CURVATURE_ENV_VAR: value}) == pytest.approx(expected)


def test_a_zero_scale_is_a_real_answer_rather_than_an_unset_one():
    """Off has to be reachable from the environment, not only by unsetting.

    The sweep needs a control arm it can name, and ``None`` cannot say whether
    seeding was declined or simply never asked about.
    """
    assert rest_curvature_override({REST_CURVATURE_ENV_VAR: "0"}) == 0.0


@pytest.mark.parametrize("value", ["", "curvy", "1.5", "-0.25", "2"])
def test_a_scale_outside_the_unit_range_is_ignored(value):
    """Above one the rod would rather bend than the vessel holding it does."""
    assert rest_curvature_override({REST_CURVATURE_ENV_VAR: value}) is None


def test_an_unset_environment_leaves_the_shipped_rod_straight():
    """The sweep must not change what the default configuration does."""
    spec = CatheterRodSpec()
    scale = seeded_rest_curvature_scale(
        from_path=spec.rest_curvature_from_path,
        spec_scale=spec.rest_curvature_scale,
        override=None,
    )
    assert scale is None


@pytest.mark.parametrize("override", [0.25, 0.5, 1.0])
def test_the_override_enables_seeding_without_flipping_the_spec(override):
    """Otherwise the sweep needs a code edit before the environment can help.

    ``rest_curvature_from_path`` ships off, so an override that only scaled an
    already-enabled seeding would be inert on exactly the configuration the
    fold was measured in.
    """
    assert seeded_rest_curvature_scale(from_path=False, spec_scale=1.0, override=override) == override


def test_a_zero_override_declines_seeding_a_spec_asked_for():
    """The control arm has to beat the spec, or there is nothing to compare to."""
    assert seeded_rest_curvature_scale(from_path=True, spec_scale=1.0, override=0.0) is None


def test_a_spec_that_asks_for_seeding_keeps_its_own_scale():
    assert seeded_rest_curvature_scale(from_path=True, spec_scale=0.4, override=None) == pytest.approx(0.4)


def test_the_override_beats_the_specs_scale():
    assert seeded_rest_curvature_scale(from_path=True, spec_scale=0.4, override=0.9) == pytest.approx(0.9)


def test_a_straight_rod_has_no_kinks_and_no_finite_percentile():
    path, radii = _straight_vessel()
    points = np.stack([np.linspace(0.0, 0.4, 9), np.zeros(9), np.zeros(9)], axis=1)

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.05)

    assert report["kinked_nodes"] == 0
    assert report["bend_radius_p05_mm"] == float("inf")
    assert report["num_bend_nodes"] == 7
    assert report["first_kinked_node"] == -1
    assert report["last_kinked_node"] == -1


def test_a_buckled_section_reports_as_one_block():
    """Nine adjacent folded nodes, which is what the live rod shows.

    A coil swallows insertion because an accordion transmits no axial force,
    and it is invisible to every other field here: it sits inside the lumen so
    penetration stays negative, and folding does not change arc length. The
    span is what says it is a coil rather than nine unrelated creases.
    """
    path, radii = _straight_vessel(radius_m=0.5, length_m=0.5)
    # The default hairpin sits at 90 and would widen the span on its own, so it
    # is placed inside the stretch the concertina overwrites.
    points = _rod_with_one_hairpin(at=110)
    # A concertina: barely any axial progress per node, alternating laterally.
    for offset, node in enumerate(range(105, 116)):
        points[node] = points[104] + np.array([0.0003 * offset, 0.0, 0.0005 * (-1.0) ** offset])

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.0025)

    assert report["kinked_nodes"] >= 6
    assert report["last_kinked_node"] - report["first_kinked_node"] < 20


def test_scattered_creases_report_a_span_covering_the_rod():
    """The other shape the same count can take."""
    path, radii = _straight_vessel(radius_m=0.5, length_m=0.5)
    points = _rod_with_one_hairpin()
    for node in (10, 30, 50, 70, 110):
        points[node] = points[node - 2] + np.array([0.0, 0.0, 0.08 * 0.0025])

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.0025)

    assert report["kinked_nodes"] >= 5
    assert report["last_kinked_node"] - report["first_kinked_node"] > 80


def _rod_with_one_hairpin(segment_m=0.0025, nodes=121, curve_radius_m=0.2, at=90):
    """A gently curved rod of realistic size, folded back at a single node.

    Sized like the shipped rod because both new fields depend on it. A fold
    bottoms out at half a segment length, so 50 mm test segments could never
    reach a 10 mm threshold, and one kink in a ten-node rod is 10% of it and so
    moves a 5th percentile that the same kink in 119 nodes leaves alone.

    The fold carries a small lateral offset on purpose. Three exactly collinear
    samples have no circle through them and :func:`bend_radii_m` reports ``inf``,
    so a mathematically perfect doubling-back reads as no bend at all.
    """
    sweep = float(segment_m) * (nodes - 1) / float(curve_radius_m)
    angle = np.linspace(0.0, sweep, nodes)
    points = np.stack([curve_radius_m * np.sin(angle), curve_radius_m * (1.0 - np.cos(angle)), np.zeros(nodes)], axis=1)
    points[at + 1] = points[at - 1] + np.array([0.0, 0.0, 0.08 * segment_m])
    return points


def test_one_hairpin_leaves_the_percentile_out_at_anatomical_scale():
    """The reading that says "one localized fold" rather than "a kinked rod".

    This is the distinction the minimum alone cannot draw, and the reason these
    two fields exist. A handful of folded nodes pin the minimum near the floor
    while the rest of the rod stays gently curved, so the count stays in the low
    single digits and the percentile sits out where the curve is.
    """
    path, radii = _straight_vessel(radius_m=0.5, length_m=0.5)
    points = _rod_with_one_hairpin()

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.0025)

    assert report["min_bend_radius_mm"] < 10.0
    assert 1 <= report["kinked_nodes"] <= 3
    assert report["bend_radius_p05_mm"] > 100.0


def test_a_rod_kinked_throughout_moves_the_count_and_the_percentile_together():
    """The other case, which reads identically on the minimum alone."""
    path, radii = _straight_vessel()
    # A tight zigzag: every interior node turns hard.
    points = np.zeros((13, 3))
    points[:, 0] = np.linspace(0.0, 0.06, 13)
    points[1::2, 1] = 0.004

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.005)

    assert report["kinked_nodes"] == report["num_bend_nodes"]
    assert report["bend_radius_p05_mm"] < 10.0


def test_the_kink_threshold_is_in_millimetres_not_segment_lengths():
    """So the same fold counts at 40 segments and at 120.

    A threshold expressed in segment lengths would move with the
    discretization, and the whole question is whether the wire is tighter than
    the vessel it is in.
    """
    path, radii = _straight_vessel(radius_m=0.5, length_m=0.5)
    points = _rod_with_one_hairpin()

    lenient = containment_report(
        points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.0025, kink_radius_m=0.0002
    )
    strict = containment_report(
        points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.0025, kink_radius_m=0.010
    )

    assert lenient["kinked_nodes"] == 0
    assert strict["kinked_nodes"] >= 1


def test_an_anatomical_curve_is_not_counted_as_a_kink():
    """The default threshold has to clear the tightest curve on the route.

    The s0011 centerline bottoms out at 13.1 mm and the rod's seeded shape at
    14.4 mm, so a 10 mm default leaves headroom without needing the anatomy to
    be gentle.
    """
    path, radii = _straight_vessel()
    points = _arc(radius_m=0.014, samples=12)

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.002)

    assert report["min_bend_radius_mm"] == pytest.approx(14.0, abs=0.5)
    assert report["kinked_nodes"] == 0


def _realized_tip_turn_rad(polyline_turn_rad, num_tip_edges, num_straight_edges=6):
    """Turn of the tip polyline, built the way the solve builds it.

    Walks the frame ladder the rest curvature describes and then takes the
    segment directions as the averages of adjacent frames, which is where the
    lost half-hinge comes from. Returns the angle between the last straight
    segment and the last tip segment -- the quantity a camera sees and a policy
    is scored on, as opposed to the total frame rotation.
    """
    component = tip_bend_rest_component(polyline_turn_rad, num_tip_edges)
    # Frame k is rotated from the previous by phi = 2 asin(component) about the
    # bend axis, in the plane the bend acts in. Only the in-plane angle matters,
    # so the ladder reduces to a running sum.
    phi = 2.0 * math.asin(min(1.0, max(-1.0, component)))
    frame_angles = [0.0] * (num_straight_edges + 1)
    for _ in range(num_tip_edges):
        frame_angles.append(frame_angles[-1] + phi)
    # A segment points along the average of the frames at its two ends.
    segment_angles = [0.5 * (frame_angles[i] + frame_angles[i + 1]) for i in range(len(frame_angles) - 1)]
    return segment_angles[-1] - segment_angles[num_straight_edges - 1]


@pytest.mark.parametrize("requested", [0.25, 0.5, 1.0, 1.5, 2.5])
@pytest.mark.parametrize("edges", [1, 2, 4, 10, 20])
def test_the_tip_polyline_turns_by_exactly_what_was_asked_for(requested, edges):
    """The property the mapping exists for, measured rather than restated."""
    if requested > max_faithful_tip_bend_rad(edges):
        pytest.skip("beyond the range the mapping reproduces exactly")

    assert _realized_tip_turn_rad(requested, edges) == pytest.approx(requested, rel=1e-9, abs=1e-12)


def test_the_obvious_reading_over_bends_by_nearly_a_factor_of_two():
    """Why this changed. ``angle / n`` per edge is the natural guess and wrong.

    The shipped ceiling of 1.5 rad realized 2.86 rad, because the polyline picks
    up ``(2n - 1)/n`` of the requested turn when the request is divided by ``n``
    instead of ``2n - 1``.
    """
    edges = 10
    phi = 2.0 * math.asin(1.5 / edges)

    assert (edges - 0.5) * phi == pytest.approx(2.861, abs=0.001)
    assert (edges - 0.5) * phi / 1.5 == pytest.approx((2 * edges - 1) / edges, abs=0.02)


def test_the_mapping_round_trips_through_the_realized_turn():
    for edges in (1, 3, 10):
        component = tip_bend_rest_component(0.8, edges)
        assert tip_bend_polyline_turn_rad(component, edges) == pytest.approx(0.8)


def test_a_zero_request_leaves_the_rest_shape_alone():
    """So sending zero every step is genuinely inert, not a small bend."""
    for edges in (1, 10):
        assert tip_bend_rest_component(0.0, edges) == 0.0


def test_the_component_stays_bounded_so_a_big_request_cannot_go_undefined():
    """The old form divided by ``n`` and went out of ``asin`` domain past it.

    A one-edge tip band reached that at 1 rad, which is inside the shipped
    1.5 rad ceiling, so the hazard was live once the band became sweepable.
    """
    for requested in (1.0, 5.0, 100.0, -100.0):
        for edges in (1, 10):
            assert abs(tip_bend_rest_component(requested, edges)) <= 1.0


def test_steering_is_symmetric_about_straight():
    for edges in (1, 4, 10):
        assert tip_bend_rest_component(-0.9, edges) == pytest.approx(-tip_bend_rest_component(0.9, edges))


def test_the_faithful_range_is_generous_at_ten_edges_and_tight_at_one():
    assert max_faithful_tip_bend_rad(10) == pytest.approx(29.845, abs=0.001)
    assert max_faithful_tip_bend_rad(1) == pytest.approx(math.pi / 2.0)
    # The shipped 1.5 rad ceiling fits at ten edges with room to spare and at
    # one edge with 0.07 rad to spare, so a tip-band sweep down to a single
    # edge stays inside the exact range. Worth pinning: it is close enough that
    # raising the ceiling would quietly leave it.
    assert max_faithful_tip_bend_rad(10) > 1.5
    assert 1.5 < max_faithful_tip_bend_rad(1) < 1.58


@pytest.mark.parametrize("edges", [0, -1])
def test_a_tip_band_with_no_edges_is_rejected(edges):
    with pytest.raises(ValueError):
        tip_bend_rest_component(1.0, edges)
    with pytest.raises(ValueError):
        max_faithful_tip_bend_rad(edges)


def test_a_finer_tip_band_needs_a_smaller_per_edge_curvature():
    """Same requested turn spread over more edges, so each does less of it."""
    coarse = tip_bend_rest_component(1.0, 4)
    fine = tip_bend_rest_component(1.0, 20)

    assert fine < coarse
    assert _realized_tip_turn_rad(1.0, 4) == pytest.approx(_realized_tip_turn_rad(1.0, 20))


def test_the_bend_stiffness_is_the_compensated_value_by_default():
    assert bend_stiffness_override({}) is None


@pytest.mark.parametrize("raw,expected", [("3", 3.0), ("10.5", 10.5), (" 30 ", 30.0), ("0.9", 0.9)])
def test_the_bend_stiffness_can_be_swept_past_the_compensated_value(raw, expected):
    assert bend_stiffness_override({BEND_STIFFNESS_ENV_VAR: raw}) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["0", "-1", "", "stiff", "1,5"])
def test_a_non_positive_or_malformed_stiffness_is_ignored(raw):
    """Zero would remove the constraint being measured rather than soften it."""
    assert bend_stiffness_override({BEND_STIFFNESS_ENV_VAR: raw}) is None


def test_the_tip_band_is_the_solver_cfgs_choice_by_default():
    assert tip_edge_count_override({}) is None


@pytest.mark.parametrize("value, expected", [("10", 10), ("4", 4), (" 2 ", 2), ("0", 0)])
def test_the_tip_band_can_be_narrowed_to_reach_the_fold(value, expected):
    """Zero included: seeding the whole rod is a measurement worth being able
    to take, even though it costs the steerable pre-bend."""
    assert tip_edge_count_override({TIP_EDGES_ENV_VAR: value}) == expected


@pytest.mark.parametrize("value", ["", "ten", "4.5", "-1"])
def test_a_negative_tip_band_is_ignored(value):
    assert tip_edge_count_override({TIP_EDGES_ENV_VAR: value}) is None


#: Particle the curvature probe reports folded, out of
#: :data:`DEFAULT_NUM_SEGMENTS`. Constant across every sample of the s0011 run
#: in ``runs/endoluminal_navigation/20260910_093406``, at 1.3 mm against the
#: 1.26 mm floor for a 2.527 mm chord.
FOLDED_NODE = 118


def test_no_useful_tip_band_reaches_the_fold():
    """Why narrowing the band is not on its own a fix.

    ``_seed_shaft_rest_darboux`` leaves the trailing ``tip_num_edges`` alone,
    and the bend at particle ``i`` is carried by edges ``i - 1`` and ``i``. The
    fold is two particles from the distal end, so seeding reaches it only for a
    band of one edge or fewer -- which is to say only by removing the band. A
    curvature sweep that keeps any steerable tip cannot touch this node.
    """
    reaches_the_fold = [band for band in range(DEFAULT_NUM_SEGMENTS) if DEFAULT_NUM_SEGMENTS - band > FOLDED_NODE]

    assert reaches_the_fold == [0, 1]


def test_one_edge_is_the_only_band_that_both_steers_and_reaches_the_fold():
    """The value the sweep should actually carry, and why it is not zero.

    A band of zero leaves no edge satisfying ``e >= num_edges - tip_num_edges``,
    so the steering kernel writes the baseline everywhere and ``set_tip_bend``
    stops doing anything. One edge is the narrowest band that still steers.

    That it can be this narrow at all is what the bend baseline bought: the
    kernel writes ``baseline + angle`` rather than overwriting, so the tip's
    rest curvature no longer has to be zero for a steering command to land on
    it. The exclusion could once be justified as protecting a floppy tip; now
    it only decides how much of the rod keeps a straight rest shape.
    """
    steers = [band for band in range(DEFAULT_NUM_SEGMENTS) if band >= 1]
    reaches_the_fold = [band for band in range(DEFAULT_NUM_SEGMENTS) if DEFAULT_NUM_SEGMENTS - band > FOLDED_NODE]

    assert sorted(set(steers) & set(reaches_the_fold)) == [1]
    assert tip_edge_count_override({TIP_EDGES_ENV_VAR: "1"}) == 1


def test_containment_is_two_sided_by_default():
    """One-sided containment is what let the bend relocate instead of leaving.

    Sweeping harder only chooses whether the tip or the shaft carries the excess
    arc length, because nothing inside the lumen prefers the axis. The deadband
    keeps the inner half free so the shape there is still the solve's answer.
    """
    spec = CatheterRodSpec()

    assert spec.containment_interior_deadband == pytest.approx(0.5)
    assert spec.containment_interior_stiffness > 0.0


def test_the_interior_pull_is_the_specs_choice_by_default():
    assert interior_containment_override({}) is None


@pytest.mark.parametrize(
    "value, expected",
    [("0.5,0.25", (0.5, 0.25)), (" 0.6 , 0.1 ", (0.6, 0.1)), ("1,0", (1.0, 0.0)), ("0,1", (0.0, 1.0))],
)
def test_the_interior_pull_can_be_walked_for_a_measurement(value, expected):
    assert interior_containment_override({INTERIOR_CONTAINMENT_ENV_VAR: value}) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "0.5", "0.5,0.25,0.1", "half,0.25", "1.5,0.25", "0.5,-0.1"])
def test_an_unusable_interior_pull_is_ignored(value):
    """A mistyped diagnostic must not decide how the physics runs.

    Both halves are required because either alone is meaningless: a deadband
    with no stiffness does nothing, and a stiffness with the deadband at the
    wall is the one-sided behaviour it is meant to replace.
    """
    assert interior_containment_override({INTERIOR_CONTAINMENT_ENV_VAR: value}) is None


# --------------------------------------------------------------------------- #
# Centerline projection
# --------------------------------------------------------------------------- #
_STRAIGHT_PATH = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_the_nearest_point_is_the_foot_of_the_perpendicular():
    closest, edge, fraction = nearest_on_polyline(np.array([[0.25, 0.3, 0.0]]), _STRAIGHT_PATH)

    assert closest[0] == pytest.approx([0.25, 0.0, 0.0])
    assert edge[0] == 0
    assert fraction[0] == pytest.approx(0.25)


def test_the_nearest_point_clamps_to_the_ends():
    """Past the end of the path the closest point is the end, not an extrapolation."""
    closest, _, fraction = nearest_on_polyline(np.array([[2.0, 0.0, 0.0]]), _STRAIGHT_PATH)

    assert closest[0] == pytest.approx([1.0, 0.0, 0.0])
    assert fraction[0] == pytest.approx(1.0)


def test_a_bend_picks_the_owning_edge():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    _, edge, _ = nearest_on_polyline(np.array([[1.2, 0.7, 0.0]]), path)

    assert edge[0] == 1


def test_a_zero_length_edge_does_not_divide_by_zero():
    """Extracted centerlines do carry repeated points."""
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    closest, _, _ = nearest_on_polyline(np.array([[0.5, 0.1, 0.0]]), path)

    assert closest[0] == pytest.approx([0.5, 0.0, 0.0])


def test_a_polyline_needs_two_samples():
    with pytest.raises(ValueError, match="at least two samples"):
        nearest_on_polyline(np.zeros((1, 3)), np.zeros((1, 3)))


def test_a_scene_without_a_centerline_has_nothing_to_report():
    handle = CatheterRodHandle(CatheterRodSpec(initial_path_world_m=None, lumen_radii_m=None))

    assert handle.report_containment(np.zeros((4, 3))) is None


def test_a_vessel_needs_a_twin():
    assert CatheterRodSpec(patient_twin_manifest=None).wants_vessel is False
    assert CatheterRodSpec(patient_twin_manifest="twin.yaml").wants_vessel is True
    assert CatheterRodSpec(patient_twin_manifest="twin.yaml", vessel_enabled=False).wants_vessel is False


# --------------------------------------------------------------------------- #
# Handle lifecycle
# --------------------------------------------------------------------------- #
def test_reading_the_particle_range_too_early_explains_why():
    handle = CatheterRodHandle(CatheterRodSpec())

    with pytest.raises(RuntimeError, match="MODEL_INIT"):
        _ = handle.particle_range


def test_no_vessel_when_no_twin_is_configured():
    handle = CatheterRodHandle(CatheterRodSpec(patient_twin_manifest=None))

    assert handle.vessel is None


# --------------------------------------------------------------------------- #
# Newton config
# --------------------------------------------------------------------------- #
class FakeNewtonCfg:
    def __init__(self, solver_cfg=None, use_cuda_graph=True):
        self.solver_cfg = solver_cfg
        self.use_cuda_graph = use_cuda_graph


class FakeSolverCfg:
    def __init__(self, **fields):
        self.__dict__.update(fields)
        self.tip_num_edges = fields.get("tip_num_edges", 10)


class FakeMJWarpSolverCfg:
    pass


class FakeCoupledSolverCfg:
    """Mirrors the real cfg's defaults so the wiring's choices stay visible."""

    def __init__(self, **fields):
        self.coupling_mode = "one_way"
        self.drive_body_index = None
        self.drive_body_name = None
        self.drive_reaction_relaxation = 1.0
        self.soft_contact_ke = 5.0e3
        self.soft_contact_kd = 1.0e2
        self.soft_contact_mu = 0.5
        self.__dict__.update(fields)


@pytest.fixture
def stub_isaac(monkeypatch):
    """Stub the Isaac Lab and solver-config modules the wiring imports."""
    newton_physics = types.ModuleType("isaaclab_newton.physics")
    newton_physics.NewtonCfg = FakeNewtonCfg
    newton_physics.MJWarpSolverCfg = FakeMJWarpSolverCfg
    newton_physics.NewtonManager = SimpleNamespace(_builder=None, register_callback=None)
    newton_pkg = types.ModuleType("isaaclab_newton")

    integration = types.ModuleType("catheter_vasculature_solver.isaaclab_integration")
    integration.XPBDRodSolverCfg = FakeSolverCfg
    integration.CoupledMJWarpXPBDRodSolverCfg = FakeCoupledSolverCfg

    for name, module in (
        ("isaaclab_newton", newton_pkg),
        ("isaaclab_newton.physics", newton_physics),
        ("catheter_vasculature_solver.isaaclab_integration", integration),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(newton_physics=newton_physics, integration=integration)


def test_solver_cfg_carries_the_scene_geometry(stub_isaac):
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec(length_m=0.4, num_segments=40, radius_m=0.001))

    assert cfg.num_segments == 40
    assert cfg.segment_length == pytest.approx(0.01)
    assert cfg.radius == pytest.approx(0.001)


def test_static_collision_and_track_stay_off(stub_isaac):
    """The deformable centerline supplies containment; two walls would fight."""
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec())

    assert cfg.collision_enabled is False
    assert cfg.track_enabled is False


def test_state_sync_stays_on_so_the_rod_starts_in_the_patient(stub_isaac):
    """It is the only route the centerline has into the solver.

    The rod builds itself as a straight rod along +X and cannot be constructed
    from a polyline, so the seeded Newton buffer reaching it on the first step
    is what puts the catheter in the vessel rather than out in the room.
    """
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec())

    assert getattr(cfg, "sync_from_state", True) is True


def test_solver_overrides_win(stub_isaac):
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec(solver_overrides={"bend_stiffness": 0.5}))

    assert cfg.bend_stiffness == pytest.approx(0.5)


@pytest.mark.parametrize(
    "spec",
    [
        CatheterRodSpec(patient_twin_manifest="twin.yaml"),
        CatheterRodSpec(patient_twin_manifest=None),
        CatheterRodSpec(patient_twin_manifest=None, rigid_bodies_enabled=True),
    ],
    ids=["vessel", "bare_rod", "arm"],
)
def test_cuda_graph_is_disabled_for_every_catheter_scene(stub_isaac, spec):
    """The rod managers raise on capture, so a bare rod may not ask for it either.

    The bare rod is the case worth covering: a vessel resizes contact scratch and
    an arm's contact counts vary with its pose, so both were already excluded on
    their own merits, and only this one relies on the latch.
    """
    from i4h_arena.medical.newton_catheter_physics import newton_physics_cfg

    assert newton_physics_cfg(spec).use_cuda_graph is False


def test_physics_cfg_does_not_set_class_type(stub_isaac):
    """NewtonCfg derives class_type from solver_cfg and rejects a manual value."""
    from i4h_arena.medical.newton_catheter_physics import newton_physics_cfg

    cfg = newton_physics_cfg(CatheterRodSpec())

    assert not hasattr(cfg, "class_type") or cfg.class_type is None
    assert cfg.solver_cfg is not None


# --------------------------------------------------------------------------- #
# Coupled solver selection
# --------------------------------------------------------------------------- #
def test_a_particle_only_scene_stays_on_the_rod_solver(stub_isaac):
    """Without rigid bodies there is nothing for MJWarp to integrate."""
    from i4h_arena.medical.newton_catheter_physics import newton_solver_cfg

    cfg = newton_solver_cfg(CatheterRodSpec(rigid_bodies_enabled=False))

    assert isinstance(cfg, FakeSolverCfg)


@pytest.mark.parametrize("segments", [20, 40, 80, 120])
def test_refinement_keeps_the_material_multiplier(stub_isaac, segments):
    from i4h_arena.medical.newton_catheter_physics import coupled_solver_cfg, rod_solver_cfg

    spec = CatheterRodSpec(rigid_bodies_enabled=True, num_segments=segments)
    assert not hasattr(rod_solver_cfg(spec), "bend_stiffness")
    assert not hasattr(coupled_solver_cfg(spec).rod_solver_cfg, "bend_stiffness")


def test_an_explicit_bend_stiffness_still_wins(stub_isaac):
    """An explicitly configured material multiplier reaches the solver."""
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    spec = CatheterRodSpec(num_segments=120, solver_overrides={"bend_stiffness": 0.25})

    assert rod_solver_cfg(spec).bend_stiffness == pytest.approx(0.25)


def test_an_arm_switches_the_scene_to_the_coupled_solver(stub_isaac):
    """The rod solver has no rigid integrator, so an articulation needs MJWarp beside it."""
    from i4h_arena.medical.newton_catheter_physics import newton_solver_cfg

    cfg = newton_solver_cfg(CatheterRodSpec(rigid_bodies_enabled=True))

    assert isinstance(cfg, FakeCoupledSolverCfg)
    assert isinstance(cfg.rigid_solver_cfg, FakeMJWarpSolverCfg)


def test_the_coupled_cfg_nests_the_same_rod_geometry(stub_isaac):
    """An arm must not quietly change the catheter the scene declared."""
    from i4h_arena.medical.newton_catheter_physics import coupled_solver_cfg

    spec = CatheterRodSpec(rigid_bodies_enabled=True, length_m=0.4, num_segments=40, radius_m=0.001)
    cfg = coupled_solver_cfg(spec)

    assert cfg.rod_solver_cfg.num_segments == 40
    assert cfg.rod_solver_cfg.segment_length == pytest.approx(0.01)
    assert cfg.rod_solver_cfg.radius == pytest.approx(0.001)


def test_coupling_stays_one_way_when_no_drive_body_is_named(stub_isaac):
    """An arm alone only pushes the catheter; feeling it back is opt-in.

    Two-way needs a specific body to load, so a scene that has not said which
    one holds the wire gets the cheaper, unconditionally stable direction.
    """
    from i4h_arena.medical.newton_catheter_physics import coupled_solver_cfg

    cfg = coupled_solver_cfg(CatheterRodSpec(rigid_bodies_enabled=True))

    assert cfg.coupling_mode == "one_way"
    assert getattr(cfg, "drive_body_name", None) is None


def test_naming_the_drive_body_selects_two_way(stub_isaac):
    """The one switch that makes the holder feel the catheter."""
    from i4h_arena.medical.newton_catheter_physics import coupled_solver_cfg

    cfg = coupled_solver_cfg(CatheterRodSpec(rigid_bodies_enabled=True, drive_body_name="TCP"))

    assert cfg.coupling_mode == "two_way"
    assert cfg.drive_body_name == "TCP"


def test_the_drive_body_is_passed_by_name_not_index(stub_isaac):
    """The solver resolves it against the builder's labels.

    Sending an index instead would keep pointing at whatever link happens to
    sit at that position if the scene's body order ever changed, which is a
    silent wrong-body failure rather than a startup error.
    """
    from i4h_arena.medical.newton_catheter_physics import coupled_solver_cfg

    cfg = coupled_solver_cfg(CatheterRodSpec(rigid_bodies_enabled=True, drive_body_name="TCP"))

    assert getattr(cfg, "drive_body_index", None) is None


def test_a_drive_body_without_rigid_bodies_is_refused():
    """A rod-only scene has nothing to push on, so this is a scene-wiring bug."""
    with pytest.raises(ValueError, match="rigid_bodies_enabled is False"):
        CatheterRodSpec(drive_body_name="TCP")


def test_soft_contact_overrides_win(stub_isaac):
    from i4h_arena.medical.newton_catheter_physics import coupled_solver_cfg

    cfg = coupled_solver_cfg(
        CatheterRodSpec(rigid_bodies_enabled=True, soft_contact_overrides={"soft_contact_mu": 0.1})
    )

    assert cfg.soft_contact_mu == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# MODEL_INIT ordering
# --------------------------------------------------------------------------- #
class FakeBuilder:
    def __init__(self):
        self.particle_count = 0


@pytest.fixture
def stub_model_init(monkeypatch, stub_isaac):
    """Record the order of builder population and rod registration."""
    calls: list[str] = []
    recorded: dict = {}
    builder = FakeBuilder()
    stub_isaac.newton_physics.NewtonManager._builder = builder

    def add_catheter_rod_to_builder(passed_builder, config, *, positions, start, direction, num_envs):
        assert passed_builder is builder
        calls.append("add_particles")
        recorded["positions"] = positions
        return SimpleNamespace(offset=0, count=(config.num_points) * num_envs, num_envs=num_envs)

    def rod_config_from_solver_cfg(solver_cfg, *, device):
        calls.append(f"rod_config:{device}")
        return SimpleNamespace(num_points=solver_cfg.num_segments + 1, device=device)

    registered: dict = {}

    class FakeRodManager:
        @staticmethod
        def register_rod(particle_range, *, rod=None):
            calls.append("register_rod")
            registered["particle_range"] = particle_range
            registered["rod"] = rod

    stub_isaac.integration.add_catheter_rod_to_builder = add_catheter_rod_to_builder
    stub_isaac.integration.rod_config_from_solver_cfg = rod_config_from_solver_cfg
    stub_isaac.integration.NewtonXPBDRodManager = FakeRodManager

    solver_module = types.ModuleType("catheter_vasculature_solver")

    def CathRodSolver(config, **kwargs):  # noqa: N802 - mirrors the real class name
        calls.append("build_rod")
        # A workspace stands in because the wiring writes the rod's rotational
        # inertia over the solver's identity default, and refuses to run blind.
        # The pinned root's zero inverse mass is part of what it reads past.
        workspace = SimpleNamespace(
            inv_masses=np.array([0.0] + [5.0e5] * 40, dtype=np.float32),
            inv_inertia_local_diag=(1.0, 1.0, 1.0),
        )
        return SimpleNamespace(config=config, kwargs=kwargs, _ws=workspace)

    def initialize_rod_state(solver, positions):
        calls.append("initialize_state")
        recorded["solver_positions"] = positions.copy()

    monkeypatch.setattr("i4h_arena.medical.catheter_initialization.initialize_rod_state", initialize_rod_state)
    solver_module.CathRodSolver = CathRodSolver
    monkeypatch.setitem(sys.modules, "catheter_vasculature_solver", solver_module)
    return SimpleNamespace(calls=calls, registered=registered, builder=builder, recorded=recorded)


def test_particles_are_added_before_the_rod_is_registered(stub_model_init):
    """Registering a rod whose particles do not exist yet leaves it driving nothing."""
    handle = CatheterRodHandle(CatheterRodSpec(num_envs=2, num_segments=8))

    handle._on_model_init()

    calls = stub_model_init.calls
    assert calls.index("add_particles") < calls.index("register_rod")
    assert calls.index("build_rod") < calls.index("initialize_state") < calls.index("register_rod")
    np.testing.assert_array_equal(stub_model_init.recorded["positions"], stub_model_init.recorded["solver_positions"])


def test_registration_passes_the_particle_range_and_the_rod(stub_model_init):
    handle = CatheterRodHandle(CatheterRodSpec(num_envs=3, num_segments=8))

    handle._on_model_init()

    registered = stub_model_init.registered
    assert registered["particle_range"].num_envs == 3
    assert registered["particle_range"].count == 9 * 3
    assert registered["rod"] is handle.rod


def test_the_rod_config_is_built_on_the_requested_device(stub_model_init):
    handle = CatheterRodHandle(CatheterRodSpec(device="cuda:1"))

    handle._on_model_init()

    assert "rod_config:cuda:1" in stub_model_init.calls


def test_the_rod_can_be_given_its_own_rotational_inertia(stub_model_init):
    """The solver ships an identity diagonal for every particle, which is wrong
    for a 0.5 mm wire even though correcting it changed no containment number."""
    spec = CatheterRodSpec(
        length_m=0.4,
        num_segments=40,
        radius_m=0.0005,
        physical_rotational_inertia=True,
    )
    handle = CatheterRodHandle(spec)

    handle._on_model_init()

    written = handle.rod._ws.inv_inertia_local_diag
    expected = segment_inverse_inertia(1.0 / 5.0e5, spec.radius_m, spec.segment_length_m)
    polar_inverse = 2.0 / ((1.0 / 5.0e5) * spec.radius_m**2)
    assert tuple(written) == pytest.approx((expected, expected, polar_inverse))
    assert written[0] > 1.0


def test_legacy_inertia_is_available_for_comparisons(stub_model_init):
    handle = CatheterRodHandle(CatheterRodSpec(physical_rotational_inertia=False))

    handle._on_model_init()

    assert handle.rod._ws.inv_inertia_local_diag == (1.0, 1.0, 1.0)


def test_a_straight_rod_seeds_the_same_world_positions_in_newton_and_the_solver(stub_model_init):
    spec = CatheterRodSpec(
        origin_world_m=(0.2, 0.3, 0.4),
        track_direction_world=(0.0, 1.0, 0.0),
        initial_path_world_m=None,
    )
    handle = CatheterRodHandle(spec)

    handle._on_model_init()

    positions = stub_model_init.recorded["positions"]
    np.testing.assert_array_equal(positions, stub_model_init.recorded["solver_positions"])
    np.testing.assert_allclose(positions[0], spec.origin_world_m)
    np.testing.assert_allclose(positions[-1], (0.2, 0.7, 0.4))


def test_the_vessel_path_seeds_the_rod_shape(stub_model_init):
    """The centerline sets the starting shape once, instead of being replayed
    into the solver every step, which would overwrite the physics result."""
    path = tuple((float(index) * 0.05, 0.0, 0.0) for index in range(8))
    spec = CatheterRodSpec(num_segments=8, length_m=0.2, initial_path_world_m=path)
    handle = CatheterRodHandle(spec)

    handle._on_model_init()

    positions = stub_model_init.recorded["positions"]
    assert positions is not None
    assert positions.shape == (spec.num_points, 3)
    # Sampled along the path, so the seeded rod spans the requested length.
    span = float(np.linalg.norm(positions[-1] - positions[0]))
    assert span == pytest.approx(spec.length_m, rel=1e-3)


def test_a_missing_builder_is_reported_not_silently_skipped(stub_model_init, stub_isaac):
    stub_isaac.newton_physics.NewtonManager._builder = None
    handle = CatheterRodHandle(CatheterRodSpec())

    with pytest.raises(RuntimeError, match="ModelBuilder"):
        handle._on_model_init()


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
class _RecordingRod:
    def __init__(self) -> None:
        self.reset_with: list[object] = []

    def reset(self, env_ids=None) -> None:
        self.reset_with.append(env_ids)


def _handle_with_rod() -> tuple[CatheterRodHandle, _RecordingRod]:
    handle = CatheterRodHandle(CatheterRodSpec())
    rod = _RecordingRod()
    handle._rod = rod
    return handle, rod


def test_reset_before_the_rod_exists_is_a_no_op():
    handle = CatheterRodHandle(CatheterRodSpec())

    handle.reset(None)  # must not raise; MODEL_INIT has not fired yet


def test_reset_forwards_every_environment_as_none():
    handle, rod = _handle_with_rod()

    handle.reset(None)

    assert rod.reset_with == [None]


def test_reset_forwards_the_index_tensor_untouched():
    """IsaacLab hands reset the device tensor it builds, and the solver takes it.

    Converting here instead would put a copy at each caller of a solver that
    already accepts device buffers at every entry point.
    """
    torch = pytest.importorskip("torch")
    handle, rod = _handle_with_rod()
    env_ids = torch.tensor([1, 0], dtype=torch.int32)

    handle.reset(env_ids)

    assert rod.reset_with[0] is env_ids


# --------------------------------------------------------------------------- #
# Installed-solver contract
# --------------------------------------------------------------------------- #
# Every test above runs against stubs, which is what keeps them on CPU and off
# Isaac. The cost is that they pass just as well when the checked-out solver
# does not have two-way coupling at all: the stub supplies the attributes the
# real package would. third_party/setup.sh pins the solver to a ref and will
# `checkout -f` back onto it, so the checkout can move under us without any
# import breaking. These two close that gap by asking the installed package.
def _installed_integration():
    return pytest.importorskip(
        "catheter_vasculature_solver.isaaclab_integration",
        reason="needs the catheter solver checkout",
    )


def test_the_installed_rod_solver_can_place_and_report_its_root():
    """The two calls that make an arm-driven catheter possible.

    ``xpbd_catheter`` drives the wire by writing the root pose every step and
    reads ``proximal_reaction`` back as the load the holder feels. Losing
    either is an AttributeError mid-rollout rather than at import.
    """
    pytest.importorskip("catheter_vasculature_solver", reason="needs the catheter solver checkout")
    from catheter_vasculature_solver.xpbd_rod_solver import XPBDRodSolver

    assert hasattr(XPBDRodSolver, "set_root_pose_gpu")
    assert hasattr(XPBDRodSolver, "proximal_reaction")
    assert hasattr(XPBDRodSolver, "proximal_wrench")


def test_the_installed_coupled_cfg_accepts_the_fields_the_wiring_sets():
    """``coupled_solver_cfg`` builds the real cfg from these names.

    The stub takes ``**fields`` and updates ``__dict__``, so it would swallow a
    field the real config had dropped or renamed.
    """
    integration = _installed_integration()
    try:
        cfg_type = integration.CoupledMJWarpXPBDRodSolverCfg
    except SystemExit as exc:
        pytest.skip(f"Installed cfg requires an initialized Isaac Sim application: {exc}")

    for field in ("coupling_mode", "drive_body_name", "drive_mount_local", "drive_reaction_relaxation"):
        assert field in cfg_type.__dataclass_fields__, field


# -- bend radius: the one probe figure that can see a fold ----------------


def _arc(radius_m, samples, sweep_rad=np.pi / 2.0):
    angle = np.linspace(0.0, sweep_rad, samples)
    return np.stack([radius_m * np.cos(angle), radius_m * np.sin(angle), np.zeros(samples)], axis=1)


def test_a_straight_rod_reports_no_bend_at_all():
    """``inf`` rather than a large number, which would read as a gentle curve."""
    points = np.stack([np.linspace(0.0, 0.4, 9), np.zeros(9), np.zeros(9)], axis=1)

    assert np.isinf(bend_radii_m(points)).all()


def test_samples_on_a_circle_report_that_circle():
    radii = bend_radii_m(_arc(0.05, 9))

    np.testing.assert_allclose(radii, 0.05, rtol=1e-9)


def test_the_bend_radius_is_comparable_across_segment_counts():
    """The property the chord percentages lack, and the reason this is a radius.

    Three samples of a circle determine that circle whatever their spacing, so
    refining the rod leaves the figure alone. A per-node turning angle would
    have halved here, and a finer rod would have looked like a straighter one.
    """
    coarse = bend_radii_m(_arc(0.05, 5))
    fine = bend_radii_m(_arc(0.05, 17))

    assert coarse.min() == pytest.approx(fine.min(), rel=1e-9)


def test_too_few_samples_to_bend_report_nothing():
    assert bend_radii_m(np.zeros((2, 3))).size == 0


def test_coincident_samples_do_not_invent_a_kink():
    """A stalled solver leaves duplicate nodes; they are no bend, not a tight one."""
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])

    assert np.isinf(bend_radii_m(points)).all()


def test_a_folded_tip_is_invisible_to_penetration_and_arc_length():
    """Why this figure was added, stated as the case that motivated it.

    A tip doubled back on itself sits well inside the lumen and carries exactly
    the arc length it had straight. Penetration and arc excess therefore both
    report a perfectly healthy rod. Only the bend radius collapses.
    """
    path, radii = _straight_vessel(radius_m=0.02, samples=11, length_m=0.2)
    points = np.zeros((11, 3))
    points[:10, 0] = np.linspace(0.0, 0.09, 10)
    # Turn 178 degrees off the incoming direction, keeping the chord at 10 mm.
    heading = np.array([np.cos(np.deg2rad(178.0)), np.sin(np.deg2rad(178.0)), 0.0])
    points[10] = points[9] + 0.01 * heading

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.01)

    assert report["particles_outside"] == 0
    assert report["worst_penetration_mm"] < 0.0
    assert report["arc_excess_mm"] == pytest.approx(0.0, abs=1e-6)
    assert report["chord_max_pct"] == pytest.approx(100.0)
    assert report["min_bend_radius_mm"] < 6.0


def test_the_report_names_the_node_that_folded():
    """A tight radius at the tip is a folded tip; the same figure mid-shaft is anatomy."""
    path, radii = _straight_vessel(radius_m=0.02, samples=11, length_m=0.2)
    points = np.zeros((11, 3))
    points[:10, 0] = np.linspace(0.0, 0.09, 10)
    heading = np.array([np.cos(np.deg2rad(178.0)), np.sin(np.deg2rad(178.0)), 0.0])
    points[10] = points[9] + 0.01 * heading

    report = containment_report(points, path_world_m=path, lumen_radii_m=radii, segment_length_m=0.01)

    assert report["min_bend_radius_node"] == 9
    assert report["num_particles"] == 11


def test_a_full_fold_bottoms_out_at_about_half_a_segment():
    """The resolution limit, worth knowing before reading a small radius.

    A rod cannot report a bend tighter than the circle through two nodes a
    segment apart, so the floor is half the segment length. At the shipped
    2.5 mm segments that is about 1.2 mm, far below the tens of millimetres
    anatomy runs at, so a fold is still unambiguous -- but the figure saturates
    rather than growing without bound as the fold tightens.
    """
    for turn_deg in (170.0, 178.0, 179.5):
        heading = np.array([np.cos(np.deg2rad(turn_deg)), np.sin(np.deg2rad(turn_deg)), 0.0])
        points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.0, 0.0] + 0.01 * heading])

        assert bend_radii_m(points).min() == pytest.approx(0.005, rel=0.05)


def test_an_anatomical_curve_reads_far_looser_than_a_fold():
    """The discrimination the probe exists to make, on one axis, in millimetres."""
    anatomy = bend_radii_m(_arc(0.04, 9)).min()
    heading = np.array([np.cos(np.deg2rad(178.0)), np.sin(np.deg2rad(178.0)), 0.0])
    fold = bend_radii_m(np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.0, 0.0] + 0.01 * heading])).min()

    assert anatomy == pytest.approx(0.04)
    assert fold < anatomy / 5.0


@pytest.mark.parametrize("segments", [40, 80, 120, 240])
def test_physical_tip_span_survives_mesh_refinement(stub_isaac, monkeypatch, segments):
    from i4h_arena.medical.newton_catheter_physics import TIP_LENGTH_ENV_VAR, rod_solver_cfg

    monkeypatch.delenv(TIP_EDGES_ENV_VAR, raising=False)
    monkeypatch.delenv(TIP_LENGTH_ENV_VAR, raising=False)
    spec = CatheterRodSpec(length_m=0.303225, num_segments=segments, tip_length_m=0.025)
    cfg = rod_solver_cfg(spec)
    span = cfg.tip_num_edges * spec.segment_length_m
    assert 0.025 <= span < 0.025 + spec.segment_length_m


def test_physical_tip_override_wins_over_legacy_edge_override(stub_isaac, monkeypatch):
    from i4h_arena.medical.newton_catheter_physics import TIP_LENGTH_ENV_VAR, rod_solver_cfg

    monkeypatch.setenv(TIP_EDGES_ENV_VAR, "1")
    monkeypatch.setenv(TIP_LENGTH_ENV_VAR, "20")
    cfg = rod_solver_cfg(CatheterRodSpec(length_m=0.3, num_segments=120))
    assert cfg.tip_num_edges == 8


@pytest.mark.parametrize("length", [0, -0.01, float("nan"), float("inf"), 0.5])
def test_invalid_physical_tip_lengths_are_rejected(length):
    with pytest.raises(ValueError, match="tip_length_m"):
        CatheterRodSpec(length_m=0.3, tip_length_m=length)
