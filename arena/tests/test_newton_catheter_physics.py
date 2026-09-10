# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for placing the catheter rod under Isaac Lab's Newton manager.

The ordering assertions are the valuable ones. Particles added after the model
is finalized are invisible to it, and a rod registered before its particles
exist has nothing to drive, so this checks that registration happens on
MODEL_INIT and in the right order against stubs rather than a live stack.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.medical.newton_catheter_physics import (
    CLEANUP_ROUNDS_ENV_VAR,
    CLEANUP_SWEEPS_ENV_VAR,
    CONTAINMENT_STAGE_ENV_VAR,
    DAMPING_ENV_VAR,
    INTERIOR_CONTAINMENT_ENV_VAR,
    DEFAULT_NUM_SEGMENTS,
    GRAVITY_WORLD_Z_UP,
    REFERENCE_BEND_STIFFNESS,
    REFERENCE_NUM_SEGMENTS,
    SEGMENT_COUNT_ENV_VAR,
    CatheterRodHandle,
    CatheterRodSpec,
    cleanup_sweeps_override,
    cleanup_rounds_override,
    containment_report,
    containment_stage_override,
    interior_containment_override,
    mesh_invariant_bend_stiffness,
    nearest_on_polyline,
    rod_damping_override,
    segment_count_override,
    segment_inverse_inertia,
    tip_bend_stiffness_profile,
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


def test_gravity_defaults_to_the_z_up_world():
    """Isaac is Z-up; the rod config's own default points along -Y."""
    assert CatheterRodSpec().gravity_world == GRAVITY_WORLD_Z_UP
    assert GRAVITY_WORLD_Z_UP[2] < 0.0


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


def test_physical_inertia_stays_out_of_the_shipped_spec():
    """It is the physical value, but it measured no better, so it is opt-in."""
    assert CatheterRodSpec().physical_rotational_inertia is False


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


def test_the_shipped_rod_is_finer_than_the_count_the_stiffness_was_tuned_at():
    """The two constants mean different things and must not drift together.

    ``REFERENCE_NUM_SEGMENTS`` calibrates the bend-stiffness compensation and
    stays where the tuning was done. ``DEFAULT_NUM_SEGMENTS`` is what ships,
    and moved because 40 was measured to snap.
    """
    assert DEFAULT_NUM_SEGMENTS > REFERENCE_NUM_SEGMENTS


@pytest.mark.parametrize("value, expected", [("80", 80), ("120", 120), (" 60 ", 60), ("2", 2)])
def test_the_segment_count_can_be_walked_for_a_measurement(value, expected):
    assert segment_count_override({SEGMENT_COUNT_ENV_VAR: value}) == expected


@pytest.mark.parametrize("value", ["", "lots", "40.5", "1", "0", "-40"])
def test_a_segment_count_below_two_is_ignored(value):
    """One segment has no interior joint, so no bend constraint to refine."""
    assert segment_count_override({SEGMENT_COUNT_ENV_VAR: value}) is None


def test_refining_at_the_reference_length_changes_no_stiffness():
    """The compensation has to be inert at the count everything was tuned at."""
    assert mesh_invariant_bend_stiffness(0.1, 0.0165, 0.0165) == pytest.approx(0.1)


def test_the_reference_stiffness_matches_the_installed_cfg():
    """The mirrored default has to track the solver's, or refining re-tunes it.

    Arena is on the light discovery path and cannot import the solver package
    at module scope, so the value is copied. This is the pin that keeps the
    copy honest; it skips where the solver is not installed.
    """
    import dataclasses

    try:
        from catheter_vasculature_solver.isaaclab_integration import XPBDRodSolverCfg
    except (ImportError, SystemExit) as exc:
        # Isaac Sim exits rather than raising when it cannot bootstrap its
        # kernel, which is what importing the solver package does here.
        pytest.skip(f"solver package not importable in this environment: {exc}")
    # ``configclass`` rewrites plain defaults into ``default_factory``, so the
    # value is not reachable as a class attribute.
    field = next(f for f in dataclasses.fields(XPBDRodSolverCfg) if f.name == "bend_stiffness")
    default = field.default if field.default is not dataclasses.MISSING else field.default_factory()

    assert float(default) == pytest.approx(REFERENCE_BEND_STIFFNESS)


def test_three_times_the_segments_needs_nine_times_the_stiffness():
    """The joint measures angle, not curvature, so the error goes as ``L^2``.

    A rod at curvature ``kappa`` turns ``kappa * L`` per joint and stores
    ``E * bend_stiffness * kappa^2 * L^3`` there, over ``length / L`` joints.
    That leaves the total going as ``L^2`` rather than staying put, so thirds
    of the segment length need nine times the stiffness to describe the same
    wire.
    """
    reference = mesh_invariant_bend_stiffness(0.1, 0.0165, 0.0165)
    refined = mesh_invariant_bend_stiffness(0.1, 0.0165, 0.0165 / 3.0)
    assert refined / reference == pytest.approx(9.0)


def test_a_coarser_rod_is_compensated_the_other_way():
    """Halving the segment count is a quarter of the stiffness, not double."""
    assert mesh_invariant_bend_stiffness(0.1, 0.0165, 0.033) == pytest.approx(0.025)


def test_total_bending_energy_is_flat_across_refinement():
    """The point of the compensation, stated as the invariant it preserves.

    Hold a rod of fixed length at a fixed curvature and sum the joint energies
    the solver would store, ``k * theta^2`` with ``k ~ bend_stiffness * L`` and
    ``theta = kappa * L``. Compensated, the total is the same rod however
    finely it is cut.
    """
    length_m, curvature = 0.66, 12.0
    totals = []
    for segments in (20, 40, 80, 160):
        segment_length = length_m / segments
        stiffness = mesh_invariant_bend_stiffness(0.1, length_m / REFERENCE_NUM_SEGMENTS, segment_length)
        joint = stiffness * segment_length * (curvature * segment_length) ** 2
        totals.append(joint * segments)
    assert totals == pytest.approx([totals[0]] * len(totals))


@pytest.mark.parametrize("reference_m, segment_m", [(0.0, 0.01), (0.01, 0.0), (-0.01, 0.01)])
def test_a_nonphysical_segment_length_is_rejected(reference_m, segment_m):
    with pytest.raises(ValueError, match="must be positive"):
        mesh_invariant_bend_stiffness(0.1, reference_m, segment_m)


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


def test_cuda_graph_is_disabled_when_a_vessel_is_present(stub_isaac):
    """Vessel containment resizes contact scratch, which a captured graph cannot express."""
    from i4h_arena.medical.newton_catheter_physics import newton_physics_cfg

    with_vessel = newton_physics_cfg(CatheterRodSpec(patient_twin_manifest="twin.yaml"))
    without_vessel = newton_physics_cfg(CatheterRodSpec(patient_twin_manifest=None))

    assert with_vessel.use_cuda_graph is False
    assert without_vessel.use_cuda_graph is True


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


def test_a_rod_at_the_reference_count_does_not_touch_the_bend_stiffness(stub_isaac):
    """At the calibration count the compensation must be completely inert.

    Writing the value back even at its own default would make this show up as
    a physics diff on a rod nobody asked to refine.
    """
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec(num_segments=REFERENCE_NUM_SEGMENTS))

    assert not hasattr(cfg, "bend_stiffness")


def test_the_shipped_rod_carries_its_compensated_stiffness(stub_isaac):
    """120 segments is three times 40, so the wire needs nine times the value.

    Without this the shipped catheter would silently be nine times floppier
    than the one every earlier measurement was taken on.
    """
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec())

    assert cfg.num_segments == DEFAULT_NUM_SEGMENTS
    assert cfg.bend_stiffness == pytest.approx(9.0 * REFERENCE_BEND_STIFFNESS)


def test_a_refined_rod_is_stiffened_to_stay_the_same_wire(stub_isaac):
    """Three times the segments, nine times the stiffness, reaching the cfg."""
    from i4h_arena.medical.newton_catheter_physics import rod_solver_cfg

    cfg = rod_solver_cfg(CatheterRodSpec(num_segments=3 * REFERENCE_NUM_SEGMENTS))

    assert cfg.bend_stiffness == pytest.approx(9.0 * REFERENCE_BEND_STIFFNESS)


def test_the_refined_rod_still_reaches_the_coupled_cfg(stub_isaac):
    """The arm scene nests the rod cfg, so it has to carry the same fix."""
    from i4h_arena.medical.newton_catheter_physics import coupled_solver_cfg

    spec = CatheterRodSpec(rigid_bodies_enabled=True, num_segments=2 * REFERENCE_NUM_SEGMENTS)
    cfg = coupled_solver_cfg(spec)

    assert cfg.rod_solver_cfg.bend_stiffness == pytest.approx(4.0 * REFERENCE_BEND_STIFFNESS)


def test_an_explicit_bend_stiffness_still_wins(stub_isaac):
    """The compensation is a default, not a policy: an override overrules it."""
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


def test_cuda_graph_is_disabled_when_an_arm_is_present(stub_isaac):
    """MJWarp's contact counts vary with the arm's pose, which a captured graph
    cannot express, so an arm disables capture even without a vessel."""
    from i4h_arena.medical.newton_catheter_physics import newton_physics_cfg

    cfg = newton_physics_cfg(CatheterRodSpec(patient_twin_manifest=None, rigid_bodies_enabled=True))

    assert cfg.use_cuda_graph is False


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

    solver_module.CathRodSolver = CathRodSolver
    monkeypatch.setitem(sys.modules, "catheter_vasculature_solver", solver_module)
    return SimpleNamespace(calls=calls, registered=registered, builder=builder, recorded=recorded)


def test_particles_are_added_before_the_rod_is_registered(stub_model_init):
    """Registering a rod whose particles do not exist yet leaves it driving nothing."""
    handle = CatheterRodHandle(CatheterRodSpec(num_envs=2, num_segments=8))

    handle._on_model_init()

    calls = stub_model_init.calls
    assert calls.index("add_particles") < calls.index("register_rod")
    assert calls.index("build_rod") < calls.index("register_rod")


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
    assert tuple(written) == pytest.approx((expected, expected, expected))
    assert written[0] > 1.0


def test_the_shipped_rod_keeps_the_solver_default(stub_model_init):
    """What the validated +3 mm configuration was measured with."""
    handle = CatheterRodHandle(CatheterRodSpec())

    handle._on_model_init()

    assert handle.rod._ws.inv_inertia_local_diag == (1.0, 1.0, 1.0)


def test_a_straight_rod_gets_no_explicit_positions(stub_model_init):
    handle = CatheterRodHandle(CatheterRodSpec(initial_path_world_m=None))

    handle._on_model_init()

    assert stub_model_init.recorded["positions"] is None


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


def test_the_installed_coupled_cfg_accepts_the_fields_the_wiring_sets():
    """``coupled_solver_cfg`` builds the real cfg from these names.

    The stub takes ``**fields`` and updates ``__dict__``, so it would swallow a
    field the real config had dropped or renamed.
    """
    integration = _installed_integration()
    cfg_type = integration.CoupledMJWarpXPBDRodSolverCfg

    for field in ("coupling_mode", "drive_body_name", "drive_reaction_relaxation"):
        assert field in cfg_type.__dataclass_fields__, field
