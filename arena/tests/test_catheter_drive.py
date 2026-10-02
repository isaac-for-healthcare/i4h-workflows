# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the introducers the catheter is fed through.

What decides whether the catheter can navigate at all is where the feed puts
the wire's proximal particle, which is prescribed and so goes exactly where the
drive says. Three answers, in the order they were tried.

A fixed introducer axis ends up 44 mm outside the lumen by 300 mm of depth on
the s0011 twin, because the vessel curves away from it.

The wire's own tangent follows the curve and closes a feedback loop: the push
bends the segment the direction is read from, so a fold feeds itself. It is the
drive the flange-mounted tests below cover, since an arm-carried unit still
uses it.

Arc length along the route has neither failure, and is what the keyboard drive
uses. Its tests are the ones that assert the root is *on* the route rather than
near it, which is the property the other two lose.

Around all of that: insertion is a roller command, so it must not depend on the
arm's travel -- the rod is a fixed-length stick and tying its reach to the
arm's would strand most of it -- and the flange carries the wire as a rigid
delta rather than an absolute offset, because the wire slides *through* the
drive unit.
"""

from __future__ import annotations

import ast
import logging
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ARENA = Path(__file__).parents[1] / "i4h_arena"
PLAIN_CATHETER = ARENA / "embodiments" / "catheter.py"
XPBD_CATHETER = ARENA / "medical" / "xpbd_catheter.py"

from i4h_arena.medical.catheter_drive import (  # noqa: E402
    FEED_LOG_ENV_VAR,
    FlangeMountedIntroducer,
    IntroducerDriveSpec,
    LumenClamp,
    RouteRailedIntroducer,
    feed_log_seconds,
    frame_along,
    quat_to_w_first,
    quat_to_xyzw,
    twist_about_axis,
)

X_AXIS = (1.0, 0.0, 0.0)

#: A route that turns, which is where a drive that steps along a direction
#: parts company with the vessel. 100 mm along +X, then 50 mm along +Y.
CORNER = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.05, 0.0)]

#: A route that curves everywhere rather than at one vertex, so drift has
#: somewhere to accumulate: a quarter circle of 0.2 m radius in the XY plane.
ARCH = [
    (0.2 * math.sin(0.5 * math.pi * index / 40), 0.2 * (1.0 - math.cos(0.5 * math.pi * index / 40)), 0.0)
    for index in range(41)
]


def quat_about(axis: tuple[float, float, float], angle_rad: float) -> torch.Tensor:
    """One ``(1, 4)`` w-first quaternion rotating ``angle_rad`` about ``axis``."""
    vec = torch.tensor(axis, dtype=torch.float32)
    vec = vec / torch.linalg.norm(vec)
    half = 0.5 * angle_rad
    return torch.cat([torch.tensor([math.cos(half)]), vec * math.sin(half)]).unsqueeze(0)


IDENTITY_QUAT = quat_about(X_AXIS, 0.0)


def vec(*values: float) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.float32)


def rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Turn ``vector`` by a ``(N, 4)`` w-first ``quat``."""
    w, axis = quat[..., :1], quat[..., 1:]
    return (
        vector * (w * w - (axis * axis).sum(-1, keepdim=True))
        + 2.0 * axis * (axis * vector).sum(-1, keepdim=True)
        + 2.0 * w * torch.linalg.cross(axis, vector.expand_as(axis), dim=-1)
    )


def rail(route, num_envs: int = 1) -> RouteRailedIntroducer:
    return RouteRailedIntroducer(torch.tensor(route, dtype=torch.float32), num_envs)


def distance_to_route(route, point: torch.Tensor) -> float:
    """How far ``point`` sits from the route polyline, metres."""
    path = torch.tensor(route, dtype=torch.float32)
    start, span = path[:-1], path[1:] - path[:-1]
    offset = point.reshape(1, 3) - start
    travel = ((offset * span).sum(-1) / (span * span).sum(-1)).clamp(0.0, 1.0)
    return float(torch.linalg.norm(offset - travel.unsqueeze(-1) * span, dim=-1).min())


def introducer(spec: IntroducerDriveSpec | None = None, num_envs: int = 1) -> FlangeMountedIntroducer:
    """An introducer whose limits are out of the way.

    The clamps have their own tests below; leaving the shipped defaults here
    would silently turn every insertion assertion into a limit assertion.
    """
    if spec is None:
        spec = IntroducerDriveSpec(
            max_insertion_velocity_mps=100.0,
            max_rotation_rate_radps=100.0,
            travel_limit_m=100.0,
        )
    return FlangeMountedIntroducer(spec, num_envs)


def feed(
    unit: FlangeMountedIntroducer,
    root=(0.0, 0.0, 0.0),
    tangent=X_AXIS,
    transport=(0.0, 0.0, 0.0),
    insertion: float = 0.0,
    rotation: float = 0.0,
    dt: float = 1.0,
    quat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One roller step, returning the root pose it asks for."""
    spent, spin = unit.advance(torch.tensor([insertion]), torch.tensor([rotation]), dt=dt)
    return unit.root_target(
        vec(*root),
        IDENTITY_QUAT if quat is None else quat,
        vec(*tangent),
        vec(*transport),
        spent,
        spin,
        dt=dt,
    )


# --------------------------------------------------------------------------- #
# Spec validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_insertion_velocity_mps": 0.0}, "max_insertion_velocity_mps"),
        ({"max_rotation_rate_radps": 0.0}, "max_rotation_rate_radps"),
        ({"travel_limit_m": 0.0}, "travel_limit_m"),
    ],
)
def test_nonsense_limits_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        IntroducerDriveSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Feed follows the wire, not a fixed axis
# --------------------------------------------------------------------------- #
def test_feeding_advances_the_root_along_the_tangent():
    unit = introducer()

    position, _ = feed(unit, root=(0.1, 0.0, 0.0), tangent=X_AXIS, insertion=0.02, dt=0.5)

    torch.testing.assert_close(position, vec(0.11, 0.0, 0.0))


def test_the_feed_direction_follows_the_wire_around_a_bend():
    """The whole reason the tangent is read every step.

    A rod whose first segment has turned must be fed along the turn. Feeding
    along the axis it started on is what walked the root out of the vessel.
    """
    unit = introducer()

    position, _ = feed(unit, root=(0.1, 0.0, 0.0), tangent=(0.0, 1.0, 0.0), insertion=0.01, dt=1.0)

    torch.testing.assert_close(position, vec(0.1, 0.01, 0.0))


def test_the_tangent_is_normalized_before_it_is_spent():
    """Segment length must not scale the feed, or a stretched rod feeds faster."""
    unit = introducer()

    position, _ = feed(unit, root=(0.0, 0.0, 0.0), tangent=(7.0, 0.0, 0.0), insertion=0.01, dt=1.0)

    torch.testing.assert_close(position, vec(0.01, 0.0, 0.0))


def test_a_collapsed_first_segment_feeds_nowhere():
    """A degenerate tangent has no direction to spend feed along.

    Normalizing it anyway would divide by ~zero and fling the root somewhere
    arbitrary, which is worse than a step of lost insertion.
    """
    unit = introducer()

    position, _ = feed(unit, root=(0.2, 0.0, 0.0), tangent=(0.0, 0.0, 0.0), insertion=1.0, dt=1.0)

    torch.testing.assert_close(position, vec(0.2, 0.0, 0.0))


def test_feed_is_measured_from_the_wire_not_from_the_start():
    """Each step starts from where the rod actually is.

    The root pose is an absolute command, so integrating it from a remembered
    position would let solver corrections and contact be quietly overwritten.
    """
    unit = introducer()

    position, _ = feed(unit, root=(0.25, 0.03, -0.01), tangent=X_AXIS, insertion=0.01, dt=1.0)

    torch.testing.assert_close(position, vec(0.26, 0.03, -0.01))


# --------------------------------------------------------------------------- #
# The flange carries the wire
# --------------------------------------------------------------------------- #
def test_flange_motion_is_transported_into_the_wire():
    """What keeps the arm in the loop once feed stopped depending on it."""
    unit = introducer()

    position, _ = feed(unit, root=(0.1, 0.0, 0.0), transport=(0.0, -0.02, 0.005), insertion=0.0)

    torch.testing.assert_close(position, vec(0.1, -0.02, 0.005))


def test_transport_and_feed_compose():
    unit = introducer()

    position, _ = feed(unit, root=(0.1, 0.0, 0.0), transport=(0.0, -0.02, 0.0), insertion=0.01, dt=1.0)

    torch.testing.assert_close(position, vec(0.11, -0.02, 0.0))


def test_transport_is_the_flange_delta_since_the_last_step():
    unit = introducer()

    unit.transport(vec(0.5, 0.0, 0.0))
    delta = unit.transport(vec(0.53, -0.01, 0.0))

    torch.testing.assert_close(delta, vec(0.03, -0.01, 0.0))


def test_the_first_step_transports_nothing():
    """No remembered pose means no delta, rather than differencing the origin."""
    unit = introducer()

    torch.testing.assert_close(unit.transport(vec(0.5, 0.2, 0.9)), vec(0.0, 0.0, 0.0))


def test_a_reset_forgets_the_flange_pose():
    """Otherwise the arm's jump home is transported straight into the wire."""
    unit = introducer()
    unit.transport(vec(0.5, 0.0, 0.0))

    unit.reset()

    torch.testing.assert_close(unit.transport(vec(0.1, 0.4, 0.0)), vec(0.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# Insertion is a roller command, independent of arm travel
# --------------------------------------------------------------------------- #
def test_a_still_arm_still_feeds_wire():
    """The distinguishing claim of the roller drive.

    Under the rejected design a stationary flange fed nothing, which capped the
    reachable insertion at the arm's own travel.
    """
    unit = introducer()

    for _ in range(10):
        unit.advance(torch.tensor([0.05]), torch.zeros(1), dt=0.1)

    assert float(unit.depth_m[0]) == pytest.approx(0.05, abs=1e-6)


def test_arm_travel_alone_feeds_no_wire():
    """The converse: repositioning the arm must not inject insertion."""
    unit = introducer()

    unit.transport(vec(0.0, 0.0, 0.0))
    unit.transport(vec(0.30, 0.10, 0.0))

    assert float(unit.depth_m[0]) == pytest.approx(0.0)


def test_retraction_is_allowed_down_to_the_access_site():
    unit = introducer()
    unit.advance(torch.tensor([0.10]), torch.zeros(1), dt=1.0)

    unit.advance(torch.tensor([-0.04]), torch.zeros(1), dt=1.0)

    assert float(unit.depth_m[0]) == pytest.approx(0.06)


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
def test_commanded_feed_is_clamped_to_the_roller_limit():
    unit = introducer(IntroducerDriveSpec(max_insertion_velocity_mps=0.05, travel_limit_m=100.0))

    spent, _ = unit.advance(torch.tensor([5.0]), torch.zeros(1), dt=1.0)

    assert float(spent[0]) == pytest.approx(0.05)


def test_commanded_spin_is_clamped_to_the_roller_limit():
    unit = introducer(IntroducerDriveSpec(max_rotation_rate_radps=1.5, travel_limit_m=100.0))

    _, spin = unit.advance(torch.zeros(1), torch.tensor([50.0]), dt=1.0)

    assert float(spin[0]) == pytest.approx(1.5)


def test_the_travel_limit_stops_the_wire():
    unit = introducer(IntroducerDriveSpec(max_insertion_velocity_mps=100.0, travel_limit_m=0.10))

    unit.advance(torch.tensor([1.0]), torch.zeros(1), dt=1.0)

    assert float(unit.depth_m[0]) == pytest.approx(0.10)


def test_a_blocked_command_reports_the_insertion_that_happened():
    """The recorded column has to match the wire, not the request.

    A command the rollers could not spend is not insertion, and logging it as
    insertion would teach a policy that the stop does not exist.
    """
    unit = introducer(IntroducerDriveSpec(max_insertion_velocity_mps=100.0, travel_limit_m=0.10))
    unit.advance(torch.tensor([0.10]), torch.zeros(1), dt=1.0)

    spent, _ = unit.advance(torch.tensor([0.10]), torch.zeros(1), dt=1.0)

    assert float(spent[0]) == pytest.approx(0.0)


def test_a_blocked_command_also_moves_the_root_nowhere():
    unit = introducer(IntroducerDriveSpec(max_insertion_velocity_mps=100.0, travel_limit_m=0.10))
    feed(unit, insertion=0.10, dt=1.0)

    position, _ = feed(unit, root=(0.10, 0.0, 0.0), insertion=0.10, dt=1.0)

    torch.testing.assert_close(position, vec(0.10, 0.0, 0.0))


def test_the_wire_cannot_be_pulled_out_backwards():
    unit = introducer()

    spent, _ = unit.advance(torch.tensor([-1.0]), torch.zeros(1), dt=1.0)

    assert float(unit.depth_m[0]) == pytest.approx(0.0)
    assert float(spent[0]) == pytest.approx(0.0)


def test_a_non_positive_step_is_rejected():
    unit = introducer()

    with pytest.raises(ValueError, match="dt must be positive"):
        unit.advance(torch.zeros(1), torch.zeros(1), dt=0.0)


# --------------------------------------------------------------------------- #
# Feed diagnostics
# --------------------------------------------------------------------------- #
def test_the_feed_log_is_off_until_an_interval_is_asked_for():
    """A line per step would bury a teleop log, so this stays opt-in."""
    assert feed_log_seconds({}) == 0.0


@pytest.mark.parametrize("value, expected", [("0.5", 0.5), ("2", 2.0), (" 1.5 ", 1.5)])
def test_the_feed_log_interval_is_read_in_seconds(value, expected):
    assert feed_log_seconds({FEED_LOG_ENV_VAR: value}) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "often", "0", "-1"])
def test_a_malformed_feed_interval_leaves_the_log_off(value):
    """A mistyped diagnostic must not take a simulator run down with it."""
    assert feed_log_seconds({FEED_LOG_ENV_VAR: value}) == 0.0


def test_the_feed_log_separates_the_ceiling_from_the_stop(monkeypatch, caplog):
    """The three numbers have to disagree, or the log cannot tell them apart.

    A command over the velocity ceiling that also runs into the travel stop
    should report the full command, the ceiling it was cut to, and the zero it
    actually spent.
    """
    unit = introducer(IntroducerDriveSpec(max_insertion_velocity_mps=0.05, travel_limit_m=0.10))
    unit.advance(torch.tensor([1.0]), torch.zeros(1), dt=10.0)  # fills the travel, log still off

    monkeypatch.setenv(FEED_LOG_ENV_VAR, "0.0001")
    with caplog.at_level(logging.INFO, logger="i4h_arena.medical.catheter_drive"):
        unit.advance(torch.tensor([1.0]), torch.zeros(1), dt=1.0)

    line = caplog.text
    assert "commanded=1000.00 mm/s" in line
    assert "clamped=50.00 mm/s" in line
    assert "spent=0.00 mm/s" in line
    assert "depth=100.0/100 mm" in line


def test_an_undriven_roller_reports_a_zero_command(monkeypatch, caplog):
    """The case the seg120 run could not distinguish: nothing asked it to move."""
    monkeypatch.setenv(FEED_LOG_ENV_VAR, "0.0001")
    unit = introducer()

    with caplog.at_level(logging.INFO, logger="i4h_arena.medical.catheter_drive"):
        unit.advance(torch.zeros(1), torch.zeros(1), dt=1.0)

    assert "commanded=0.00 mm/s" in caplog.text
    assert "spent=0.00 mm/s" in caplog.text


def test_the_feed_log_respects_its_interval(monkeypatch, caplog):
    """Otherwise it is the per-step logging it exists to avoid."""
    monkeypatch.setenv(FEED_LOG_ENV_VAR, "3600")
    unit = introducer()

    with caplog.at_level(logging.INFO, logger="i4h_arena.medical.catheter_drive"):
        for _ in range(5):
            unit.advance(torch.tensor([0.01]), torch.zeros(1), dt=0.01)

    assert caplog.text.count("catheter feed") == 1


# --------------------------------------------------------------------------- #
# Roll
# --------------------------------------------------------------------------- #
def test_spinning_the_rollers_twists_the_wire_about_its_own_axis():
    unit = introducer()

    _, quat = feed(unit, tangent=X_AXIS, rotation=math.pi / 2, dt=1.0)

    axis = torch.tensor(X_AXIS, dtype=torch.float32)
    assert float(twist_about_axis(IDENTITY_QUAT, quat, axis)[0]) == pytest.approx(math.pi / 2, abs=1e-5)


def test_roll_is_about_the_tangent_not_a_fixed_axis():
    """Twist has to follow the bend too, or a turned wire rolls about nothing."""
    unit = introducer()

    _, quat = feed(unit, tangent=(0.0, 1.0, 0.0), rotation=math.pi / 2, dt=1.0)

    about_y = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32)
    assert float(twist_about_axis(IDENTITY_QUAT, quat, about_y)[0]) == pytest.approx(math.pi / 2, abs=1e-5)


# --------------------------------------------------------------------------- #
# Solver boundary
# --------------------------------------------------------------------------- #
def test_quaternions_are_reordered_for_the_solver():
    """The rod solver stores root orientation xyzw; this side is w-first.

    Silent if wrong -- both are four floats -- and the failure is a rod whose
    root twist is a plausible-looking wrong rotation.
    """
    w_first = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    torch.testing.assert_close(quat_to_xyzw(w_first), torch.tensor([[0.2, 0.3, 0.4, 0.1]]))


def test_the_reordering_round_trips():
    w_first = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    torch.testing.assert_close(quat_to_w_first(quat_to_xyzw(w_first)), w_first)


# --------------------------------------------------------------------------- #
# Lumen containment for the prescribed proximal particle
# --------------------------------------------------------------------------- #
def straight_lumen(radius: float = 0.01, margin: float = 0.0) -> LumenClamp:
    """A 0.3 m vessel of constant radius running along +X."""
    path = torch.tensor([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32)
    return LumenClamp(path, torch.full((3,), radius), margin_m=margin)


def test_a_point_inside_the_lumen_is_left_alone():
    """The clamp restores wall contact, not a rail.

    A wire is free to sit off-centre or lie against a wall; only leaving the
    vessel is forbidden. Snapping to the centerline would rail the catheter
    down the middle of a vessel it is supposed to be navigating.
    """
    clamp = straight_lumen(radius=0.01)

    torch.testing.assert_close(clamp.project(vec(0.1, 0.006, 0.0)), vec(0.1, 0.006, 0.0))


def test_a_point_outside_the_lumen_is_pulled_to_the_wall():
    clamp = straight_lumen(radius=0.01)

    torch.testing.assert_close(clamp.project(vec(0.1, 0.05, 0.0)), vec(0.1, 0.01, 0.0))


def test_the_margin_keeps_the_wire_surface_off_the_wall():
    """The clamp constrains the wire's axis, so the wire's own radius has to come off."""
    clamp = straight_lumen(radius=0.01, margin=0.002)

    torch.testing.assert_close(clamp.project(vec(0.1, 0.05, 0.0)), vec(0.1, 0.008, 0.0))


def test_a_clamped_point_stays_on_its_own_side_of_the_centerline():
    """A wire against one wall must not be flicked across to the other."""
    clamp = straight_lumen(radius=0.01)

    assert float(clamp.project(vec(0.1, -0.05, 0.0))[0, 1]) < 0.0


def test_clamping_preserves_position_along_the_vessel():
    """Containment is radial. Moving the point along the vessel would be feed."""
    clamp = straight_lumen(radius=0.01)

    torch.testing.assert_close(clamp.project(vec(0.2, 0.4, 0.0))[:, 0], torch.tensor([0.2]))


def test_the_clamp_narrows_with_a_tapering_vessel():
    """A vessel that narrows has to hold the wire tighter downstream."""
    path = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32)
    clamp = LumenClamp(path, torch.tensor([0.02, 0.004]), margin_m=0.0)

    assert float(clamp.project(vec(0.29, 0.5, 0.0))[0, 1]) == pytest.approx(0.004, abs=1e-6)


def test_a_point_past_the_end_is_held_against_the_last_station():
    """Feed can outrun the sampled path; the wire is still inside a vessel."""
    clamp = straight_lumen(radius=0.01)

    projected = clamp.project(vec(0.5, 0.05, 0.0))

    assert float(torch.linalg.norm(projected - vec(0.3, 0.0, 0.0))) == pytest.approx(0.01, abs=1e-6)


def test_environments_are_clamped_independently():
    clamp = straight_lumen(radius=0.01)

    projected = clamp.project(torch.tensor([[0.1, 0.002, 0.0], [0.1, 0.05, 0.0]]))

    torch.testing.assert_close(projected, torch.tensor([[0.1, 0.002, 0.0], [0.1, 0.01, 0.0]]))


@pytest.mark.parametrize(
    ("path", "radii", "margin", "message"),
    [
        (torch.zeros((1, 3)), torch.ones(1), 0.0, "S >= 2"),
        (torch.zeros((3, 3)), torch.ones(2), 0.0, "one radius per path vertex"),
        (torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), torch.ones(2), -0.1, "margin_m"),
    ],
)
def test_nonsense_lumen_geometry_is_rejected(path, radii, margin, message):
    with pytest.raises(ValueError, match=message):
        LumenClamp(path, radii, margin_m=margin)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def test_environments_are_fed_independently():
    unit = introducer(num_envs=3)

    unit.advance(torch.tensor([0.01, 0.02, 0.03]), torch.zeros(3), dt=1.0)

    torch.testing.assert_close(unit.depth_m, torch.tensor([0.01, 0.02, 0.03]))


def test_resetting_one_environment_leaves_the_others_fed():
    unit = introducer(num_envs=3)
    unit.advance(torch.tensor([0.01, 0.02, 0.03]), torch.zeros(3), dt=1.0)

    unit.reset(torch.tensor([1]))

    torch.testing.assert_close(unit.depth_m, torch.tensor([0.01, 0.0, 0.03]))


# --------------------------------------------------------------------------- #
# Frames along a direction
# --------------------------------------------------------------------------- #
def test_a_frame_along_z_is_the_identity():
    """The rod's frames carry local +Z along the tangent, so +Z needs no rotation."""
    torch.testing.assert_close(frame_along(vec(0.0, 0.0, 1.0)), IDENTITY_QUAT)


@pytest.mark.parametrize(
    "tangent",
    [X_AXIS, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 1.0, 1.0), (0.3, -0.7, 0.2)],
)
def test_a_frame_carries_local_z_onto_its_tangent(tangent):
    """The property the rod's stretch constraint reads the frame for."""
    unit = vec(*tangent)
    unit = unit / torch.linalg.norm(unit)

    carried = rotate(frame_along(unit), vec(0.0, 0.0, 1.0))

    torch.testing.assert_close(carried, unit, atol=1e-6, rtol=1e-5)


def test_a_frame_is_normalized_even_at_the_reversal():
    """A tangent at -Z leaves the minimal-rotation axis undefined, not the frame."""
    quat = frame_along(vec(0.0, 0.0, -1.0))

    assert torch.isfinite(quat).all()
    torch.testing.assert_close(torch.linalg.norm(quat, dim=-1), torch.ones(1))


# --------------------------------------------------------------------------- #
# Feeding along the route
# --------------------------------------------------------------------------- #
def test_the_root_starts_at_the_head_of_the_route():
    """Arc zero is where the rod was seeded, so the first pose is no jump."""
    unit = rail(CORNER)

    position, quat = unit.root_target()

    torch.testing.assert_close(position, vec(0.0, 0.0, 0.0))
    torch.testing.assert_close(quat, frame_along(vec(*X_AXIS)))


def test_insertion_advances_the_root_by_arc_length():
    unit = rail(CORNER)

    unit.advance(torch.tensor([0.04]), torch.zeros(1), dt=1.0)

    torch.testing.assert_close(unit.root_target()[0], vec(0.04, 0.0, 0.0))


def test_the_root_turns_the_corner_instead_of_cutting_it():
    """The arc is measured along the route, so 150 mm of feed rounds the bend.

    A drive that stepped along a direction would still be heading down +X here,
    50 mm outside a vessel that turned 50 mm ago.
    """
    unit = rail(CORNER)

    unit.advance(torch.tensor([0.15]), torch.zeros(1), dt=1.0)

    position, quat = unit.root_target()
    torch.testing.assert_close(position, vec(0.1, 0.05, 0.0))
    torch.testing.assert_close(quat, frame_along(vec(0.0, 1.0, 0.0)), atol=1e-6, rtol=1e-5)


def test_the_root_never_leaves_a_curved_route():
    """The claim the whole class exists for, over a full traverse.

    Tangent-following accumulates its error; an arc lookup has none to
    accumulate, so the prescribed particle is on the vessel at every depth
    rather than drifting millimetres out of it.
    """
    unit = rail(ARCH)
    off_route = []

    for _ in range(200):
        unit.advance(torch.full((1,), 0.004), torch.zeros(1), dt=1.0)
        off_route.append(distance_to_route(ARCH, unit.root_target()[0]))

    assert max(off_route) < 1e-6, f"root drifted {1000 * max(off_route):.3f} mm off the route"


def test_the_root_stops_at_the_end_of_the_route():
    """Past the far end there is no vessel to prescribe a pose on."""
    unit = rail(CORNER)

    unit.advance(torch.tensor([1.0]), torch.zeros(1), dt=1.0)

    torch.testing.assert_close(unit.depth_m, torch.tensor([0.15]))
    torch.testing.assert_close(unit.root_target()[0], vec(0.1, 0.05, 0.0))


def test_feed_the_route_could_not_take_is_not_reported_as_spent():
    """Recorded insertion has to be the wire's motion, not the keypress."""
    unit = rail(CORNER)

    spent, _ = unit.advance(torch.tensor([1.0]), torch.zeros(1), dt=1.0)

    torch.testing.assert_close(spent, torch.tensor([0.15]))


def test_withdrawal_runs_back_down_the_route():
    unit = rail(CORNER)
    unit.advance(torch.tensor([0.12]), torch.zeros(1), dt=1.0)

    unit.advance(torch.tensor([-0.04]), torch.zeros(1), dt=1.0)

    torch.testing.assert_close(unit.depth_m, torch.tensor([0.08]))


def test_withdrawal_stops_at_the_access_site():
    """The root cannot be pulled back past where the wire entered."""
    unit = rail(CORNER)

    spent, _ = unit.advance(torch.tensor([-1.0]), torch.zeros(1), dt=1.0)

    torch.testing.assert_close(unit.depth_m, torch.zeros(1))
    torch.testing.assert_close(spent, torch.zeros(1))


def test_axial_rotation_rolls_the_root_about_the_route():
    """Rotation rides on the pose because the solver latches one command a step.

    Were it sent as a rate instead, the pose issued in the same step would
    overwrite it and the catheter could not be aimed at a branch at all.
    """
    unit = rail(CORNER)

    unit.advance(torch.zeros(1), torch.full((1,), math.pi / 2.0), dt=1.0)

    _, quat = unit.root_target()
    # The route runs along +X here, so a quarter turn about it takes the
    # frame's local +X onto the direction its local +Y pointed.
    torch.testing.assert_close(
        rotate(quat, vec(0.0, 0.0, 1.0)),
        vec(*X_AXIS),
        atol=1e-6,
        rtol=1e-5,
    )
    assert abs(float(twist_about_axis(frame_along(vec(*X_AXIS)), quat, vec(*X_AXIS))) - math.pi / 2.0) < 1e-5


def test_rotation_accumulates_rather_than_springing_back():
    """Releasing the key holds the aim, the way a rotated wire keeps its aim."""
    unit = rail(CORNER)

    for _ in range(3):
        unit.advance(torch.zeros(1), torch.full((1,), 0.5), dt=1.0)
    unit.advance(torch.zeros(1), torch.zeros(1), dt=1.0)

    _, quat = unit.root_target()
    assert abs(float(twist_about_axis(frame_along(vec(*X_AXIS)), quat, vec(*X_AXIS))) - 1.5) < 1e-5


def test_the_pose_does_not_depend_on_where_the_rod_went():
    """Asked twice without a feed, the answer is the same.

    The point of prescribing from arc length: the pose is a function of the
    feed integral and the route, so it cannot inherit last step's drift.
    """
    unit = rail(ARCH)
    unit.advance(torch.tensor([0.3]), torch.tensor([0.4]), dt=1.0)

    first, first_quat = unit.root_target()
    second, second_quat = unit.root_target()

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_quat, second_quat)


def test_resetting_returns_the_root_to_the_start():
    unit = rail(CORNER)
    unit.advance(torch.tensor([0.1]), torch.tensor([1.0]), dt=1.0)

    unit.reset()

    torch.testing.assert_close(unit.depth_m, torch.zeros(1))
    torch.testing.assert_close(unit.root_target()[1], frame_along(vec(*X_AXIS)))


def test_the_route_is_fed_independently_per_environment():
    unit = rail(CORNER, num_envs=3)

    unit.advance(torch.tensor([0.02, 0.08, 0.14]), torch.zeros(3), dt=1.0)

    torch.testing.assert_close(unit.depth_m, torch.tensor([0.02, 0.08, 0.14]))
    torch.testing.assert_close(
        unit.root_target()[0],
        torch.tensor([[0.02, 0.0, 0.0], [0.08, 0.0, 0.0], [0.1, 0.04, 0.0]]),
    )


def test_resetting_one_environment_leaves_the_others_on_the_route():
    unit = rail(CORNER, num_envs=3)
    unit.advance(torch.tensor([0.02, 0.08, 0.14]), torch.zeros(3), dt=1.0)

    unit.reset(torch.tensor([1]))

    torch.testing.assert_close(unit.depth_m, torch.tensor([0.02, 0.0, 0.14]))


def test_the_rail_reports_the_arc_it_can_feed_along():
    assert rail(CORNER).route_length_m == pytest.approx(0.15)


@pytest.mark.parametrize(
    "route",
    [
        [(0.0, 0.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0)],
        [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]],
    ],
)
def test_the_rail_rejects_a_route_it_cannot_follow(route):
    with pytest.raises(ValueError, match="route must have shape"):
        RouteRailedIntroducer(torch.tensor(route, dtype=torch.float32), 1)


def test_the_rail_rejects_a_repeated_route_vertex():
    """A zero-length segment carries no direction to prescribe a frame from."""
    with pytest.raises(ValueError, match="distinct"):
        rail([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.0)])


def test_the_rail_rejects_a_non_finite_route():
    with pytest.raises(ValueError, match="non-finite"):
        rail([(0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)])


def test_the_rail_rejects_a_non_positive_step():
    with pytest.raises(ValueError, match="dt must be positive"):
        rail(CORNER).advance(torch.zeros(1), torch.zeros(1), dt=0.0)


def test_the_rail_reports_feed_against_the_route(caplog, monkeypatch):
    monkeypatch.setenv(FEED_LOG_ENV_VAR, "0.0001")
    unit = rail(CORNER)

    with caplog.at_level(logging.INFO, logger="i4h_arena.medical.catheter_drive"):
        unit.advance(torch.tensor([1.0]), torch.zeros(1), dt=1.0)

    assert "commanded=1000.00 mm/s" in caplog.text
    assert "spent=150.00 mm/s" in caplog.text
    assert "route=150.0/150 mm" in caplog.text


# --------------------------------------------------------------------------- #
# Wiring the keyboard drive
# --------------------------------------------------------------------------- #
def source_of(path, name: str) -> str:
    """One class or function's source, read without importing Isaac Sim."""
    text = path.read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{name} not found in {path.name}")


def test_the_keyboard_drive_feeds_along_the_route():
    """Insertion has to reach the rail, not the solver's own tangent push.

    Left on the tangent push, everything else here is inert: the root would
    still be stepped along the segment the push bends, which is the loop the
    rail exists to remove.
    """
    body = source_of(PLAIN_CATHETER, "_feed")

    assert "self._rail.advance(" in body
    assert "self._rail.root_target()" in body
    assert "place_proximal" in body


def test_the_keyboard_drive_records_the_feed_the_route_allowed():
    """Not the keypress: a command the route could not take is not insertion."""
    body = source_of(PLAIN_CATHETER, "_feed")

    assert "torch.stack((spent, spin)" in body


def test_the_embodiment_rails_the_drive_onto_the_seeded_path():
    """Arc zero has to be where the root already is.

    Any other polyline and the first prescribed pose teleports the root off the
    shaft it is attached to.
    """
    body = source_of(PLAIN_CATHETER, "_align_to_patient_centerline")

    assert "self.action_config.catheter.route_world_m = self.rod_spec.initial_path_world_m" in body


def test_a_route_less_scene_still_has_a_drive():
    """No centerline means nothing to prescribe against, not a crash."""
    body = source_of(PLAIN_CATHETER, "_build_rail")

    assert "if route is None:" in body
    assert "return None" in body


def test_both_drives_sample_the_probe_before_commanding():
    """A prescribed root is exactly where the diagnostics are load-bearing.

    The wall constraint cannot push back on that particle, so penetration and
    proximal bend only surface in the readout. Moving the keyboard drive onto
    the pose path once dropped it silently.
    """
    for name in ("advance", "place_proximal"):
        body = source_of(XPBD_CATHETER, name)

        assert "self._log_insertion()" in body, f"{name} does not sample insertion"
        assert "self._log_probe()" in body, f"{name} does not sample the probe"
