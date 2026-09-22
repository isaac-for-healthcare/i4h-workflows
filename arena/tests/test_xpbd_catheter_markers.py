# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug marker sizing, the containment probe's switch, and the insertion log.

The markers exist to make a sub-millimetre wire visible in the 3D overview, but
the vessels it runs through are only a few millimetres wide. A marker wider than
the lumen reads as wall perforation everywhere the anatomy is tight, which is
precisely where the render is being asked a question it then answers wrongly.

The insertion log separates a feed that was never commanded from one the drive
delivered to a wire that did not carry it to the tip. Both look like a stalled
catheter on the fluoroscopy view, and they have different causes.
"""

from __future__ import annotations

import numpy as np
import pytest

from i4h_arena.medical.xpbd_catheter import (
    _SHAFT_MARKER_CAP_M,
    _TIP_MARKER_CAP_M,
    INSERTION_LOG_ENV_VAR,
    PROBE_ENV_VAR,
    axial_rate,
    insertion_log_seconds,
    marker_radius_m,
    probe_interval,
)

# The narrowest lumen radius on the s0011 aorto-iliac route.
NARROWEST_LUMEN_RADIUS_M = 0.003
GUIDEWIRE_RADIUS_M = 0.0005


def test_the_guidewire_marker_is_inflated_for_legibility():
    radius = marker_radius_m(GUIDEWIRE_RADIUS_M, inflation=3.0, cap_m=_SHAFT_MARKER_CAP_M)

    assert radius > GUIDEWIRE_RADIUS_M
    assert radius == pytest.approx(0.0015)


@pytest.mark.parametrize("inflation, cap", [(3.0, _SHAFT_MARKER_CAP_M), (6.0, _TIP_MARKER_CAP_M)])
def test_no_marker_outgrows_the_narrowest_vessel(inflation, cap):
    """Otherwise the marker manufactures perforation in the iliac."""
    radius = marker_radius_m(GUIDEWIRE_RADIUS_M, inflation=inflation, cap_m=cap)

    assert radius <= NARROWEST_LUMEN_RADIUS_M


def test_the_cap_binds_on_a_wire_thick_enough_to_reach_it():
    radius = marker_radius_m(0.002, inflation=6.0, cap_m=_TIP_MARKER_CAP_M)

    assert radius == pytest.approx(_TIP_MARKER_CAP_M)


def test_a_thick_catheter_is_never_drawn_thinner_than_it_is():
    """A 6 Fr catheter is already wider than the cap; shrinking it would lie."""
    radius = marker_radius_m(0.00105, inflation=3.0, cap_m=_SHAFT_MARKER_CAP_M)

    assert radius >= 0.00105


@pytest.mark.parametrize("kwargs", [{"radius_m": 0.0}, {"radius_m": -0.001}])
def test_a_nonphysical_radius_is_rejected(kwargs):
    with pytest.raises(ValueError, match="radius_m must be positive"):
        marker_radius_m(inflation=3.0, cap_m=_SHAFT_MARKER_CAP_M, **kwargs)


def test_shrinking_a_marker_is_not_inflation():
    with pytest.raises(ValueError, match="inflation must be at least 1"):
        marker_radius_m(0.0005, inflation=0.5, cap_m=_SHAFT_MARKER_CAP_M)


# --------------------------------------------------------------------------- #
# Probe switch
# --------------------------------------------------------------------------- #
def test_the_probe_is_off_unless_asked_for():
    assert probe_interval({}) == 0


def test_the_probe_reports_every_n_steps():
    assert probe_interval({PROBE_ENV_VAR: "60"}) == 60


@pytest.mark.parametrize("value", ["", "0", "-5", "every", "1.5"])
def test_an_unusable_probe_setting_just_stays_off(value):
    """A malformed diagnostic must not take a simulator run down with it."""
    assert probe_interval({PROBE_ENV_VAR: value}) == 0


# --------------------------------------------------------------------------- #
# Insertion log
# --------------------------------------------------------------------------- #
def test_the_insertion_log_is_off_until_an_interval_is_asked_for():
    assert insertion_log_seconds({}) == 0.0


@pytest.mark.parametrize("value, expected", [("1", 1.0), ("0.5", 0.5), (" 2.5 ", 2.5)])
def test_the_insertion_log_takes_an_interval_in_seconds(value, expected):
    assert insertion_log_seconds({INSERTION_LOG_ENV_VAR: value}) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "0", "-1", "sometimes"])
def test_an_unusable_insertion_setting_just_stays_off(value):
    assert insertion_log_seconds({INSERTION_LOG_ENV_VAR: value}) == 0.0


def test_a_tip_moving_with_the_wire_reads_as_a_positive_rate():
    rate = axial_rate(np.array([0.009, 0.0, 0.0]), np.array([0.05, 0.0, 0.0]), 1.0)

    assert rate == pytest.approx(0.009)


def test_a_retreating_tip_reads_as_a_negative_rate():
    """The reading the whole log exists for.

    A magnitude would report this as 0.4 mm/s of travel, which is the one answer
    that would hide a wire handing insertion back.
    """
    rate = axial_rate(np.array([-0.0004, 0.0, 0.0]), np.array([0.05, 0.0, 0.0]), 1.0)

    assert rate == pytest.approx(-0.0004)


def test_only_motion_along_the_wire_counts_as_feed():
    """A tip swinging sideways is not advancing, however far it moved."""
    rate = axial_rate(np.array([0.0, 0.02, 0.0]), np.array([0.05, 0.0, 0.0]), 1.0)

    assert rate == pytest.approx(0.0)


def test_the_rate_is_per_second_not_per_report():
    rate = axial_rate(np.array([0.018, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 2.0)

    assert rate == pytest.approx(0.009)


def test_a_collapsed_tangent_cannot_order_a_sign():
    """Coincident particles do occur; a direction that does not exist is not one."""
    rate = axial_rate(np.array([0.01, 0.0, 0.0]), np.zeros(3), 1.0)

    assert rate == 0.0


def test_a_rate_needs_time_to_have_passed():
    with pytest.raises(ValueError, match="elapsed_s must be positive"):
        axial_rate(np.array([0.01, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0)
