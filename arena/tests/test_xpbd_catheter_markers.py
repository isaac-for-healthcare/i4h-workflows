# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug marker sizing and the containment probe's switch.

The markers exist to make a sub-millimetre wire visible in the 3D overview, but
the vessels it runs through are only a few millimetres wide. A marker wider than
the lumen reads as wall perforation everywhere the anatomy is tight, which is
precisely where the render is being asked a question it then answers wrongly.
"""

from __future__ import annotations

import pytest

from i4h_arena.medical.xpbd_catheter import (
    _SHAFT_MARKER_CAP_M,
    _TIP_MARKER_CAP_M,
    PROBE_ENV_VAR,
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
