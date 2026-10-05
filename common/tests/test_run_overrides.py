# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a run records about the environment it was given."""

from __future__ import annotations

from i4h_common.run_overrides import LAUNCHER_VARIABLES, environment_overrides


def test_the_arrival_tolerance_is_recorded_because_it_defines_success():
    """The flag on a recorded episode means nothing without the tolerance.

    Two of this workflow's demonstrations were labelled successful at 8 mm
    against a 5 mm default, and the only surviving trace of a tolerance in
    ``runs/`` was a hand-named directory.
    """
    recorded = environment_overrides({"I4H_CATHETER_ARRIVAL_MM": "8"})

    assert recorded == {"I4H_CATHETER_ARRIVAL_MM": "8"}


def test_a_knob_added_later_is_captured_without_being_declared():
    """Collected by prefix, so a new override cannot be forgotten here."""
    recorded = environment_overrides({"I4H_CATHETER_A_KNOB_NOBODY_HAS_WRITTEN_YET": "3"})

    assert recorded == {"I4H_CATHETER_A_KNOB_NOBODY_HAS_WRITTEN_YET": "3"}


def test_launcher_wiring_is_not_reported_as_an_override():
    """``run.sh`` sets these to connect the run, not to change it, and the
    paths among them are already recorded verbatim in the same file."""
    environ = dict.fromkeys(LAUNCHER_VARIABLES, "/somewhere")

    assert environment_overrides(environ) == {}


def test_nothing_overridden_reads_as_nothing_rather_than_as_unfilled():
    assert environment_overrides({"PATH": "/usr/bin", "HOME": "/root"}) == {}


def test_an_empty_assignment_is_not_an_override():
    """``I4H_CATHETER_DAMPING=`` leaves the resolver on its default, so
    recording it would claim a change that did not happen."""
    assert environment_overrides({"I4H_CATHETER_DAMPING": "  "}) == {}


def test_the_rl_mirror_agrees_with_this_one():
    """``rl`` depends on PyYAML alone and cannot import this module, so the
    scan exists twice. Pinned here because two copies can drift apart."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl"))
    try:
        from i4h_rl.artifacts import LAUNCHER_VARIABLES as rl_launcher
        from i4h_rl.artifacts import environment_overrides as rl_overrides
    finally:
        sys.path.pop(0)

    environ = {
        "I4H_CATHETER_ARRIVAL_MM": "8",
        "I4H_CATHETER_DAMPING": "0.3",
        "I4H_RUN_DIR": "/runs/x",
        "PATH": "/usr/bin",
    }

    assert rl_launcher == LAUNCHER_VARIABLES
    assert rl_overrides(environ) == environment_overrides(environ)
