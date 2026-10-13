# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conversion's half of the derived navigation goal columns.

The arithmetic is tested in ``common/tests/test_navigation_route.py``. What is
tested here is the wiring: which descriptors ask for the columns, where the twin
comes from, and that the descriptor and the derivation still agree about how
many columns there are and what they are called.
"""

from __future__ import annotations

import json

import pytest

from i4h_common.config import get_robot_config
from i4h_common.navigation_route import GOAL_COLUMN_NAMES
from i4h_tools.dataset.cli import (
    _declares_derived_goal_columns,
    _resolve_patient_twin,
    _split_width,
)


def test_the_catheter_descriptor_and_the_derivation_agree() -> None:
    """The contract that makes the whole thing work, pinned in one place.

    ``state_names`` has to end in exactly the columns the derivation appends,
    and ``state_split`` has to tile the result. If either side moves alone the
    conversion either raises or -- worse -- mislabels, so this is asserted
    rather than left to the descriptor's comment.
    """
    config = get_robot_config("catheter")
    declared = tuple(config.state_names)

    assert declared[-len(GOAL_COLUMN_NAMES) :] == GOAL_COLUMN_NAMES
    assert _split_width(config.state_split) == len(declared)
    # Four real drive channels recorded, five derived on top.
    assert len(declared) - len(GOAL_COLUMN_NAMES) == 4


def test_the_arm_descriptor_asks_for_the_same_columns() -> None:
    """Covered by the declared names rather than by being named, so it needs no second code path."""
    config = get_robot_config("franka_catheter")

    assert _declares_derived_goal_columns(config)
    assert tuple(config.state_names)[-len(GOAL_COLUMN_NAMES) :] == GOAL_COLUMN_NAMES


@pytest.mark.parametrize("robot", ["g1", "panda", "dvrk_psm"])
def test_unrelated_descriptors_are_not_handed_goal_columns(robot: str) -> None:
    """A descriptor merely short of its own split must not be quietly given a navigation goal."""
    assert not _declares_derived_goal_columns(get_robot_config(robot))


def test_an_explicit_twin_wins_over_the_recorded_one(tmp_path) -> None:
    """So a moved or re-exported twin can be pointed at without editing run.json."""
    recording = tmp_path / "demos.hdf5"
    recording.touch()
    recorded = tmp_path / "recorded.yaml"
    recorded.touch()
    (tmp_path / "run.json").write_text(json.dumps({"patient_twin": str(recorded)}))
    explicit = tmp_path / "explicit.yaml"
    explicit.touch()

    assert _resolve_patient_twin(recording, explicit) == explicit


def test_the_twin_falls_back_to_the_launcher_s_run_json(tmp_path) -> None:
    """The common case takes no flag, because the launcher already wrote the answer down."""
    recording = tmp_path / "demos.hdf5"
    recording.touch()
    twin = tmp_path / "patient_twin.yaml"
    twin.touch()
    (tmp_path / "run.json").write_text(json.dumps({"patient_twin": str(twin)}))

    assert _resolve_patient_twin(recording, None) == twin


def test_a_missing_twin_says_what_to_do_about_it(tmp_path) -> None:
    """One subject's route against another's recording is a plausible-looking wrong label, so guess nothing."""
    recording = tmp_path / "demos.hdf5"
    recording.touch()

    with pytest.raises(ValueError, match="--patient-twin"):
        _resolve_patient_twin(recording, None)


def test_a_run_json_without_a_twin_is_not_treated_as_absent(tmp_path) -> None:
    """A recording made without ``--patient-twin`` has no route, and that is worth saying distinctly."""
    recording = tmp_path / "demos.hdf5"
    recording.touch()
    (tmp_path / "run.json").write_text(json.dumps({"workflow": "endoluminal_navigation"}))

    with pytest.raises(ValueError, match="no patient_twin entry"):
        _resolve_patient_twin(recording, None)


def test_a_twin_that_moved_is_reported_as_moved(tmp_path) -> None:
    """Rather than as a centerline problem, which is what the failure would otherwise look like."""
    recording = tmp_path / "demos.hdf5"
    recording.touch()
    (tmp_path / "run.json").write_text(json.dumps({"patient_twin": str(tmp_path / "gone.yaml")}))

    with pytest.raises(ValueError, match="no longer exists"):
        _resolve_patient_twin(recording, None)
