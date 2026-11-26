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
from pathlib import Path

import h5py
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
    with h5py.File(recording, "w") as handle:
        handle.create_group("data")
    recorded = tmp_path / "recorded.yaml"
    recorded.touch()
    (tmp_path / "run.json").write_text(json.dumps({"patient_twin": str(recorded)}))
    explicit = tmp_path / "explicit.yaml"
    explicit.touch()

    assert _resolve_patient_twin(recording, explicit) == explicit


def test_the_twin_falls_back_to_the_launcher_s_run_json(tmp_path) -> None:
    """The common case takes no flag, because the launcher already wrote the answer down."""
    recording = tmp_path / "demos.hdf5"
    with h5py.File(recording, "w") as handle:
        handle.create_group("data")
    twin = tmp_path / "patient_twin.yaml"
    twin.touch()
    (tmp_path / "run.json").write_text(json.dumps({"patient_twin": str(twin)}))

    assert _resolve_patient_twin(recording, None) == twin


def test_a_missing_twin_says_what_to_do_about_it(tmp_path) -> None:
    """One subject's route against another's recording is a plausible-looking wrong label, so guess nothing."""
    recording = tmp_path / "demos.hdf5"
    with h5py.File(recording, "w") as handle:
        handle.create_group("data")

    with pytest.raises(ValueError, match="--patient-twin"):
        _resolve_patient_twin(recording, None)


def test_a_run_json_without_a_twin_is_not_treated_as_absent(tmp_path) -> None:
    """A recording made without ``--patient-twin`` has no route, and that is worth saying distinctly."""
    recording = tmp_path / "demos.hdf5"
    with h5py.File(recording, "w") as handle:
        handle.create_group("data")
    (tmp_path / "run.json").write_text(json.dumps({"workflow": "endoluminal_navigation"}))

    with pytest.raises(ValueError, match="no patient_twin entry"):
        _resolve_patient_twin(recording, None)


def test_a_twin_that_moved_is_reported_as_moved(tmp_path) -> None:
    """Rather than as a centerline problem, which is what the failure would otherwise look like."""
    recording = tmp_path / "demos.hdf5"
    with h5py.File(recording, "w") as handle:
        handle.create_group("data")
    (tmp_path / "run.json").write_text(json.dumps({"patient_twin": str(tmp_path / "gone.yaml")}))

    with pytest.raises(ValueError, match="no longer exists"):
        _resolve_patient_twin(recording, None)


def test_launcher_sidecar_for_external_recording_is_automatically_resolved(tmp_path, monkeypatch):
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("provenance_launcher", root / "scripts/run_workflow.py")
    launcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "installed_numerical_revision", lambda: "candidate-revision")
    twin = tmp_path / "patient.yaml"
    twin.write_text("patient_id: test\n")
    recording = tmp_path / "external/recording.hdf5"
    built = launcher.plan(
        [
            "endoluminal_navigation",
            "--mode",
            "demo",
            "--record",
            str(recording),
            "--patient-twin",
            str(twin),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    launcher._write_recording_metadata(built)
    assert _resolve_patient_twin(recording, None) == twin
    override = tmp_path / "override.yaml"
    override.write_text(twin.read_text())
    assert _resolve_patient_twin(recording, override) == override
    assert _resolve_patient_twin(recording.with_name("filtered.hdf5"), None) == twin


def test_changed_patient_anatomy_requires_explicit_override(tmp_path):
    import hashlib

    recording = tmp_path / "recording.hdf5"
    twin = tmp_path / "patient.yaml"
    twin.write_text("first anatomy")
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "patient_twin": str(twin),
                "recording": str(recording),
                "patient_twin_sha256": hashlib.sha256(twin.read_bytes()).hexdigest(),
            }
        )
    )
    twin.write_text("different anatomy")
    with pytest.raises(ValueError, match="conflicting patient anatomy provenance"):
        _resolve_patient_twin(recording, None)
    assert _resolve_patient_twin(recording, twin) == twin


@pytest.mark.parametrize("name", ["demos.hdf5", "expanded.hdf5", "filtered.hdf5"])
def test_sibling_recordings_inherit_the_original_patient_manifest(tmp_path, name):
    import hashlib

    import numpy as np

    twin = tmp_path / "patient.yaml"
    twin.write_text("patient_id: same-anatomy\n")
    original = tmp_path / "demos.hdf5"
    source = tmp_path / name
    with h5py.File(source, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((2, 4)))
        if name == "filtered.hdf5":
            data.attrs["filtered_from"] = str(tmp_path / "expanded.hdf5")
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "recording": str(original),
                "patient_twin": str(twin),
                "patient_twin_sha256": hashlib.sha256(twin.read_bytes()).hexdigest(),
            }
        )
    )
    assert _resolve_patient_twin(source, None) == twin
    twin.write_text("patient_id: changed\n")
    with pytest.raises(ValueError, match="conflicting patient anatomy provenance"):
        _resolve_patient_twin(source, None)
    assert _resolve_patient_twin(source, twin) == twin


def test_explicit_patient_cannot_conflict_with_typed_native_provenance(tmp_path):
    import hashlib

    recorded = tmp_path / "recorded.yaml"
    recorded.write_text("patient_id: original\n")
    other = tmp_path / "other.yaml"
    other.write_text("patient_id: other\n")
    recording = tmp_path / "recording.hdf5"
    with h5py.File(recording, "w") as handle:
        handle.create_group("data").attrs["env_args"] = json.dumps(
            dict(
                catheter_backend="numerical",
                patient_twin=str(recorded),
                patient_twin_sha256=hashlib.sha256(recorded.read_bytes()).hexdigest(),
            )
        )
    with pytest.raises(ValueError, match="conflicts.*provenance"):
        _resolve_patient_twin(recording, other)
    assert _resolve_patient_twin(recording, recorded) == recorded
