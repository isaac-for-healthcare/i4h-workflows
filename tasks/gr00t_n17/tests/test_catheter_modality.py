# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin the catheter task declaration against the things it has to agree with.

Every mismatch here fails silently at training time. A modality key the dataset
does not carry is not an error to GR00T -- the loader finds no such key and the
run proceeds on whatever is left -- so the group names, the camera and the
control rate are asserted against the embodiment and scene manifests that
produced the data rather than trusted to stay in step.

Most of this reads manifests and runs anywhere, following
``test_gr00t_n15_finetune.py`` in importing from the stack without importing
the stack's heavy dependencies. The assertions that need the registered
modality config itself skip where ``gr00t`` is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from i4h_common.config import get_robot_config
from i4h_common.paths import workflow_root
from i4h_common.training import task_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASK_ID = "gr00t_n17/catheter_navigation"
CAMERA = "fluoroscopy"


@pytest.fixture
def spec():
    return task_spec(TASK_ID)


@pytest.fixture
def catheter():
    return get_robot_config("catheter")


@pytest.fixture
def modality():
    pytest.importorskip("gr00t", reason="registering modalities needs the gr00t_n17 venv")
    from i4h_tasks.gr00t_n17.config_catheter import CATHETER_CONFIG

    return CATHETER_CONFIG


def scene_manifest(name: str) -> dict:
    path = workflow_root() / "arena" / "i4h_arena" / "scenes" / "manifest" / f"{name}.yaml"
    return yaml.safe_load(path.read_text()) or {}


def test_the_task_is_finetunable(spec) -> None:
    """A `train:` block is what marks a task trainable rather than serve-only."""
    assert spec.trainable


def test_the_task_names_the_catheter_embodiment(spec) -> None:
    assert spec.embodiment == "catheter"


def test_the_task_asks_for_the_velocity_action_space(spec) -> None:
    """Insertion and rotation are rates; a joint-position scene is not a match."""
    assert spec.requires.get("action_space") == "catheter_carm_velocity"


def test_the_modality_groups_match_the_embodiment_splits(modality, catheter) -> None:
    """GR00T reads these keys out of the dataset's modality.json, which
    conversion writes from the embodiment's splits."""
    declared = [name for name, _start, _end in catheter.action_split]
    assert list(modality["action"].modality_keys) == declared
    assert list(modality["state"].modality_keys) == [name for name, _s, _e in catheter.state_split]


def test_the_video_key_matches_the_camera_the_task_declares(modality, spec) -> None:
    assert list(modality["video"].modality_keys) == list(spec.cameras) == [CAMERA]


def test_every_action_group_has_an_action_config(modality) -> None:
    """Zipped positionally by GR00T, so a missing entry silently shifts groups."""
    action = modality["action"]
    assert action.action_configs is not None
    assert len(action.action_configs) == len(action.modality_keys)


def test_the_actions_are_absolute_not_relative(modality) -> None:
    """These are velocity commands the environment integrates, so an action is
    not an offset from the current state and must not get relative stats."""
    ActionRepresentation = pytest.importorskip("gr00t.data.types").ActionRepresentation

    for config in modality["action"].action_configs:
        assert config.rep is ActionRepresentation.ABSOLUTE


def test_the_action_horizon_matches_the_declared_delta_indices(modality, spec) -> None:
    assert len(modality["action"].delta_indices) == spec.model["action_horizon"]


def test_the_control_rate_matches_the_scene_that_records_the_data(spec) -> None:
    """Training on data recorded at a different rate than the manifest claims
    is the kind of error that shows up as a policy that moves at the wrong
    speed, so it is cheaper to assert than to debug."""
    scene = scene_manifest("endoluminal_navigation")
    assert spec.model["control_hz"] == scene["control_hz"]


def test_the_state_names_match_the_embodiment(spec, catheter) -> None:
    assert tuple(spec.observation["state_names"]) == catheter.state_names


def test_every_state_column_past_the_joints_has_a_term_to_fill_it(spec, catheter) -> None:
    """Declaring wider state than the joints provide requires saying where the
    rest comes from, or a rollout hands the checkpoint a short vector and the
    groups past the end arrive empty.

    Only the count is asserted. Which Scene term supplies which column is the
    Scene's business, and pinning the names here would duplicate the manifest
    rather than check it.
    """
    extra = len(spec.observation["state_names"]) - len(catheter.joint_names)
    terms = spec.observation.get("state_terms", ())
    if extra <= 0:
        assert not terms, "nothing to fill, so naming terms would widen the vector past its declaration"
        return
    assert terms, f"{extra} state columns past the joints and no state_terms to supply them"


def test_the_manifest_modality_config_resolves(spec) -> None:
    """The trainer turns this name into a file path and the launcher loads it."""
    from i4h_tasks.gr00t_n17.train import _modality_config_file

    assert _modality_config_file(spec.model["modality_config"]).is_file()


def test_the_task_reports_the_velocity_action_space_to_the_runtime(spec) -> None:
    """`requires` is the scene-matching side; this is what the backend reports
    on the ready handshake, and the runtime refuses the pair if they disagree
    with the scene."""
    assert spec.model["action_space"] == spec.requires["action_space"]


def test_an_unknown_modality_config_is_rejected() -> None:
    from i4h_tasks.gr00t_n17.train import _modality_config_file

    with pytest.raises(FileNotFoundError):
        _modality_config_file("config_no_such_embodiment")
