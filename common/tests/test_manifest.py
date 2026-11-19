# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest parsing: one entity per YAML file, named by its filename."""

from __future__ import annotations

import pytest

from i4h_common.manifest import ManifestError, load_scene_manifest, load_scene_spec
from i4h_common.taskdef import TaskDefError, load_taskdef

TASK = """
summary: Grip the scissors and put them into the tray
prompt: Pick up the scissors and place them in the tray
embodiment: so101
cameras: [room, wrist]
model:
  repo: nvidia/SO_ARM_Starter_Gr00t
train:
  max_steps: 10000
"""

SCENE = """
impl: i4h_arena.scenes.soarm_scissors:SoArmScissorsScene
embodiment: so101
action_space: joint_position
dof: 6
cameras: [room, wrist]
objects: [table, scissors, tray]
"""


def _write(tmp_path, body, name="grasp.yaml"):
    path = tmp_path / name
    path.write_text(body)
    return path


# -- policy tasks ----------------------------------------------------------


def test_task_declaration(tmp_path):
    definition = load_taskdef(_write(tmp_path, TASK, "scissor.yaml"))
    assert definition.embodiment == "so101"
    assert definition.prompt == "Pick up the scissors and place them in the tray"
    assert definition.cameras == ("room", "wrist")
    assert definition.model["repo"] == "nvidia/SO_ARM_Starter_Gr00t"


def test_train_block_marks_a_task_trainable(tmp_path):
    assert load_taskdef(_write(tmp_path, TASK, "scissor.yaml")).trainable is True


def test_absent_train_block_means_inference_only(tmp_path):
    body = TASK[: TASK.index("train:")]
    assert load_taskdef(_write(tmp_path, body, "surgical_reach_psm.yaml")).trainable is False


def test_merged_requires_folds_in_embodiment_and_cameras(tmp_path):
    definition = load_taskdef(_write(tmp_path, TASK, "scissor.yaml"))
    assert definition.merged_requires() == {"embodiment": "so101", "cameras": ["room", "wrist"]}


# -- validation -----------------------------------------------------------


def test_a_task_must_name_an_embodiment(tmp_path):
    with pytest.raises(TaskDefError, match="must name the embodiment"):
        load_taskdef(_write(tmp_path, "summary: nothing\n"))


def test_prompt_is_optional_when_summary_is_enough(tmp_path):
    assert load_taskdef(_write(tmp_path, "summary: Close the jaw\nembodiment: psm\n")).prompt == ""


def test_duplicate_prompt_is_rejected(tmp_path):
    with pytest.raises(TaskDefError, match="prompt duplicates summary"):
        load_taskdef(_write(tmp_path, "summary: Close the jaw\nprompt: close the jaw\nembodiment: psm\n"))


def test_every_task_must_define_a_summary(tmp_path):
    with pytest.raises(TaskDefError, match="must define a summary"):
        load_taskdef(_write(tmp_path, "embodiment: psm\n"))


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(TaskDefError, match="unknown keys"):
        load_taskdef(_write(tmp_path, TASK + "embodimant: so101\n"))


def test_malformed_yaml_is_an_error(tmp_path):
    with pytest.raises(TaskDefError):
        load_taskdef(_write(tmp_path, "summary: [unclosed\n"))


def test_a_list_at_the_top_level_is_an_error(tmp_path):
    with pytest.raises(TaskDefError, match="expected a mapping"):
        load_taskdef(_write(tmp_path, "- summary: a\n"))


# -- scenes ---------------------------------------------------------------


def test_scene_spec(tmp_path):
    scene = load_scene_spec(_write(tmp_path, SCENE, "soarm_scissors.yaml"))
    assert scene.name == "soarm_scissors"
    provides = scene.provides()
    assert provides["embodiment"] == "so101"
    assert provides["cameras"] == ["room", "wrist"]
    assert provides["robots"] == ["robot"]


def test_scene_spec_preserves_explicit_empty_robots(tmp_path):
    scene = load_scene_spec(
        _write(
            tmp_path,
            SCENE + "robots: []\n",
            "blank.yaml",
        )
    )
    assert scene.robots == ()
    assert scene.provides()["robots"] == []


def test_scene_missing_field(tmp_path):
    with pytest.raises(ManifestError, match="missing embodiment"):
        load_scene_spec(_write(tmp_path, "impl: a:B\n", "x.yaml"))


def test_scene_directory_scan(tmp_path):
    _write(tmp_path, SCENE, "soarm_scissors.yaml")
    _write(tmp_path, SCENE.replace("so101", "dvrk_psm"), "psm_reach.yaml")
    scenes = load_scene_manifest(tmp_path)
    assert sorted(s.name for s in scenes) == ["psm_reach", "soarm_scissors"]


def test_scene_directory_must_exist(tmp_path):
    with pytest.raises(ManifestError, match="no such manifest directory"):
        load_scene_manifest(tmp_path / "nope")


PHANTOM_SCENE = """\
impl: i4h_arena.scenes.panda_phantom:PandaPhantomScene
embodiment: panda
action_space: ee_pose
dof: 6
cameras: [room, wrist]
"""


def test_a_scene_declares_how_its_recordings_are_graded(tmp_path):
    body = (
        PHANTOM_SCENE
        + "annotate:\n  criterion: >-\n    The probe rests\n    on the phantom.\n"
        + "  cameras: [room, wrist]\n  judge: trajectory\n"
    )
    spec = load_scene_spec(_write(tmp_path, body, "panda_phantom.yaml"))

    assert spec.annotate is not None
    assert spec.annotate.criterion == "The probe rests on the phantom."
    assert spec.annotate.cameras == ("room", "wrist")
    assert spec.annotate.judge == "trajectory"
    assert load_scene_spec(_write(tmp_path, PHANTOM_SCENE, "psm_reach.yaml")).annotate is None


@pytest.mark.parametrize(
    ("annotate", "message"),
    [
        ("  cameras: [room, overview]\n", r"\['overview'\] the scene does not record \['room', 'wrist'\]"),
        ("  judge: final\n", "judge"),
        ("  cameras: room\n", "sequence of camera names"),
    ],
)
def test_a_scene_cannot_grade_with_what_it_does_not_record(tmp_path, annotate, message):
    with pytest.raises(ManifestError, match=message):
        load_scene_spec(_write(tmp_path, PHANTOM_SCENE + "annotate:\n" + annotate, "panda_phantom.yaml"))
