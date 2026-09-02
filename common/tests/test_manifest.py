# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest parsing: one entity per YAML file, named by its filename."""

from __future__ import annotations

import pytest

from i4h_common.manifest import ManifestError, load_scene_manifest, load_scene_spec
from i4h_common.taskdef import TaskDefError, load_taskdef

INPROCESS = """
summary: Close the jaw
impl: i4h_tasks.basic.gripper.grasp:Grasp
"""

REMOTE = """
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


# -- impl decides the runtime --------------------------------------------


def test_impl_means_inprocess(tmp_path):
    runtime, impl, definition = load_taskdef(_write(tmp_path, INPROCESS))
    assert runtime == "inprocess"
    assert impl == "i4h_tasks.basic.gripper.grasp:Grasp"
    assert definition.summary == "Close the jaw"
    assert definition.prompt == ""
    assert definition.requires == {}


def test_no_impl_means_remote(tmp_path):
    runtime, impl, definition = load_taskdef(_write(tmp_path, REMOTE, "scissor.yaml"))
    assert runtime == "remote"
    assert impl is None
    assert definition.embodiment == "so101"
    assert definition.prompt == "Pick up the scissors and place them in the tray"
    assert definition.cameras == ("room", "wrist")
    assert definition.model["repo"] == "nvidia/SO_ARM_Starter_Gr00t"


def test_train_block_marks_a_task_trainable(tmp_path):
    _, _, definition = load_taskdef(_write(tmp_path, REMOTE, "scissor.yaml"))
    assert definition.trainable is True


def test_absent_train_block_means_inference_only(tmp_path):
    body = REMOTE[: REMOTE.index("train:")]
    _, _, definition = load_taskdef(_write(tmp_path, body, "surgical_reach_psm.yaml"))
    assert definition.trainable is False


def test_merged_requires_folds_in_embodiment_and_cameras(tmp_path):
    _, _, definition = load_taskdef(_write(tmp_path, REMOTE, "scissor.yaml"))
    assert definition.merged_requires() == {"embodiment": "so101", "cameras": ["room", "wrist"]}


# -- validation -----------------------------------------------------------


def test_remote_task_must_name_an_embodiment(tmp_path):
    with pytest.raises(TaskDefError, match="must name.*embodiment"):
        load_taskdef(_write(tmp_path, "summary: nothing\n"))


def test_prompt_is_optional_when_summary_is_enough(tmp_path):
    _, _, definition = load_taskdef(_write(tmp_path, "summary: Close the jaw\nimpl: a:B\n"))
    assert definition.prompt == ""


def test_duplicate_prompt_is_rejected(tmp_path):
    with pytest.raises(TaskDefError, match="prompt duplicates summary"):
        load_taskdef(_write(tmp_path, "summary: Close the jaw\nprompt: close the jaw\nimpl: a:B\n"))


def test_every_task_must_define_a_summary(tmp_path):
    with pytest.raises(TaskDefError, match="must define a summary"):
        load_taskdef(_write(tmp_path, "impl: a:B\n"))


def test_a_contradicting_name_key_is_rejected(tmp_path):
    # Two sources of truth is what one-file-per-entity removes.
    with pytest.raises(TaskDefError, match="declares name 'gripper'"):
        load_taskdef(_write(tmp_path, "name: gripper\n" + INPROCESS))


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(TaskDefError, match="unknown keys"):
        load_taskdef(_write(tmp_path, INPROCESS + "embodimant: so101\n"))


def test_runtime_may_not_be_declared(tmp_path):
    # It is derived from impl; accepting both would let them contradict.
    with pytest.raises(TaskDefError, match="unknown keys.*runtime"):
        load_taskdef(_write(tmp_path, "runtime: inprocess\n" + INPROCESS))


def test_malformed_yaml_is_an_error(tmp_path):
    with pytest.raises(TaskDefError):
        load_taskdef(_write(tmp_path, "impl: [unclosed\n"))


def test_a_list_at_the_top_level_is_an_error(tmp_path):
    with pytest.raises(TaskDefError, match="expected a mapping"):
        load_taskdef(_write(tmp_path, "- impl: a:B\n"))


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


def test_scene_mode_override_changes_controller_contract(tmp_path):
    scene = load_scene_spec(
        _write(
            tmp_path,
            SCENE + "mode_overrides:\n  teleop:\n    dof: 23\n",
            "g1_tray.yaml",
        )
    )
    assert scene.dof == 6
    assert scene.for_mode("policy").dof == 6
    assert scene.for_mode("teleop").dof == 23
    assert scene.for_mode("teleop").name == "g1_tray"


def test_scene_mode_override_rejects_structure_changes(tmp_path):
    with pytest.raises(ManifestError, match="unknown keys.*impl"):
        load_scene_spec(
            _write(
                tmp_path,
                SCENE + "mode_overrides:\n  teleop:\n    impl: other:Scene\n",
                "g1_tray.yaml",
            )
        )


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
