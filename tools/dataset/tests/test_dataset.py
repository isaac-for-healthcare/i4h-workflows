# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from i4h_tools.dataset.cli import (
    G1_WBC_ACTION_NAMES,
    _column_names,
    _g1_wbc_policy_actions,
    _to_policy_joint_coordinates,
    _translation,
    _uses_declared_modality,
    _uses_g1_wbc_modality,
    _write_dataset_stats,
    _write_g1_wbc_modality,
    _write_split_modality,
    build_parser,
)


def test_viz_script_resolves_promoted_repository_root() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "viz.sh"
    assignment = next(line for line in script.read_text(encoding="utf-8").splitlines() if line.startswith("REPO_ROOT="))
    relative = assignment.split('${BASH_SOURCE[0]}")/', 1)[1].split('" && pwd)', 1)[0]

    assert (script.parent / relative).resolve() == Path(__file__).resolve().parents[3]


def test_column_names_use_exact_descriptor() -> None:
    config = SimpleNamespace(name="arm", action_names=("a", "b"), state_names=("s",))
    assert _column_names(config, "action", 2) == ["a", "b"]


def test_convert_defaults_to_decord_compatible_video() -> None:
    args = build_parser().parse_args(["convert", "input.hdf5", "output", "--robot", "g1"])
    assert args.video_codec == "h264"


def test_translation_counts_motion_in_either_direction() -> None:
    demo = {
        "obs/robot_pos": np.array(
            [
                [1.8, 0.0, 0.78],
                [0.8, 0.0, 0.78],
            ],
            dtype=np.float32,
        )
    }
    dx, dy, total = _translation(demo)
    assert dx == pytest.approx(-1.0)
    assert dy == pytest.approx(0.0)
    assert total == pytest.approx(1.0)


def test_column_names_append_g1_wbc_tail() -> None:
    joints = tuple(f"joint_{index}" for index in range(43))
    config = SimpleNamespace(name="g1", action_names=joints, state_names=joints)
    assert _column_names(config, "action", 50) == [*joints, *G1_WBC_ACTION_NAMES]


def test_column_names_fall_back_to_positions() -> None:
    config = SimpleNamespace(name="panda", action_names=(), state_names=())
    assert _column_names(config, "state", 3) == ["state_0", "state_1", "state_2"]


def test_g1_wbc_policy_actions_lift_teleop_contract() -> None:
    actions = np.zeros((2, 23), dtype=np.float32)
    actions[:, 16:23] = np.array([0.3, -0.1, 0.2, 0.75, 0.01, 0.02, 0.03], dtype=np.float32)
    states = np.arange(86, dtype=np.float32).reshape(2, 43)
    mapped = _g1_wbc_policy_actions(actions, states)
    assert mapped.shape == (2, 50)
    assert np.array_equal(mapped[:, :43], states)
    assert np.array_equal(mapped[:, 43:50], actions[:, 16:23])


def test_policy_joint_coordinates_apply_declared_calibration() -> None:
    config = SimpleNamespace(
        name="arm",
        isaaclab_joint_pos_limit_range=((-180.0, 180.0), (-90.0, 90.0)),
        lerobot_joint_pos_limit_range=((-100.0, 100.0), (0.0, 100.0)),
    )
    values = np.array([[0.0, -np.pi / 2], [np.pi, np.pi / 2]], dtype=np.float32)
    converted = _to_policy_joint_coordinates(values, config)
    np.testing.assert_allclose(converted, [[0.0, 0.0], [100.0, 100.0]], atol=1e-5)


def test_policy_joint_coordinates_leave_non_joint_tail_contract_unchanged() -> None:
    config = SimpleNamespace(
        name="mobile_arm",
        isaaclab_joint_pos_limit_range=((-180.0, 180.0),),
        lerobot_joint_pos_limit_range=((-100.0, 100.0),),
    )
    values = np.array([[0.0, 0.5]], dtype=np.float32)
    assert _to_policy_joint_coordinates(values, config) is values


def test_write_g1_wbc_modality(tmp_path) -> None:
    (tmp_path / "meta").mkdir()
    path = _write_g1_wbc_modality(tmp_path)
    content = path.read_text()
    assert '"left_arm"' in content
    assert '"navigate_command"' in content
    assert '"observation.images.head"' in content


def test_write_declared_so101_modality(tmp_path) -> None:
    (tmp_path / "meta").mkdir()
    config = SimpleNamespace(
        state_split=(("single_arm", 0, 5), ("gripper", 5, 6)),
        action_split=(("single_arm", 0, 5), ("gripper", 5, 6)),
    )

    assert _uses_declared_modality(config=config, state_width=6, action_width=6)
    path = _write_split_modality(tmp_path, config=config, cameras=("room", "wrist"))
    content = path.read_text()
    assert '"single_arm"' in content
    assert '"gripper"' in content
    assert '"observation.images.room"' in content
    assert '"observation.images.wrist"' in content


def test_declared_modality_requires_complete_tensor_coverage() -> None:
    config = SimpleNamespace(
        state_split=(("arm", 0, 5),),
        action_split=(("arm", 0, 5),),
    )
    assert not _uses_declared_modality(config=config, state_width=6, action_width=5)


@pytest.mark.parametrize(
    ("robot", "state_width", "action_width", "expected"),
    [
        ("g1", 43, 50, True),
        ("g1", 43, 23, False),
        ("g1", 50, 50, False),
        ("so101", 43, 50, False),
    ],
)
def test_uses_g1_wbc_modality(
    robot: str,
    state_width: int,
    action_width: int,
    expected: bool,
) -> None:
    assert (
        _uses_g1_wbc_modality(
            robot=robot,
            state_width=state_width,
            action_width=action_width,
        )
        is expected
    )


def test_write_dataset_stats(tmp_path) -> None:
    stats = {
        "action": {
            "mean": np.array([0.1, 0.2], dtype=np.float32),
            "std": np.array([0.3, 0.4], dtype=np.float32),
        }
    }
    path = _write_dataset_stats(tmp_path, stats)
    content = path.read_text()
    assert '"action"' in content
    assert '"mean"' in content
    assert "0.1" in content
