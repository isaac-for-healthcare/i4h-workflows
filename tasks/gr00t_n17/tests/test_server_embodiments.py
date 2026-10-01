# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The N1.7 backend has to serve more than one shape of embodiment.

It used to report ``joint_position`` for every checkpoint and read the state as
``single_arm`` plus a trailing ``gripper``. That is true of the SO-ARM and of
nothing else, and because the runtime refuses a checkpoint whose action space
differs from the scene's, it made any other embodiment unservable. These tests
hold both shapes at once: whatever changes for the catheter must leave the arm
reporting exactly what it reported before.

Imports the module without importing ``gr00t``, which is only reached inside
``load()``, so this runs in the light venv like ``test_gr00t_n15_finetune.py``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from i4h_common.bus.inproc import InProcBus
from i4h_common.config import get_robot_config
from i4h_common.server import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17.server import (  # noqa: E402
    Gr00tN17Server,
    _is_calibrated,
    _modality_keys,
    _register_modalities,
)


@pytest.fixture
def server():
    return Gr00tN17Server(namespace="test", bus=InProcBus())


def session(embodiment: str, **model) -> Session:
    return Session(
        task_uid="uid",
        task_id=f"gr00t_n17/{embodiment}",
        run_id="run",
        episode_index=0,
        prompt="",
        checkpoint="",
        embodiment=embodiment,
        model=model,
    )


# -- group names ---------------------------------------------------------


def test_the_catheter_groups_come_from_its_splits() -> None:
    assert _modality_keys(get_robot_config("catheter")) == ("catheter", "carm")


def test_the_arm_groups_are_unchanged() -> None:
    assert _modality_keys(get_robot_config("so101")) == ("single_arm", "gripper")


def test_the_carrier_arm_shares_the_catheter_groups() -> None:
    """One checkpoint serves both endoluminal scenes, so the group names the
    backend slices into have to be the same in both. Without splits of its own
    this embodiment fell through to the SO-ARM layout and the checkpoint's
    `action.catheter` was looked up as `action.single_arm`."""
    assert _modality_keys(get_robot_config("franka_catheter")) == ("catheter", "carm")


def test_an_embodiment_with_no_splits_falls_back_to_the_arm_layout() -> None:
    """Every checkpoint predating the splits behaved this way implicitly."""
    bare = replace(get_robot_config("so101"), action_split=())
    assert _modality_keys(bare) == ("single_arm", "gripper")


# -- joint calibration ---------------------------------------------------


def test_the_arm_needs_its_motor_calibration() -> None:
    assert _is_calibrated(get_robot_config("so101"))


def test_the_catheter_does_not() -> None:
    """Metres and radians per second were recorded in the scene's own units."""
    assert not _is_calibrated(get_robot_config("catheter"))


def test_a_half_declared_calibration_is_still_rejected(server, monkeypatch) -> None:
    """Dropping the requirement outright would turn this into silent
    passthrough, which is worse than the error it replaced."""
    from i4h_tasks.gr00t_n17 import server as module

    half = replace(get_robot_config("so101"), lerobot_joint_pos_limit_range=())
    monkeypatch.setattr(module, "get_robot_config", lambda _name: half)
    with pytest.raises(ValueError, match="only one of"):
        server.action_contract(session("so101"))


def test_an_absent_embodiment_is_rejected(server) -> None:
    with pytest.raises(ValueError, match="no embodiment"):
        server.action_contract(session(""))


# -- the action contract -------------------------------------------------


def test_the_catheter_reports_its_velocity_space(server) -> None:
    contract = server.action_contract(session("catheter", action_space="catheter_carm_velocity"))
    assert contract.space == "catheter_carm_velocity"
    assert contract.gripper == "none"
    assert contract.dof == 4


def test_the_arm_still_reports_joint_positions_and_a_gripper(server) -> None:
    """No `action_space` in the manifest, exactly as the SO-ARM task has it."""
    contract = server.action_contract(session("so101"))
    assert contract.space == "joint_position"
    assert contract.gripper == "last"


def test_the_carrier_arm_reports_four_channels_not_eleven_joints(server) -> None:
    """`dof` counts what is commanded, which is not what is articulated.

    The Franka that carries the drive unit is servo'd along the introducer, so
    it adds seven joints and no commands. Counting joints here reported 11
    against a scene that accepts 4, and the runtime rejected the checkpoint for
    not matching the scene it was built for.
    """
    contract = server.action_contract(session("franka_catheter", action_space="catheter_carm_velocity"))
    assert contract.dof == 4
    assert contract.space == "catheter_carm_velocity"
    assert contract.gripper == "none"


def test_an_embodiment_that_commands_every_joint_is_unchanged(server) -> None:
    """Joints and actions agree for the SO-ARM, so the count it reports must
    not move with the change that made the servo'd arm report fewer."""
    assert server.action_contract(session("so101")).dof == 6
    assert server.action_contract(session("catheter", action_space="catheter_carm_velocity")).dof == 4


# -- observation shaping -------------------------------------------------


def test_the_catheter_state_splits_into_instrument_and_gantry(server) -> None:
    state = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)
    groups = server._state_groups(state, get_robot_config("catheter"))
    assert sorted(groups) == ["carm", "catheter"]
    assert groups["catheter"].shape == (1, 1, 3)
    assert groups["carm"].shape == (1, 1, 1)
    np.testing.assert_allclose(groups["catheter"][0, 0], [0.10, 0.20, 0.30])
    np.testing.assert_allclose(groups["carm"][0, 0], [0.40])


def test_the_carrier_arm_joints_arrive_as_a_third_group(server) -> None:
    """The splits have to tile the whole vector for conversion to use them, so
    the servo'd joints are declared rather than dropped. The leading two groups
    stay byte-identical to the armless case; GR00T's loader finds no `arm` key
    in the modality config it was given and proceeds without it.
    """
    state = np.arange(11, dtype=np.float32) / 100.0
    groups = server._state_groups(state, get_robot_config("franka_catheter"))
    assert sorted(groups) == ["arm", "carm", "catheter"]
    assert groups["catheter"].shape == (1, 1, 3)
    assert groups["carm"].shape == (1, 1, 1)
    assert groups["arm"].shape == (1, 1, 7)
    np.testing.assert_allclose(groups["catheter"][0, 0], [0.00, 0.01, 0.02])
    np.testing.assert_allclose(groups["carm"][0, 0], [0.03])


def test_the_arm_state_still_splits_five_and_one(server) -> None:
    state = np.arange(6, dtype=np.float32)
    groups = server._state_groups(state, get_robot_config("so101"))
    assert groups["single_arm"].shape == (1, 1, 5)
    assert groups["gripper"].shape == (1, 1, 1)
    np.testing.assert_allclose(groups["gripper"][0, 0], [5.0])


def test_the_action_chunk_is_concatenated_in_group_order(server) -> None:
    """Order follows the splits, so it matches the columns the scene expects."""
    chunk = {
        "action.catheter": np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float32),
        "action.carm": np.array([0.5, 0.6], dtype=np.float32),
    }
    actions = server._flatten(chunk, ("catheter", "carm"))
    assert actions.shape == (2, 4)
    np.testing.assert_allclose(actions[0], [0.01, 0.02, 0.03, 0.5])
    np.testing.assert_allclose(actions[1], [0.04, 0.05, 0.06, 0.6])


def test_a_missing_action_group_is_an_error(server) -> None:
    with pytest.raises(KeyError, match="carm"):
        server._flatten({"action.catheter": np.zeros((2, 2))}, ("catheter", "carm"))


# -- modality registration -----------------------------------------------


def test_an_unknown_modality_config_names_the_alternatives() -> None:
    """A manifest typo should not surface as a bare ImportError."""
    with pytest.raises(ValueError, match="config_catheter"):
        _register_modalities("config_no_such_embodiment")
