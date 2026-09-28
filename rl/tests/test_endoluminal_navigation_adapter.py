# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the catheter navigation GR00T/RLinf mapping.

The mapping is split across three files that have to agree: the profile
declares widths, the trainer config slices the state into GR00T groups, and
the adapter builds the vector those slices index into. Nothing at runtime
checks that they match -- a wrong slice silently hands GR00T the tip
direction labelled as drive state and training merely goes badly. So the
tests that matter most here compare the YAML against the adapter rather than
exercising either alone.

Only the pure conversion functions are covered. ``register()`` needs RLinf
and a live Isaac Sim, which is the integration this repo cannot run on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from i4h_rl.adapters.endoluminal_navigation import (
    ACTION_DIM,
    ACTION_KEYS,
    OBS_CONVERTER,
    STATE_DIM,
    STATE_KEYS,
    STATE_WIDTHS,
    convert_gr00t_to_workflow_action,
    convert_workflow_obs_to_gr00t,
    wrap_workflow_observation,
)
from i4h_rl.profile import RLProfile

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "rl/profiles/endoluminal_navigation.yaml"
CONFIG_PATH = REPO / "rl/config/endoluminal_navigation_ppo_gr00t.yaml"


@pytest.fixture(scope="module")
def trainer_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def policy_obs() -> dict[str, torch.Tensor]:
    """One observation batch shaped like the Workflow observation manager's."""
    obs = {"fluoroscopy_rgb": torch.zeros(2, 8, 8, 4)}
    for index, (key, width) in enumerate(zip(STATE_KEYS, STATE_WIDTHS, strict=True)):
        obs[key] = torch.full((2, width), float(index + 1))
    return obs


def _bridge(policy_obs):
    return wrap_workflow_observation({"policy": policy_obs}, task_description="navigate", num_envs=2)


# --------------------------------------------------------------------------- #
# The three files agree
# --------------------------------------------------------------------------- #
def test_profile_state_width_matches_the_adapter():
    assert RLProfile.load(PROFILE_PATH).state_dof == STATE_DIM


def test_profile_action_width_matches_the_adapter():
    profile = RLProfile.load(PROFILE_PATH)
    assert profile.action_dof == ACTION_DIM
    # Every catheter channel is commanded, so nothing is padded away.
    assert profile.policy_action_dof == profile.action_dof


def test_profile_points_at_this_adapter_and_config():
    profile = RLProfile.load(PROFILE_PATH)
    assert profile.adapter_module == "i4h_rl.adapters.endoluminal_navigation"
    assert profile.trainer_config.resolve() == CONFIG_PATH.resolve()


def test_trainer_config_state_order_matches_the_adapter(trainer_config):
    """A reordered ``states`` list silently relabels every GR00T state group."""
    declared = [entry["key"] for entry in trainer_config["env"]["train"]["isaaclab"]["states"]]
    assert tuple(declared) == STATE_KEYS


def test_trainer_config_slices_match_the_adapter_layout(trainer_config):
    """The YAML slices must index the vector the adapter actually concatenates."""
    boundaries = np.cumsum((0,) + STATE_WIDTHS)
    expected = {
        "state.catheter": [int(boundaries[0]), int(boundaries[1])],
        "state.tip_pose": [int(boundaries[1]), int(boundaries[3])],
        "state.navigation": [int(boundaries[3]), int(boundaries[5])],
    }
    declared = {
        entry["gr00t_key"]: entry["slice"]
        for entry in trainer_config["env"]["train"]["isaaclab"]["gr00t_mapping"]["state"]
    }
    assert declared == expected


def test_trainer_config_slices_tile_the_whole_state(trainer_config):
    """No gap and no overlap: every state number reaches exactly one GR00T group."""
    slices = sorted(entry["slice"] for entry in trainer_config["env"]["train"]["isaaclab"]["gr00t_mapping"]["state"])
    assert slices[0][0] == 0
    assert slices[-1][1] == STATE_DIM
    for earlier, later in zip(slices, slices[1:], strict=False):
        assert earlier[1] == later[0]


def test_trainer_config_declares_this_obs_converter(trainer_config):
    assert trainer_config["env"]["train"]["isaaclab"]["obs_converter_type"] == OBS_CONVERTER


def test_trainer_config_pads_nothing(trainer_config):
    mapping = trainer_config["env"]["train"]["isaaclab"]["action_mapping"]
    assert mapping["prefix_pad"] == 0
    assert mapping["suffix_pad"] == 0


def test_trainer_config_action_dim_matches_the_adapter(trainer_config):
    assert trainer_config["actor"]["model"]["action_dim"] == ACTION_DIM


def test_eval_env_shares_the_train_mapping(trainer_config):
    """The YAML anchor must actually be reused, or eval drifts from training."""
    env = trainer_config["env"]
    assert env["eval"]["isaaclab"] == env["train"]["isaaclab"]


def test_episode_cap_is_not_raised_above_the_scene_budget(trainer_config):
    for split in ("train", "eval"):
        assert trainer_config["env"][split]["max_episode_steps"] == 600


# --------------------------------------------------------------------------- #
# Observation mapping
# --------------------------------------------------------------------------- #
def test_bridge_concatenates_state_in_declared_order(policy_obs):
    states = _bridge(policy_obs)["states"]
    assert states.shape == (2, STATE_DIM)
    expected = torch.cat([policy_obs[key] for key in STATE_KEYS], dim=-1)
    assert torch.equal(states, expected)


def test_bridge_drops_the_alpha_channel(policy_obs):
    """GR00T wants RGB; the fluoroscopy sensor hands over RGBA."""
    assert _bridge(policy_obs)["main_images"].shape == (2, 8, 8, 3)


def test_bridge_offers_no_second_view(policy_obs):
    """A C-arm has one view; the key stays for schema parity with other workflows."""
    assert _bridge(policy_obs)["extra_view_images"] is None


def test_bridge_rejects_a_missing_term(policy_obs):
    del policy_obs["route_state"]
    with pytest.raises(KeyError, match="route_state"):
        _bridge(policy_obs)


def test_bridge_rejects_a_wrong_state_width(policy_obs):
    """Catch a changed observation term here, not as bad training later."""
    policy_obs["drive_state"] = torch.zeros(2, 5)
    with pytest.raises(ValueError, match="drive_state width 4"):
        _bridge(policy_obs)


def test_gr00t_groups_carry_the_right_slices(policy_obs):
    groups = convert_workflow_obs_to_gr00t(_bridge(policy_obs))
    # Each source term was filled with its own constant, so a mislabeled
    # slice shows up as the wrong value rather than the wrong shape.
    assert np.allclose(groups["state.catheter"], 1.0)
    assert np.allclose(groups["state.tip_pose"][:, :, :3], 2.0)
    assert np.allclose(groups["state.tip_pose"][:, :, 3:], 3.0)
    assert np.allclose(groups["state.navigation"][:, :, :3], 4.0)
    assert np.allclose(groups["state.navigation"][:, :, 3:], 5.0)


def test_gr00t_video_key_gains_the_time_axis(policy_obs):
    assert convert_workflow_obs_to_gr00t(_bridge(policy_obs))["video.fluoroscopy_view"].shape == (2, 1, 8, 8, 3)


def test_gr00t_carries_the_task_description(policy_obs):
    groups = convert_workflow_obs_to_gr00t(_bridge(policy_obs))
    assert groups["annotation.human.task_description"] == ["navigate", "navigate"]


def test_gr00t_conversion_rejects_a_wrong_state_width(policy_obs):
    bridge = _bridge(policy_obs)
    bridge["states"] = torch.zeros(2, STATE_DIM + 1)
    with pytest.raises(ValueError, match=f"state width {STATE_DIM}"):
        convert_workflow_obs_to_gr00t(bridge)


# --------------------------------------------------------------------------- #
# Action mapping
# --------------------------------------------------------------------------- #
def _chunk(catheter: float = 0.0, carm: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "action.catheter": np.full((2, 1, 3), catheter),
        "action.carm": np.full((2, 1, 1), carm),
    }


def test_action_is_the_identity_not_a_padded_slice():
    action = convert_gr00t_to_workflow_action(_chunk(catheter=0.5, carm=-0.25))
    assert action.shape == (2, 1, ACTION_DIM)
    assert np.allclose(action[:, :, :3], 0.5)
    # The C-arm lands last; padding here would zero a real control.
    assert np.allclose(action[:, :, 3], -0.25)


def test_action_rejects_a_missing_head():
    with pytest.raises(KeyError, match="action.carm"):
        convert_gr00t_to_workflow_action({"action.catheter": np.zeros((2, 1, 3))})


def test_action_rejects_a_wrong_head_width():
    chunk = _chunk()
    chunk["action.catheter"] = np.zeros((2, 1, 7))
    with pytest.raises(ValueError, match=f"{ACTION_DIM} catheter"):
        convert_gr00t_to_workflow_action(chunk)


def test_action_honours_the_chunk_size():
    chunk = {"action.catheter": np.zeros((2, 4, 3)), "action.carm": np.zeros((2, 4, 1))}
    assert convert_gr00t_to_workflow_action(chunk, chunk_size=2).shape == (2, 2, ACTION_DIM)


def test_action_keys_cover_the_action_space():
    assert ACTION_KEYS == ("action.catheter", "action.carm")
