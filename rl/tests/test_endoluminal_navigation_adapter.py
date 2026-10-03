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

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from i4h_rl.adapters.endoluminal_navigation import (
    ACTION_DIM,
    ACTION_KEYS,
    ARM_EVAL_TASK_ID,
    ARM_TRAIN_TASK_ID,
    EVAL_TASK_ID,
    GR00T_LANGUAGE_KEY,
    GR00T_STATE_DIM,
    GR00T_STATE_GROUPS,
    GR00T_VIDEO_KEY,
    OBS_CONVERTER,
    STATE_DIM,
    STATE_KEYS,
    STATE_WIDTHS,
    TASK_IDS,
    TRAIN_TASK_ID,
    _assert_contract_matches,
    convert_gr00t_to_workflow_action,
    convert_workflow_obs_to_gr00t,
    wrap_workflow_observation,
)
from i4h_rl.profile import RLProfile

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "rl/profiles/endoluminal_navigation.yaml"
CONFIG_PATH = REPO / "rl/config/endoluminal_navigation_ppo_gr00t.yaml"
ARM_PROFILE_PATH = REPO / "rl/profiles/endoluminal_navigation_arm.yaml"
ARM_CONFIG_PATH = REPO / "rl/config/endoluminal_navigation_arm_ppo_gr00t.yaml"


@pytest.fixture(scope="module")
def trainer_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arm_trainer_config() -> dict:
    return yaml.safe_load(ARM_CONFIG_PATH.read_text(encoding="utf-8"))


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
    expected = {key: [start, stop] for key, start, stop in GR00T_STATE_GROUPS}
    declared = {
        entry["gr00t_key"]: entry["slice"]
        for entry in trainer_config["env"]["train"]["isaaclab"]["gr00t_mapping"]["state"]
    }
    assert declared == expected


def test_trainer_config_video_key_matches_the_adapter(trainer_config):
    """A renamed camera key is not refused by GR00T; it arrives as no image."""
    video = trainer_config["env"]["train"]["isaaclab"]["gr00t_mapping"]["video"]
    assert video["main_images"] == GR00T_VIDEO_KEY


def test_trainer_config_slices_select_drive_target_and_route(trainer_config):
    """The policy receives nine scalars while tip pose remains privileged."""
    slices = [
        entry["slice"]
        for entry in trainer_config["env"]["train"]["isaaclab"]["gr00t_mapping"]["state"]
    ]
    assert slices == [[0, 3], [3, 4], [10, 13], [13, 15]]


def test_the_policy_reads_drive_target_and_route_but_not_tip_pose(trainer_config):
    assert GR00T_STATE_DIM == STATE_WIDTHS[0] + STATE_WIDTHS[3] + STATE_WIDTHS[4]
    assert GR00T_STATE_DIM < STATE_DIM


def test_the_state_groups_have_the_new_checkpoint_widths():
    assert [stop - start for _key, start, stop in GR00T_STATE_GROUPS] == [3, 1, 3, 2]


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
    # Every source term was filled with its own constant, so a mislabeled
    # slice shows up as the wrong value rather than the wrong shape. All four
    # drive values come from term 1, while goal and route come from terms 4/5.
    assert groups["state.catheter"].shape == (2, 1, 3)
    assert groups["state.carm"].shape == (2, 1, 1)
    assert groups["state.target"].shape == (2, 1, 3)
    assert groups["state.route"].shape == (2, 1, 2)
    assert np.allclose(groups["state.catheter"], 1.0)
    assert np.allclose(groups["state.carm"], 1.0)
    assert np.allclose(groups["state.target"], 4.0)
    assert np.allclose(groups["state.route"], 5.0)


def test_gr00t_is_handed_exactly_the_new_checkpoint_groups(policy_obs):
    groups = convert_workflow_obs_to_gr00t(_bridge(policy_obs))
    assert set(groups) == {
        GR00T_VIDEO_KEY,
        GR00T_LANGUAGE_KEY,
        "state.catheter",
        "state.carm",
        "state.target",
        "state.route",
    }


def test_tip_pose_stays_out_but_navigation_goal_reaches_the_policy(policy_obs):
    groups = convert_workflow_obs_to_gr00t(_bridge(policy_obs))
    states = np.concatenate(
        [groups["state.catheter"], groups["state.carm"], groups["state.target"], groups["state.route"]],
        axis=-1,
    )
    assert states.shape[-1] == GR00T_STATE_DIM
    assert np.any(states == 4.0)
    assert np.any(states == 5.0)
    assert not np.any(states == 2.0)
    assert not np.any(states == 3.0)


def test_gr00t_video_key_gains_the_time_axis(policy_obs):
    assert convert_workflow_obs_to_gr00t(_bridge(policy_obs))[GR00T_VIDEO_KEY].shape == (2, 1, 8, 8, 3)


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
def _chunk(catheter: float = 0.0, carm: float = 0.0, prefix: str = "") -> dict[str, np.ndarray]:
    return {
        f"{prefix}catheter": np.full((2, 1, 3), catheter),
        f"{prefix}carm": np.full((2, 1, 1), carm),
    }


def test_action_is_the_identity_not_a_padded_slice():
    action = convert_gr00t_to_workflow_action(_chunk(catheter=0.5, carm=-0.25))
    assert action.shape == (2, 1, ACTION_DIM)
    assert np.allclose(action[:, :, :3], 0.5)
    # The C-arm lands last; padding here would zero a real control.
    assert np.allclose(action[:, :, 3], -0.25)


def test_action_accepts_the_n15_prefixed_spelling():
    """N1.5 emitted ``action.``-prefixed group names and N1.7 emits them bare.

    Both resolve so that the generation the checkpoint was trained with does
    not have to be mirrored here; upstream's own N1.7 Libero converter keeps
    the same tolerance for the same reason.
    """
    action = convert_gr00t_to_workflow_action(_chunk(catheter=0.5, carm=-0.25, prefix="action."))
    assert np.allclose(action[:, :, :3], 0.5)
    assert np.allclose(action[:, :, 3], -0.25)


def test_action_rejects_a_missing_head():
    with pytest.raises(KeyError, match="carm"):
        convert_gr00t_to_workflow_action({"catheter": np.zeros((2, 1, 3))})


def test_action_rejects_a_wrong_head_width():
    chunk = _chunk()
    chunk["catheter"] = np.zeros((2, 1, 7))
    with pytest.raises(ValueError, match=f"{ACTION_DIM} catheter"):
        convert_gr00t_to_workflow_action(chunk)


def test_action_honours_the_chunk_size():
    chunk = {"catheter": np.zeros((2, 4, 3)), "carm": np.zeros((2, 4, 1))}
    assert convert_gr00t_to_workflow_action(chunk, chunk_size=2).shape == (2, 2, ACTION_DIM)


def test_action_keys_are_the_registered_modality_groups():
    """These are the group names ``config_catheter.py`` registers with GR00T."""
    assert ACTION_KEYS == ("catheter", "carm")


# --------------------------------------------------------------------------- #
# The startup guard against silent drift
# --------------------------------------------------------------------------- #
class _Group:
    """Stands in for GR00T's ``ModalityConfig``, which needs the heavy venv."""

    def __init__(self, *keys: str) -> None:
        self.modality_keys = list(keys)


def _registered(**overrides) -> dict[str, _Group]:
    """What ``config_catheter.CATHETER_CONFIG`` declares."""
    config = {
        "video": _Group("fluoroscopy"),
        "state": _Group("catheter", "carm", "target", "route"),
        "action": _Group("catheter", "carm"),
        "language": _Group("annotation.human.task_description"),
    }
    config.update(overrides)
    return config


def test_the_guard_passes_against_the_real_registered_groups():
    """The contract this module emits is the one the checkpoint was trained on."""
    _assert_contract_matches(_registered())


def test_the_guard_catches_a_renamed_camera():
    """The defect this guard exists for: `video.fluoroscopy_view` reached GR00T
    as an undeclared key, so the policy trained on no image at all."""
    with pytest.raises(ValueError, match="video groups"):
        _assert_contract_matches(_registered(video=_Group("fluoroscopy_view")))


def test_the_guard_catches_an_extra_state_group():
    """Groups the processor does not declare are dropped rather than refused."""
    with pytest.raises(ValueError, match="state groups"):
        _assert_contract_matches(_registered(state=_Group("catheter", "carm", "target", "route", "tip")))


def test_the_guard_catches_a_reordered_state_group():
    """Order is positional in GR00T, so swapping these swaps the projectors."""
    with pytest.raises(ValueError, match="state groups"):
        _assert_contract_matches(_registered(state=_Group("carm", "catheter", "target", "route")))


# --------------------------------------------------------------------------- #
# The GR00T generation is wired consistently
# --------------------------------------------------------------------------- #
def test_trainer_config_selects_the_n17_model(trainer_config):
    """The SFT checkpoint reports ``Gr00tN1d7``; RLinf picks by ``model_type``."""
    assert trainer_config["actor"]["model"]["model_type"] == "gr00t_n1d7"


def test_rollout_model_type_tracks_the_actor(trainer_config):
    """Left unset, RLinf's ``get_model`` defaults to N1.5 and loads the wrong class."""
    assert trainer_config["rollout"]["model"]["model_type"] == "${actor.model.model_type}"


def test_trainer_config_leaves_data_config_class_unset(trainer_config):
    """Setting it diverts loading to a path that hardcodes the N1.5 class."""
    assert "data_config_class" not in trainer_config["env"]["train"]["isaaclab"]


def test_profile_pins_the_n17_training_runtime():
    """A task venv pins one GR00T generation, so N1.7 cannot use the N1.5 venv."""
    profile = RLProfile.load(PROFILE_PATH)
    assert profile.model_runtime == "tasks/gr00t_n17/.venv/bin/python"
    assert (REPO / profile.model_runtime).is_file()


def test_profile_opts_into_shared_gb300_gpu():
    profile = RLProfile.load(PROFILE_PATH)
    assert profile.resources is not None
    assert profile.resources.model_gpu == "0"
    assert profile.resources.simulator_gpu == "0"
    assert profile.resources.allow_shared_gpu is True


# --------------------------------------------------------------------------- #
# The patient twin reaches the scene
# --------------------------------------------------------------------------- #
def test_profile_declares_the_twin_mandatory():
    """Without a twin the embodiment publishes no observation or reward config."""
    assert RLProfile.load(PROFILE_PATH).requires_patient_twin is True


def test_launching_without_a_twin_is_refused_before_isaac_starts():
    from i4h_rl import cli

    with pytest.raises(SystemExit, match="requires --patient-twin"):
        cli.main(["endoluminal_navigation", "--model-path", str(REPO), "--dry-run"])


def test_a_twin_is_refused_for_a_scene_that_has_no_use_for_one():
    """Accepting and ignoring it would make a run look patient-specific."""
    from i4h_rl import cli

    with pytest.raises(SystemExit, match="does not take --patient-twin"):
        cli.main(["assemble_trocar", "--model-path", str(REPO), "--patient-twin", "whatever.yaml", "--dry-run"])


def test_a_missing_twin_manifest_is_caught_at_the_cli():
    from i4h_rl import cli

    with pytest.raises(SystemExit, match="manifest does not exist"):
        cli.main(
            [
                "endoluminal_navigation",
                "--model-path",
                str(REPO),
                "--patient-twin",
                "data/no-such-twin/patient_twin.yaml",
                "--dry-run",
            ]
        )


def test_the_simulator_accepts_every_scene_argument_the_twin_needs():
    """sim_server is a separate process, so its parser is the real contract."""
    from i4h_rl.sim_server import _parser, _scene_args

    args = _parser().parse_args(
        [
            "--scene",
            "endoluminal_navigation",
            "--socket",
            "/tmp/s",
            "--ready-file",
            "/tmp/r",
            "--num-envs",
            "2",
            "--max-episode-steps",
            "600",
            "--env-spacing",
            "2.0",
            "--presets",
            "physx",
            "--enable-cameras",
            "--patient-twin",
            "data/twins/s0058/patient_twin.yaml",
        ]
    )
    scene_args = _scene_args(args)
    assert scene_args.patient_twin == "data/twins/s0058/patient_twin.yaml"
    # Left for the scene to resolve, which picks Slang when a twin is present.
    assert scene_args.fluoro_backend is None
    assert scene_args.fluoro_device == "vulkan"


def test_a_twinless_simulator_still_carries_the_scene_attributes():
    """A Scene reads these off the namespace; absent, it would raise instead."""
    from i4h_rl.sim_server import _parser, _scene_args

    args = _parser().parse_args(
        [
            "--scene",
            "assemble_trocar",
            "--socket",
            "/tmp/s",
            "--ready-file",
            "/tmp/r",
            "--num-envs",
            "2",
            "--max-episode-steps",
            "600",
            "--env-spacing",
            "2.0",
            "--presets",
            "physx",
        ]
    )
    scene_args = _scene_args(args)
    assert scene_args.patient_twin is None
    assert scene_args.fluoro_backend is None


def test_runtime_pythonpath_does_not_shadow_gr00t_17_with_15():
    """PYTHONPATH outranks the venv, so the wrong checkout here wins silently.

    The failure that follows is not an import error: GR00T 1.5 has no modality
    registration, and the action converters are registered per generation, so
    the mismatch surfaces far from its cause.
    """
    from i4h_rl.backends.rlinf import _runtime_env

    entries = _runtime_env(REPO, RLProfile.load(PROFILE_PATH))["PYTHONPATH"].split(os.pathsep)
    gr00t_sources = [entry for entry in entries if "Isaac-GR00T" in entry]
    assert [Path(entry).name for entry in gr00t_sources] == ["Isaac-GR00T-1.7"]
    assert any(entry.endswith("tasks/gr00t_n17") for entry in entries)


# --------------------------------------------------------------------------- #
# The arm-borne scene post-trains through the same adapter
# --------------------------------------------------------------------------- #
def test_the_arm_profile_names_the_arm_scene_and_this_adapter():
    profile = RLProfile.load(ARM_PROFILE_PATH)
    assert profile.workflow == "endoluminal_navigation_arm"
    assert profile.scene == "endoluminal_navigation_arm"
    assert profile.adapter_module == "i4h_rl.adapters.endoluminal_navigation"
    assert profile.trainer_config.resolve() == ARM_CONFIG_PATH.resolve()


def test_the_adapter_answers_for_both_scenes_task_ids():
    """One environment class serves every id; the Scene is chosen by the
    profile, through the simulator process, not by the id."""
    assert TASK_IDS == (TRAIN_TASK_ID, EVAL_TASK_ID, ARM_TRAIN_TASK_ID, ARM_EVAL_TASK_ID)
    assert len(set(TASK_IDS)) == len(TASK_IDS)


def test_the_arm_profile_task_ids_are_its_own():
    """Sharing ids would work but would log a run under the wrong scene."""
    profile = RLProfile.load(ARM_PROFILE_PATH)
    armless = RLProfile.load(PROFILE_PATH)
    assert profile.train_task_id == ARM_TRAIN_TASK_ID
    assert profile.eval_task_id == ARM_EVAL_TASK_ID
    assert profile.train_task_id != armless.train_task_id
    assert profile.eval_task_id != armless.eval_task_id


def test_the_arm_profile_agrees_with_its_trainer_config():
    """The backend cross-checks these and refuses the run if they disagree."""
    from i4h_rl.backends.rlinf import validate_profile

    validate_profile(RLProfile.load(ARM_PROFILE_PATH), REPO)


def test_the_arm_carries_the_same_widths_as_the_armless_scene():
    """The arm appends joints to recordings, not to the RL observation.

    ``drive_state`` is read off the action terms rather than the articulation,
    so it is four values whether or not something holds the drive unit, and
    ``franka_catheter.yaml`` declares the same four ``action_names``.
    """
    profile = RLProfile.load(ARM_PROFILE_PATH)
    armless = RLProfile.load(PROFILE_PATH)
    assert profile.state_dof == armless.state_dof == STATE_DIM
    assert profile.action_dof == armless.action_dof == ACTION_DIM
    assert profile.policy_action_dof == armless.policy_action_dof
    assert profile.cameras == armless.cameras
    assert profile.task_description == armless.task_description
    assert profile.requires_patient_twin == armless.requires_patient_twin is True
    assert profile.model_runtime == armless.model_runtime


def test_the_arm_runs_fewer_environments():
    """The coupled MJWarp + XPBD solver costs more per step than the rod-only
    one, and there is no CUDA graph capture to amortise it."""
    assert RLProfile.load(ARM_PROFILE_PATH).default_num_envs < RLProfile.load(PROFILE_PATH).default_num_envs


#: What the arm trainer config is allowed to differ on. Everything else is the
#: same optimiser acting on the same observation, so a difference would be
#: drift between two copies rather than a deliberate choice.
ARM_CONFIG_DIFFERENCES = {
    ("runner", "logger", "experiment_name"),
    ("env", "train", "init_params", "id"),
    ("env", "eval", "init_params", "id"),
    ("env", "train", "total_num_envs"),
    ("env", "eval", "total_num_envs"),
}


def _flatten(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _flatten(value, (*prefix, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _flatten(value, (*prefix, index))
    else:
        yield prefix, node


def test_the_arm_config_differs_only_where_it_has_to(trainer_config, arm_trainer_config):
    """The two configs are copies, so this is what keeps them from drifting.

    Composition via ``defaults:`` is not available: ``validate_profile`` reads
    these files with plain ``yaml.safe_load`` to cross-check them against the
    profile, and an inherited key would be absent from the raw mapping.
    """
    armless = dict(_flatten(trainer_config))
    arm = dict(_flatten(arm_trainer_config))
    assert set(armless) == set(arm), "the two configs must declare the same keys"
    differing = {key for key, value in armless.items() if arm[key] != value}
    assert differing == ARM_CONFIG_DIFFERENCES


def test_the_arm_config_task_ids_match_the_adapter(arm_trainer_config):
    init = arm_trainer_config["env"]
    assert init["train"]["init_params"]["id"] == ARM_TRAIN_TASK_ID
    assert init["eval"]["init_params"]["id"] == ARM_EVAL_TASK_ID


def test_both_catheter_profiles_are_discovered():
    """``available_profiles`` refuses duplicate workflow names, so this also
    pins that the two profiles are distinct rather than one shadowing the other."""
    from i4h_rl.profile import available_profiles

    assert {"endoluminal_navigation", "endoluminal_navigation_arm"} <= set(available_profiles())


def test_n15_profiles_keep_their_own_gr00t_source():
    """Adding the N1.7 runtime must not drag other profiles onto 1.7."""
    from i4h_rl.backends.rlinf import _runtime_env

    trocar = RLProfile.load(REPO / "rl/profiles/assemble_trocar.yaml")
    entries = _runtime_env(REPO, trocar)["PYTHONPATH"].split(os.pathsep)
    assert any(entry.endswith("Isaac-GR00T-1.5") for entry in entries)
    assert not any("Isaac-GR00T-1.7" in entry for entry in entries)
