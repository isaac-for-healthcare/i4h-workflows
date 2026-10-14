# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tactile stack is declared as data; these hold the declaration together.

Everything the server tells a client and everything the trainer passes to the
launcher is derived from one manifest, so a change there has to show up in
both. Runs in the light venv: ``gr00t`` is only reached inside ``load_policy``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from i4h_common.taskdef import load_taskdef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.server import ServeOptions, inference_settings, main, server_info  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.tactile.interface import FINGERTIPS  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.train import finetune_argv  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = (
    Path(__file__).resolve().parents[1] / "i4h_tasks" / "gr00t_n17_tactile" / "manifest" / "h2_sharpa_tactile.yaml"
)


@pytest.fixture(scope="module")
def spec():
    runtime, impl, definition = load_taskdef(MANIFEST)
    assert runtime == "remote" and impl is None
    return definition


# -- the declaration ------------------------------------------------------


def test_the_task_drives_an_embodiment_with_a_descriptor(spec) -> None:
    assert spec.embodiment == "h2_sharpa"
    assert (ROOT / "arena" / "i4h_arena" / "embodiments" / "manifest" / f"{spec.embodiment}.yaml").is_file()


def test_no_published_checkpoint_is_claimed(spec) -> None:
    """A rollout passes --checkpoint; a `repo` here would say one exists."""
    assert "repo" not in spec.model
    assert spec.trainable


def test_one_feature_set_per_fingertip(spec) -> None:
    assert spec.model["tactile"]["num_tokens"] == len(FINGERTIPS)


def test_the_state_groups_tile_the_whole_vector(spec) -> None:
    groups = sorted(spec.observation["state_groups"].values())
    assert groups[0][0] == 0
    assert all(end == next_start for (_, end), (next_start, _) in zip(groups, groups[1:], strict=False))
    assert groups[-1][1] == 58


# -- what the client is told ----------------------------------------------


def test_server_info_follows_the_manifest(spec) -> None:
    info = server_info(spec, ServeOptions())
    assert info["video_modality_keys"] == ["high", "left_wrist_view", "right_wrist_view"]
    assert info["tactile"]["fingertips"] == list(FINGERTIPS)
    assert info["tactile"]["force_window"] == 16
    assert info["tactile"]["delay_aug"] is True
    assert info["tactile"]["refresh"] == "per-step"
    assert info["cascade"] == {"enabled": False, "fast_loops": 2}


def test_disabling_tactile_is_visible_to_the_client(spec) -> None:
    assert server_info(spec, ServeOptions(disable_tactile=True))["tactile"]["enabled"] is False


def test_the_command_line_overrides_the_inference_block(spec) -> None:
    settings = inference_settings(spec, ServeOptions(cascade=True, cascade_fast_loops=3, tactile_refresh="per-chunk"))
    assert settings == {"tactile_refresh": "per-chunk", "cascade": True, "cascade_fast_loops": 3}


def test_an_unknown_refresh_mode_is_rejected(spec) -> None:
    with pytest.raises(ValueError, match="tactile_refresh"):
        inference_settings(spec, ServeOptions(tactile_refresh="never"))


def test_print_server_info_runs_without_the_model_stack(capsys) -> None:
    assert main(["--print-server-info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["task"] == "gr00t_n17_tactile/h2_sharpa_tactile"
    assert info["embodiment"] == "h2_sharpa"


# -- what the launcher is told --------------------------------------------


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_the_finetune_argv_carries_the_tactile_recipe(spec) -> None:
    argv = finetune_argv(spec, "/data/pads")
    assert {"--use_tactile", "--use_tactile_force", "--delay_aug", "--tune_visual"} <= set(argv)
    assert _value(argv, "--num_tactile_tokens") == "10"
    assert _value(argv, "--force_window") == "16"
    assert _value(argv, "--tau_split") == "0.5"
    assert _value(argv, "--dataset_path") == "/data/pads"
    assert _value(argv, "--modality_config_path").endswith("gr00t_n17_tactile/config.py")


def test_frozen_assets_are_named_by_their_public_source(spec) -> None:
    argv = finetune_argv(spec, "/data/pads")
    assert _value(argv, "--sharpa_encoder_ckpt_path") == (
        "hf://Sharpa-Robotics/sharpawave-deform-encoder/sharpawave_deform_autoencoder.safetensors"
    )
    assert _value(argv, "--trex_vqvae_ckpt_path").startswith("hf://miniFranka/T-Rex_")
    assert _value(argv, "--sharpa_left_urdf_path") == "third_party/sharpa-urdf-usd-xml/wave_01/left_sharpa_wave"


def test_the_vision_only_baseline_drops_every_tactile_flag(spec) -> None:
    argv = finetune_argv(spec, "/data/pads", {"use_tactile": False})
    flags = [arg for arg in argv if arg.startswith("--")]
    assert not [flag for flag in flags if any(word in flag for word in ("tactile", "sharpa", "trex", "delay"))]
    assert "--base_model_path" in flags


def test_overrides_win_over_the_manifest(spec) -> None:
    argv = finetune_argv(spec, "/data/pads", {"max_steps": 5, "base_model": "/ckpt/base", "tune_visual": False})
    assert _value(argv, "--max_steps") == "5"
    assert _value(argv, "--base_model_path") == "/ckpt/base"
    assert "--no-tune_visual" in argv
