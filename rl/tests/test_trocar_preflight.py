# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject the documented Trocar failures before starting expensive workers."""

import argparse
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from i4h_rl.backends import rlinf
from i4h_rl.profile import RLProfile

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def profile():
    return RLProfile.load(ROOT / "rl/profiles/assemble_trocar.yaml")


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / "checkpoint/experiment_cfg"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({"new_embodiment": {"modalities": {"state": {}}}}))
    return directory.parent


def launch_args(checkpoint, **changes):
    values = dict(
        only_eval=False, rl_model_path=None, resume_dir=None, model_path=str(checkpoint), num_envs=None, overrides=[]
    )
    values.update(changes)
    return argparse.Namespace(**values)


def test_base_checkpoint_rejected_before_gpu_assignment(profile, checkpoint, monkeypatch):
    (checkpoint / "experiment_cfg/metadata.json").write_text(json.dumps({"gr1": {"modalities": {}}}))
    monkeypatch.setattr(rlinf, "_gpu_assignment", lambda _: pytest.fail("GPU assignment ran before metadata rejection"))
    with pytest.raises(SystemExit, match="no metadata for embodiment 'new_embodiment'"):
        rlinf.validate_launch(launch_args(checkpoint), profile, ROOT)


@pytest.mark.parametrize("contents", [None, "not json", "[]", '{"new_embodiment": null}'])
def test_invalid_checkpoint_metadata(profile, checkpoint, contents):
    path = checkpoint / "experiment_cfg/metadata.json"
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    with pytest.raises(SystemExit, match="metadata"):
        rlinf.validate_launch(launch_args(checkpoint), profile, ROOT)


def test_default_64_env_config_accepts_compatible_metadata(profile, checkpoint, monkeypatch):
    monkeypatch.delenv("I4H_RL_MODEL_GPU", raising=False)
    monkeypatch.delenv("I4H_RL_SIM_GPU", raising=False)
    args = launch_args(checkpoint)
    rlinf.validate_launch(args, profile, ROOT)
    assert args.resolved_model_path == checkpoint


@pytest.mark.parametrize("prefix", ["", "++"])
def test_original_2048_batch_fails_in_preflight(profile, checkpoint, prefix):
    with pytest.raises(SystemExit, match="128000 rollout samples.*2048"):
        rlinf.validate_launch(
            launch_args(checkpoint, overrides=[f"{prefix}actor.global_batch_size=2048"]), profile, ROOT
        )


def test_batch_override_and_environment_override_are_respected(profile, checkpoint):
    args = launch_args(checkpoint, num_envs=128, overrides=["actor.global_batch_size=2048"])
    rlinf.validate_launch(args, profile, ROOT)
    rlinf.validate_launch(launch_args(checkpoint, overrides=["actor.global_batch_size=1024"]), profile, ROOT)


@pytest.mark.parametrize(
    "override", ["actor.global_batch_size=0", "actor.global_batch_size=true", "actor.model.num_action_chunks=3"]
)
def test_bad_batch_controls_rejected(profile, override):
    cfg = OmegaConf.merge(OmegaConf.create(rlinf._config(profile)), OmegaConf.from_dotlist([override]))
    with pytest.raises(SystemExit):
        rlinf.validate_training_batch(cfg, 64)


def test_timeout_default_and_override(profile, monkeypatch):
    monkeypatch.delenv("RLINF_TIMEOUT", raising=False)
    assert rlinf._runtime_env(ROOT, profile)["RLINF_TIMEOUT"] == "720"
    monkeypatch.setenv("RLINF_TIMEOUT", "1440")
    assert rlinf._runtime_env(ROOT, profile)["RLINF_TIMEOUT"] == "1440"
    monkeypatch.setenv("RLINF_TIMEOUT", "0")
    with pytest.raises(SystemExit, match="positive integer"):
        rlinf._runtime_env(ROOT, profile)


def test_n17_keeps_its_existing_checkpoint_contract(tmp_path, monkeypatch):
    profile = RLProfile.load(ROOT / "rl/profiles/endoluminal_navigation.yaml")
    monkeypatch.delenv("I4H_RL_MODEL_GPU", raising=False)
    monkeypatch.delenv("I4H_RL_SIM_GPU", raising=False)
    monkeypatch.delenv("RLINF_TIMEOUT", raising=False)
    # N1.7's adapter validates dataset_statistics.json; N1.5 metadata.json
    # must not become a new requirement for its dry-run discovery.
    rlinf.validate_launch(launch_args(tmp_path), profile, ROOT)
    assert "RLINF_TIMEOUT" not in rlinf._runtime_env(ROOT, profile)


@pytest.mark.parametrize(
    "override", ["+actor.global_batch_size=2048", "actor.global_batch_size", "actor.global_batch_size=["]
)
def test_invalid_hydra_override_is_concise(profile, checkpoint, override):
    with pytest.raises(SystemExit, match="cannot compose RLinf config"):
        rlinf.validate_launch(launch_args(checkpoint, overrides=[override]), profile, ROOT)


def test_hydra_additions_and_repeated_composition(profile):
    from i4h_rl.composition import compose_config

    for _ in range(3):
        cfg = compose_config(profile.trainer_config, ["+actor.review_value=42", "++actor.global_batch_size=1024"])
        assert cfg.actor.review_value == 42
        assert cfg.actor.global_batch_size == 1024
        assert compose_config(profile.trainer_config).actor.global_batch_size == 2000
    assert rlinf._episode_steps(profile, only_eval=False, overrides=("++env.train.max_episode_steps=42",)) == 42


@pytest.mark.parametrize("only_eval", [False, True])
@pytest.mark.parametrize("flag, configured", [(False, False), (True, False), (False, True)])
def test_video_policy(profile, only_eval, flag, configured):
    from i4h_rl.composition import compose_config, configure_video

    cfg = compose_config(profile.trainer_config, [f"env.eval.video_cfg.save_video={str(configured).lower()}"])
    required = configure_video(cfg, only_eval=only_eval, video=flag)
    assert required == (only_eval and (flag or configured))
    assert cfg.env.eval.video_cfg.raise_on_error == required
    assert cfg.env.train.video_cfg.raise_on_error is False


def test_launcher_composition_errors_precede_heavy_imports(profile, checkpoint, tmp_path):
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "i4h_rl.launcher",
            "--config",
            str(profile.trainer_config),
            "--model-path",
            str(checkpoint),
            "--run-dir",
            str(tmp_path / "run"),
            "--num-envs",
            "64",
            "--max-epochs",
            "1",
            "--set",
            "actor.global_batch_size=[",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "rl")},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "cannot compose RLinf config" in result.stderr
    assert "Traceback" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
