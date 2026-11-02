# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the checkpoint contract at the preflight and worker-import boundaries."""

import argparse
import builtins
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from i4h_rl import launcher
from i4h_rl.adapters.endoluminal_navigation import (
    _assert_checkpoint_matches,
    _checkpoint_contract,
    _resolve_checkpoint_path,
)
from i4h_rl.backends import rlinf
from i4h_rl.composition import compose_config
from i4h_rl.profile import RLProfile

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["endoluminal_navigation", "endoluminal_navigation_arm"])
def profile(request):
    return RLProfile.load(ROOT / "rl/profiles" / f"{request.param}.yaml")


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / "IL checkpoint with spaces/experiment_cfg"
    directory.mkdir(parents=True)
    groups = {"state": {"catheter": 3, "carm": 1, "target": 3, "route": 2}, "action": {"catheter": 3, "carm": 1}}
    statistics = {
        "new_embodiment": {
            modality: {group: {"min": [0] * width} for group, width in widths.items()}
            for modality, widths in groups.items()
        }
    }
    (directory / "dataset_statistics.json").write_text(json.dumps(statistics))
    return directory.parent


def check_contract(env):
    # The same YAML reader and contract checks used by adapter registration,
    # without importing Isaac Sim or constructing a GR00T model on CPU.
    cfg = yaml.safe_load(Path(env["RLINF_CONFIG_FILE"]).read_text())
    model = _resolve_checkpoint_path(cfg)
    tag = cfg["env"]["train"]["isaaclab"]["embodiment_tag"]
    _assert_checkpoint_matches(_checkpoint_contract(model, tag), source=model)
    return cfg


@pytest.fixture
def runtime(monkeypatch):
    seen = {}

    def preflight(_runtime, env, **_kwargs):
        seen["preflight"] = check_contract(env)
        seen["config_path"] = env["RLINF_CONFIG_FILE"]

    monkeypatch.setattr(rlinf, "_model_runtime_python", lambda *_: Path(sys.executable))
    monkeypatch.setattr(rlinf, "_sim_runtime_python", lambda *_: Path(sys.executable))
    monkeypatch.setattr(rlinf, "_model_preflight", preflight)
    monkeypatch.setattr(rlinf, "_sim_preflight", lambda *_: None)
    monkeypatch.delenv("I4H_RL_MODEL_GPU", raising=False)
    monkeypatch.delenv("I4H_RL_SIM_GPU", raising=False)
    return seen


def launch_args(checkpoint, tmp_path, **changes):
    values = dict(
        model_path=str(checkpoint),
        only_eval=False,
        rl_model_path=None,
        resume_dir=None,
        num_envs=1,
        overrides=[],
        runtime_python=None,
        sim_runtime_python=None,
        run_dir=str(tmp_path / "run"),
        video=False,
    )
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("mode", ["train", "resume", "eval"])
def test_preflight_and_launch_use_the_same_checkpoint(profile, checkpoint, runtime, tmp_path, monkeypatch, mode):
    source = profile.trainer_config.read_bytes()
    native = tmp_path / "previous/actor/model_state_dict/full_weights.pt"
    native.parent.mkdir(parents=True)
    native.touch()
    args = launch_args(
        checkpoint,
        tmp_path,
        only_eval=mode == "eval",
        rl_model_path=str(tmp_path / "previous") if mode == "eval" else None,
        resume_dir=str(tmp_path / "previous") if mode == "resume" else None,
        overrides=[
            "+actor.test_addition=42",
            "actor.model.model_path=/wrong-actor",
            "rollout.model.model_path=/wrong-rollout",
        ],
    )
    rlinf.validate_launch(args, profile, ROOT)

    class Simulator:
        def __init__(self, command, *, env, cwd):
            assert check_contract(env) == runtime["preflight"]
            Path(command[command.index("--ready-file") + 1]).touch()

        def poll(self):
            return 0

    def run_model(command, *, env):
        parsed = launcher._parser().parse_args(command[3:])
        cfg = compose_config(parsed.config, parsed.overrides)
        assert str(parsed.config) == env["RLINF_CONFIG_FILE"] == runtime["config_path"]
        assert cfg.actor.model.model_path == cfg.rollout.model.model_path == str(parsed.model_path) == str(checkpoint)
        assert cfg.actor.test_addition == 42
        assert cfg.runner.max_epochs == 2
        assert cfg.env.train.total_num_envs == cfg.env.eval.total_num_envs == 1
        assert cfg.runner.only_eval == (mode == "eval")
        if mode == "resume":
            assert cfg.runner.resume_dir == str(tmp_path / "previous")
        if mode == "eval":
            assert cfg.rollout.model.rl_model_path == str(tmp_path / "previous")
            assert cfg.rollout.model.action_dim == cfg.actor.model.action_dim == 4
        else:
            weights = tmp_path / "run/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
            weights.parent.mkdir(parents=True)
            weights.touch()
        runtime["launched"] = True
        return 0

    monkeypatch.setattr(rlinf.subprocess, "Popen", Simulator)
    monkeypatch.setattr(rlinf.subprocess, "call", run_model)
    monkeypatch.setattr(rlinf, "finalize_evaluation", lambda *_args, **_kwargs: True)
    assert rlinf.launch(args, profile, ROOT, num_envs=1, epochs=2, overrides=tuple(args.overrides)) == 0
    assert runtime["launched"]
    assert profile.trainer_config.read_bytes() == source


@pytest.mark.parametrize("invalid", ["missing", "wrong_groups"])
def test_bad_cli_checkpoint_is_rejected_before_any_worker(profile, checkpoint, runtime, tmp_path, monkeypatch, invalid):
    statistics = checkpoint / "experiment_cfg/dataset_statistics.json"
    if invalid == "missing":
        statistics.unlink()
        error = FileNotFoundError
    else:
        statistics.write_text(json.dumps({"new_embodiment": {"state": {}, "action": {}}}))
        error = ValueError
    args = launch_args(checkpoint, tmp_path)
    rlinf.validate_launch(args, profile, ROOT)
    monkeypatch.setattr(
        rlinf.subprocess, "Popen", lambda *_a, **_k: pytest.fail("worker started before contract check")
    )
    with pytest.raises(error) as caught:
        rlinf.launch(args, profile, ROOT, num_envs=1, epochs=2, overrides=())
    assert str(checkpoint) in str(caught.value)
    assert "/models/replace-with-local-sft-checkpoint" not in str(caught.value)


@pytest.mark.parametrize("only_eval", [False, True])
def test_direct_launcher_publishes_cli_checkpoint_before_rlinf_import(
    profile, checkpoint, tmp_path, monkeypatch, only_eval
):
    class ReachedRLinfImport(Exception):
        pass

    original_import = builtins.__import__

    def inspect_import(name, *args, **kwargs):
        if name == "rlinf.config":
            cfg = check_contract(os.environ)
            assert cfg["actor"]["model"]["model_path"] == cfg["rollout"]["model"]["model_path"] == str(checkpoint)
            assert cfg["actor"]["test_addition"] == 42
            assert cfg["runner"]["only_eval"] == only_eval
            raise ReachedRLinfImport
        return original_import(name, *args, **kwargs)

    for name in ("RLINF_CONFIG_FILE", "RLINF_EXT_MODULE", "I4H_RL_RUN_DIR"):
        monkeypatch.setenv(name, "stale-value")
    monkeypatch.setattr(builtins, "__import__", inspect_import)
    command = [
        "--config",
        str(profile.trainer_config),
        "--model-path",
        str(checkpoint),
        "--run-dir",
        str(tmp_path / "direct-run"),
        "--num-envs",
        "1",
        "--max-epochs",
        "2",
        "--set",
        "+actor.test_addition=42",
        "--set",
        "actor.model.model_path=/wrong-checkpoint",
    ]
    if only_eval:
        command.append("--only-eval")
    with pytest.raises(ReachedRLinfImport):
        launcher.main(command)


@pytest.mark.parametrize("explicit_config", [False, True])
def test_export_preflight_uses_the_conversion_config(
    profile, checkpoint, runtime, tmp_path, monkeypatch, explicit_config
):
    native = tmp_path / "previous/actor/model_state_dict/full_weights.pt"
    native.parent.mkdir(parents=True)
    native.touch()
    config = tmp_path / ("explicit.yaml" if explicit_config else "previous/config.yaml")
    cfg = yaml.safe_load(profile.trainer_config.read_text())
    cfg["actor"]["model"]["model_path"] = cfg["rollout"]["model"]["model_path"] = str(checkpoint)
    config.write_text(yaml.safe_dump(cfg))
    args = argparse.Namespace(
        rl_model_path=str(tmp_path / "previous"),
        train_config=str(config) if explicit_config else None,
        runtime_python=None,
        output_dir=str(tmp_path / "export"),
    )

    def convert(command, *, env):
        assert env["RLINF_CONFIG_FILE"] == runtime["config_path"] == str(config)
        assert f"+convertor.train_config_path={config}" in command
        for filename in ("config.json", "model.safetensors.index.json"):
            (tmp_path / "export" / filename).write_text("{}")
        return 0

    monkeypatch.setattr(rlinf.subprocess, "call", convert)
    assert rlinf.export(args, profile, ROOT) == 0
