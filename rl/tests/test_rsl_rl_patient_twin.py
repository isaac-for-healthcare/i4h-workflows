# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The RSL-RL launcher must deliver anatomy to every scene it constructs."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from i4h_rl import cli, rsl_rl_eval, rsl_rl_interop
from i4h_rl.backends import rsl_rl
from i4h_rl.profile import load_profile

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def navigation_profile(tmp_path, monkeypatch):
    # The maintained navigation profile selects RLinf. Exercise the RSL-RL
    # backend with the same scene and its existing numeric-observation config.
    source = ROOT / "rl/config/endoluminal_navigation_ppo_rsl_rl.yaml"
    config = yaml.safe_load(source.read_text())
    config["runner"]["experiment_name"] = "endoluminal_navigation"
    trainer_config = tmp_path / "navigation_ppo.yaml"
    trainer_config.write_text(yaml.safe_dump(config))
    profile = load_profile("endoluminal_navigation")
    profile = replace(
        profile,
        trainer="rsl_rl",
        algorithm="ppo",
        trainer_config=trainer_config,
        train_task_id=profile.scene,
        eval_task_id=profile.scene,
        cameras=(),
        default_epochs=config["runner"]["max_iterations"],
        resources=None,
        model_runtime=None,
        adapter_module=None,
        simulation=replace(profile.simulation, enable_cameras=False),
    )
    monkeypatch.setattr(cli, "load_profile", lambda _name: profile)
    monkeypatch.setenv("I4H_WORKFLOWS", str(ROOT))
    return profile


@pytest.fixture
def isaac_cli(monkeypatch):
    """Stub Kit imports while using the real registration/evaluation parsers."""

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)

    def add_lab_args(parser):
        parser.add_argument("--num_envs", type=int, default=1)
        parser.add_argument("--env_spacing", type=float, default=2.0)
        parser.add_argument("--presets", default="physx")

    module("isaaclab")
    module("isaaclab.app", AppLauncher=SimpleNamespace(add_app_launcher_args=lambda _parser: None))
    module("isaaclab.utils")
    module("isaaclab.utils.configclass", configclass=lambda cls: cls)
    module(
        "isaaclab_arena.cli.isaaclab_arena_cli",
        add_isaac_lab_cli_args=add_lab_args,
        add_isaaclab_arena_cli_args=lambda _parser: None,
    )
    monkeypatch.setattr(rsl_rl_interop, "_simulation_is_running", lambda: True)
    return module


@pytest.mark.parametrize("operation", ["train", "eval", "export"])
def test_patient_twin_reaches_rsl_rl_scene_and_artifacts(
    navigation_profile, isaac_cli, tmp_path, monkeypatch, operation
):
    twin = tmp_path / "patient with spaces" / "patient_twin.yaml"
    twin.parent.mkdir()
    twin.write_text("schema_version: 1\n")
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    output = tmp_path / operation
    launched = []

    monkeypatch.setattr(rsl_rl, "_runtime_python", lambda *_args: Path(sys.executable))
    monkeypatch.setattr(rsl_rl, "_preflight", lambda *_args: None)

    def run_child(command, *, env, cwd):
        launched.append(command)
        assert cwd == ROOT
        assert env["I4H_RL_TRAINER_CONFIG"] == str(navigation_profile.trainer_config)
        assert command[command.index("--patient-twin") + 1] == str(twin.resolve())
        if operation == "train":
            scenes = []

            def load_scene(task, args):
                assert task == navigation_profile.scene
                assert args.patient_twin == str(twin.resolve())
                assert args.fluoro_backend is None and args.fluoro_device is None
                scenes.append(task)
                return SimpleNamespace(configure_args=lambda _args: None, build=lambda: object())

            isaac_cli("i4h_arena.scenes.base", load_scene=load_scene)
            isaac_cli(
                "isaaclab_arena.environments.arena_env_builder",
                ArenaEnvBuilder=lambda *_args: SimpleNamespace(build_registered=lambda: None),
            )
            monkeypatch.setattr(sys, "argv", [command[1], *command[2:]])
            remaining = rsl_rl_interop.environment_registration_callback()
            assert "--patient-twin" not in remaining
            assert scenes == [navigation_profile.scene]
            bundle = output / "trainer"
            (bundle / "params").mkdir(parents=True)
            (bundle / "model_1.pt").write_bytes(b"checkpoint")
            (bundle / "params/agent.yaml").write_text("policy: {}\n")
        else:
            args = rsl_rl_eval._parser().parse_args(command[3:])
            assert args.patient_twin == str(twin.resolve())
            assert args.fluoro_backend is None and args.fluoro_device is None
            Path(args.output).write_text('{"episodes": 1}\n')
            if args.export_policy:
                Path(args.export_policy).write_bytes(b"policy")
        return 0

    monkeypatch.setattr(rsl_rl.subprocess, "call", run_child)
    argv = [navigation_profile.workflow, "--patient-twin", str(twin.relative_to(tmp_path))]
    if operation == "export":
        argv = ["export", *argv, "--checkpoint", str(checkpoint), "--output-dir", str(output)]
    else:
        argv += ["--run-dir", str(output), "--num-envs", "1", "--epochs", "1"]
        if operation == "eval":
            argv += ["--eval", "--checkpoint", str(checkpoint), "--episodes", "1"]

    assert cli.main(argv) == 0
    assert len(launched) == 1
    metadata = json.loads((output / ("policy.json" if operation == "export" else "run.json")).read_text())
    assert metadata["patient_twin"] == str(twin.resolve())


@pytest.mark.parametrize("operation", ["train", "eval", "export"])
@pytest.mark.parametrize("twin_arg", [None, "missing.yaml"])
def test_navigation_requires_an_existing_twin_before_launch(
    navigation_profile, tmp_path, monkeypatch, operation, twin_arg
):
    monkeypatch.setattr(rsl_rl, "_preflight", lambda *_args: pytest.fail("must reject the twin before runtime startup"))
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    output = tmp_path / "output"
    argv = [navigation_profile.workflow]
    if twin_arg:
        argv += ["--patient-twin", str(tmp_path / twin_arg)]
    if operation == "eval":
        argv += ["--eval", "--checkpoint", str(checkpoint)]
    elif operation == "export":
        argv = ["export", *argv, "--checkpoint", str(checkpoint), "--output-dir", str(output)]
    message = "requires --patient-twin" if twin_arg is None else "manifest does not exist"
    with pytest.raises(SystemExit, match=message):
        cli.main(argv)
    assert not output.exists()


def test_non_patient_rsl_export_still_needs_no_twin(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setenv("I4H_WORKFLOWS", str(ROOT))
    received = []
    monkeypatch.setattr(rsl_rl, "export", lambda args, *_rest: received.append(args) or 0)
    assert (
        cli.main(
            [
                "export",
                "ultrasound_probe_reach",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "exported"),
            ]
        )
        == 0
    )
    assert received[0].resolved_patient_twin is None


def test_rlinf_export_still_rejects_an_unused_twin(tmp_path):
    with pytest.raises(SystemExit, match="export does not accept --patient-twin"):
        cli.main(
            [
                "export",
                "endoluminal_navigation",
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--output-dir",
                str(tmp_path / "exported"),
                "--patient-twin",
                str(tmp_path / "patient.yaml"),
            ]
        )


def test_evaluation_defaults_allow_scenes_without_anatomy(isaac_cli):
    args = rsl_rl_eval._parser().parse_args(
        ["--task", "ultrasound_probe_reach", "--checkpoint", "model.pt", "--output", "evaluation.json"]
    )
    assert args.patient_twin is None
    assert args.fluoro_backend is None and args.fluoro_device is None
