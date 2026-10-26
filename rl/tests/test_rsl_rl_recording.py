# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disable the inherited episode recorder in the RSL-RL training and evaluation paths."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from i4h_rl import rsl_rl_eval, rsl_rl_interop


class _AppLauncher:
    @staticmethod
    def add_app_launcher_args(_parser):
        pass

    def __init__(self, _args):
        self.app = SimpleNamespace(close=lambda: None)


def _add_lab_args(parser):
    parser.add_argument("--num_envs", type=int, default=1)


def _stub_modules(monkeypatch, modules):
    base = {
        "isaaclab": {},
        "isaaclab.app": {"AppLauncher": _AppLauncher},
        "isaaclab.utils": {},
        "isaaclab.utils.configclass": {"configclass": lambda cls: cls},
        "isaaclab_arena.cli.isaaclab_arena_cli": {
            "add_isaac_lab_cli_args": _add_lab_args,
            "add_isaaclab_arena_cli_args": lambda _parser: None,
        },
    }
    for name, attributes in {**base, **modules}.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)


def test_rsl_rl_training_registers_cfg_without_episode_recorders(monkeypatch):
    registered = []

    class ArenaEnvBuilder:
        def __init__(self, _arena_env, _args):
            pass

        def compose_manager_cfg(self):
            return SimpleNamespace(recorders=object()), {}

        def build_registered(self, env_cfg=None, env_kwargs=None):
            # Like Arena 0.3.1: register the given cfg (or a fresh composition) and
            # return a parsed copy that Isaac Lab's train.py never sees.
            cfg = env_cfg if env_cfg is not None else self.compose_manager_cfg()[0]
            registered.append(cfg)
            return "test", SimpleNamespace(**vars(cfg)), env_kwargs or {}

    scene = SimpleNamespace(configure_args=lambda _args: None, arena_environment=lambda: object())
    _stub_modules(
        monkeypatch,
        {
            "isaaclab_arena.environments.arena_env_builder": {"ArenaEnvBuilder": ArenaEnvBuilder},
            "i4h_arena.scenes.base": {"load_scene": lambda *_args: scene, "arena_builder_cfg": lambda args: args},
        },
    )
    monkeypatch.setattr(rsl_rl_interop, "_simulation_is_running", lambda: True)
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--task", "ultrasound_probe_reach", "--rl_training_mode", "--num_envs", "4"]
    )

    rsl_rl_interop.environment_registration_callback()

    assert len(registered) == 1
    assert registered[0].recorders is None


def test_rsl_rl_evaluation_does_not_inherit_episode_recorders(tmp_path, monkeypatch):
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1), recorders=object())
    scene = SimpleNamespace(configure_args=lambda _args: None, gym_spec=lambda: ("test", cfg))
    observed = []

    class Exited(Exception):
        pass

    def make(_id, **kwargs):
        observed.append(kwargs["cfg"])
        raise RuntimeError("stop at gym construction boundary")

    def exit_process(code):
        raise Exited(code)

    _stub_modules(
        monkeypatch,
        {
            "gymnasium": {"make": make},
            "torch": {},
            "isaaclab_rl.rsl_rl": {
                "RslRlVecEnvWrapper": object,
                "handle_deprecated_rsl_rl_cfg": lambda agent_cfg, _version: agent_cfg,
            },
            "isaaclab_tasks.utils.parse_cfg": {"load_cfg_from_registry": lambda *_args: SimpleNamespace()},
            "rsl_rl.runners": {"OnPolicyRunner": object},
            "i4h_arena.scenes.base": {"load_scene": lambda *_args: scene},
        },
    )
    monkeypatch.setattr(rsl_rl_eval, "metadata", SimpleNamespace(version=lambda _name: "0"))
    # main() exits the process on failure so the physics shutdown handler
    # cannot mask it; surface that exit as an exception instead.
    monkeypatch.setattr(rsl_rl_eval, "os", SimpleNamespace(_exit=exit_process))
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsl_rl_eval.py",
            "--task",
            "ultrasound_probe_reach",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "evaluation.json"),
            "--num_envs",
            "4",
        ],
    )

    with pytest.raises(Exited):
        rsl_rl_eval.main()
    assert observed and observed[0].recorders is None
    assert observed[0].scene.num_envs == 4
