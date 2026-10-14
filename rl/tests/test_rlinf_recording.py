# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disable the inherited episode recorder before constructing the RL simulator."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from i4h_rl import sim_server


def test_rl_simulator_does_not_inherit_episode_recorders(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1), recorders=object(), terminations=SimpleNamespace(time_out=object())
    )
    scene = SimpleNamespace(configure_args=lambda _: None, gym_spec=lambda: ("test", cfg))
    observed = []
    closed = []

    def make(_id, **kwargs):
        observed.append(kwargs["cfg"])
        raise RuntimeError("stop at gym construction boundary")

    modules = {
        "isaaclab.app": {
            "AppLauncher": lambda **kwargs: SimpleNamespace(app=SimpleNamespace(close=lambda: closed.append(True)))
        },
        "torch": {},
        "gymnasium": {"make": make},
        "i4h_arena.scenes.base": {"load_scene": lambda *args: scene},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match="gym construction boundary"):
        sim_server.main(
            [
                "--scene",
                "g1_trocar",
                "--socket",
                str(tmp_path / "socket"),
                "--ready-file",
                str(tmp_path / "ready"),
                "--num-envs",
                "64",
                "--max-episode-steps",
                "250",
                "--env-spacing",
                "6",
                "--presets",
                "physx",
            ]
        )
    assert observed and observed[0].recorders is None
    assert observed[0].scene.num_envs == 64
    assert closed
