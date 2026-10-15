# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The RLinf simulator must report why it failed instead of exiting 0 before becoming ready."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from i4h_rl import sim_server


class _Exited(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--scene",
        "endoluminal_navigation",
        "--socket",
        str(tmp_path / "sim.sock"),
        "--ready-file",
        str(tmp_path / "simulator.ready"),
        "--num-envs",
        "1",
        "--max-episode-steps",
        "600",
        "--env-spacing",
        "2.0",
        "--presets",
        "physx",
    ]


@pytest.fixture
def fake_kit(monkeypatch):
    """Stand in for Kit, the scene loader, gymnasium and torch; record how the process ends."""
    closed: list[bool] = []
    app = types.SimpleNamespace(close=lambda: closed.append(True))
    isaaclab_app = types.ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = lambda **kwargs: types.SimpleNamespace(app=app)
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    scene = types.SimpleNamespace(
        configure_args=lambda args: None,
        gym_spec=lambda: ("Isaac-Endoluminal-v0", types.SimpleNamespace(scene=types.SimpleNamespace())),
    )
    scenes_base = types.ModuleType("i4h_arena.scenes.base")
    scenes_base.load_scene = lambda name, args: scene
    monkeypatch.setitem(sys.modules, "i4h_arena.scenes.base", scenes_base)

    def make(gym_id, **kwargs):
        raise RuntimeError("Failed to create device!")

    gymnasium = types.ModuleType("gymnasium")
    gymnasium.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    def exit_now(code: int):
        raise _Exited(code)

    monkeypatch.setattr(os, "_exit", exit_now)
    return closed


def test_an_environment_creation_failure_is_printed_and_exits_non_zero(tmp_path, fake_kit, capsys) -> None:
    with pytest.raises(_Exited) as exited:
        sim_server._serve(_argv(tmp_path))

    assert exited.value.code == 1
    stderr = capsys.readouterr().err
    assert "Traceback" in stderr
    assert "RuntimeError: Failed to create device!" in stderr
    # Closing Kit first would end the process with status 0 before any of this was reported.
    assert fake_kit == []
    assert not (tmp_path / "simulator.ready").exists()
    assert not (tmp_path / "sim.sock").exists()
