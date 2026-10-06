# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""third_party/setup.sh names each pinned checkout once; everything else derives its path from it."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rl"))

from i4h_rl.third_party import isaaclab_dir, pinned_dir  # noqa: E402

SETUP = (ROOT / "third_party" / "setup.sh").read_text()


def pin(name: str) -> str:
    return re.search(rf'^{name}="([^"]+)"', SETUP, re.MULTILINE).group(1)


def test_isaaclab_and_arena_pins_name_their_directories():
    assert pin("ISAACLAB_DIR") == f"IsaacLab-{pin('ISAACLAB_REV')[:7]}"
    assert pin("ISAACLAB_ARENA_DIR") == f"IsaacLab-Arena-{pin('ISAACLAB_ARENA_REV')[:7]}"
    assert isaaclab_dir(ROOT) == ROOT / "third_party" / pin("ISAACLAB_DIR")
    assert pinned_dir(ROOT, "ISAACLAB_ARENA_DIR").name == pin("ISAACLAB_ARENA_DIR")
    with pytest.raises(SystemExit, match="does not define"):
        pinned_dir(ROOT, "NO_SUCH_DIR")


def test_arena_project_follows_the_pins():
    project = tomllib.loads((ROOT / "arena" / "pyproject.toml").read_text())
    sources = project["tool"]["uv"]["sources"]
    lab = f"../third_party/{pin('ISAACLAB_DIR')}"
    assert sources["isaaclab-dev"]["path"] == lab
    for name, source in sources.items():
        if name.startswith("isaaclab") and name not in ("isaaclab-dev", "isaaclab-arena"):
            assert source["path"] == f"{lab}/source/{name.replace('-', '_')}", name
            assert name in project["project"]["dependencies"], f"{name}: a workspace member must be named directly"
    assert sources["isaaclab-arena"]["path"] == f"../third_party/{pin('ISAACLAB_ARENA_DIR')}"
    overrides = project["tool"]["uv"]["override-dependencies"]
    assert "warp-lang==1.17.0" in overrides and "newton[sim]==1.6.1" in overrides
    assert not any(o.startswith("usd-core") for o in overrides), "pxr comes from Isaac Lab's usd-exchange"


def test_no_tracked_file_names_a_stale_checkout():
    listed = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.split()
    stale = []
    for path in listed:
        if path.endswith((".lock", ".patch")) or path == "tests/test_third_party_pins.py":
            continue
        try:
            text = (ROOT / path).read_text()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for old in ("IsaacLab-ffff603", "IsaacLab-Arena-0a1b8c2"):
            if old in text:
                stale.append(f"{path}: {old}")
    assert not stale, stale


def test_rsl_rl_launches_isaac_labs_unified_trainer(tmp_path, monkeypatch):
    from i4h_rl.backends import rsl_rl
    from i4h_rl.profile import load_profile

    calls = []
    monkeypatch.setattr(rsl_rl, "_runtime_python", lambda root, explicit: Path("/runtime/python"))
    monkeypatch.setattr(rsl_rl, "_runtime_env", lambda root, profile: {})
    monkeypatch.setattr(rsl_rl, "_preflight", lambda runtime, env: None)
    monkeypatch.setattr(rsl_rl.subprocess, "call", lambda command, **kw: calls.append(command) or 1)
    args = SimpleNamespace(only_eval=False, runtime_python=None, run_dir=str(tmp_path / "run"), video=False)
    rsl_rl.launch(args, load_profile("ultrasound_probe_reach"), ROOT, num_envs=2, epochs=1, overrides=())
    command = calls[0]
    assert command[1] == str(isaaclab_dir(ROOT) / "scripts" / "reinforcement_learning" / "train.py")
    assert command[2:4] == ["--rl_library", "rsl_rl"]
