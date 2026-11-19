# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run.sh launches arena from the environment ./setup.sh made, without syncing it."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

WORKFLOWS_ROOT = Path(__file__).resolve().parent.parent
RUN_SH = WORKFLOWS_ROOT / "run.sh"


def test_every_arena_launch_skips_the_sync() -> None:
    # A sync resolves every path source of arena/pyproject.toml, so one missing
    # component checkout (xray-simulator, catheter-vasculature-solver) would stop
    # workflows that never import it (NVBug 7047868).
    source = RUN_SH.read_text()
    launches = [line for line in source.splitlines() if re.search(r"uv run --project arena\b", line)]
    for line in launches:
        assert "--no-sync" in line, line
    # The native launcher executes the already-installed interpreter directly.
    # That also avoids dependency resolution and never creates an empty venv.
    assert "exec env -u VIRTUAL_ENV arena/.venv/bin/python" in source
    runner = (WORKFLOWS_ROOT / "scripts/run_workflow.py").read_text()
    assert 'ARENA_PYTHON = ROOT / "arena" / ".venv" / "bin" / "python"' in runner


def test_robot_pd_refuses_an_unsynced_arena(tmp_path: Path) -> None:
    # Without the guard, `uv run --no-sync` would create an empty arena/.venv,
    # which also hides run.sh's own "Run: ./setup.sh" check for later launches.
    shutil.copy2(RUN_SH, tmp_path / "run.sh")
    (tmp_path / "arena").mkdir()
    result = subprocess.run(
        [str(tmp_path / "run.sh"), "robot-pd", "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "arena is not synced. Run: ./setup.sh" in result.stderr
    assert not (tmp_path / "arena" / ".venv").exists()
