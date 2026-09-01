# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for setup selections and external virtual environments."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP = ROOT / "setup.sh"


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    project = workspace / "common"
    project.mkdir(parents=True)
    shutil.copy2(SETUP, workspace / "setup.sh")
    (project / "pyproject.toml").write_text("[project]\nname = 'test-common'\nversion = '0'\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'mkdir -p "$UV_PROJECT_ENVIRONMENT"\n'
        'printf \'%s\\n\' "$*" > "$SETUP_TEST_OUTPUT"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return workspace, bin_dir


def test_selected_project_uses_external_persistent_venv(tmp_path: Path) -> None:
    workspace, bin_dir = _workspace(tmp_path)
    venv_root = tmp_path / "state" / "venvs"
    output = tmp_path / "uv-args"
    env = os.environ.copy()
    env.update(
        {
            "I4H_SETUP_PROJECTS": "common",
            "I4H_VENV_ROOT": str(venv_root),
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "SETUP_TEST_OUTPUT": str(output),
        }
    )

    result = subprocess.run(
        [str(workspace / "setup.sh")],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").strip() == "sync --project common"
    assert (workspace / "common" / ".venv").is_symlink()
    assert (workspace / "common" / ".venv").resolve() == venv_root / "common"

    linked = subprocess.run(
        [str(workspace / "setup.sh"), "links"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert linked.returncode == 0, linked.stderr
    assert linked.stdout.strip() == "linked: common"


def test_unknown_project_selection_is_rejected(tmp_path: Path) -> None:
    workspace, bin_dir = _workspace(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "I4H_SETUP_PROJECTS": "not-a-project",
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(workspace / "setup.sh")],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == "unknown I4H_SETUP_PROJECTS entry: not-a-project"
