# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for private third-party repository transport selection."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_SETUP = ROOT / "third_party" / "setup.sh"


def _run_setup(tmp_path: Path, root_origin: str) -> tuple[subprocess.CompletedProcess[str], str]:
    workspace = tmp_path / "workspace"
    third_party = workspace / "third_party"
    third_party.mkdir(parents=True)
    shutil.copy2(THIRD_PARTY_SETUP, third_party / "setup.sh")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git_log = tmp_path / "git.log"
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$*" >> "$GIT_TEST_LOG"\n'
        "repo=''\n"
        "if [ \"${1:-}\" = '-C' ]; then repo=$2; shift 2; fi\n"
        'if [ "${1:-}" = \'init\' ]; then mkdir -p "$repo/.git"; exit 0; fi\n'
        "if [ \"${1:-} ${2:-} ${3:-}\" = 'remote get-url origin' ]; then\n"
        '  if [ "$repo" = "$GIT_TEST_WORKSPACE" ]; then\n'
        '    if [ -n "$GIT_TEST_ROOT_ORIGIN" ]; then printf \'%s\\n\' "$GIT_TEST_ROOT_ORIGIN"; else exit 2; fi\n'
        "  else\n"
        "    printf '%s\\n' 'https://github.com/previous/origin.git'\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"${1:-}\" = 'rev-parse' ]; then\n"
        "  printf '%s\\n' '1111111111111111111111111111111111111111'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "GIT_TEST_LOG": str(git_log),
            "GIT_TEST_ROOT_ORIGIN": root_origin,
            "GIT_TEST_WORKSPACE": str(workspace),
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        }
    )
    result = subprocess.run(
        [str(third_party / "setup.sh"), "arena"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, git_log.read_text(encoding="utf-8") if git_log.exists() else ""


def test_private_repositories_follow_ssh_root_origin(tmp_path: Path) -> None:
    result, git_log = _run_setup(
        tmp_path,
        "git@github.com:isaac-for-healthcare/i4h-workflows-internal.git",
    )

    assert result.returncode == 0, result.stderr
    assert "private repository transport: ssh" in result.stdout
    assert "remote set-url origin git@github.com:isaac-for-healthcare/i4h-physics-simulation-internal.git" in git_log
    assert "remote set-url origin git@github.com:isaac-for-healthcare/i4h-sensor-simulation-internal.git" in git_log
    assert "remote set-url origin git@github.com:isaac-for-healthcare/i4h-digital-twin-internal.git" in git_log


def test_private_repositories_follow_https_root_origin(tmp_path: Path) -> None:
    result, git_log = _run_setup(
        tmp_path,
        "https://github.com/isaac-for-healthcare/i4h-workflows-internal.git",
    )

    assert result.returncode == 0, result.stderr
    assert "private repository transport: https" in result.stdout
    assert (
        "remote set-url origin https://github.com/isaac-for-healthcare/i4h-physics-simulation-internal.git" in git_log
    )
    assert "remote set-url origin https://github.com/isaac-for-healthcare/i4h-sensor-simulation-internal.git" in git_log
    assert "remote set-url origin https://github.com/isaac-for-healthcare/i4h-digital-twin-internal.git" in git_log


def test_private_repositories_default_to_https_without_root_origin(tmp_path: Path) -> None:
    result, git_log = _run_setup(
        tmp_path,
        "",
    )

    assert result.returncode == 0, result.stderr
    assert "private repository transport: https" in result.stdout
    assert "https://github.com/isaac-for-healthcare/i4h-physics-simulation-internal.git" in git_log
