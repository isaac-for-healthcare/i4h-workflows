# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docker entrypoint command aliases."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"


def test_image_defaults_and_caches_use_persistent_state(tmp_path: Path) -> None:
    output = tmp_path / "environment"
    default_setup = tmp_path / "default-setup-projects"
    default_setup.write_text("common arena\n", encoding="utf-8")
    env = os.environ.copy()
    for name in ("GIT_TERMINAL_PROMPT", "HF_HOME", "I4H_SETUP_PROJECTS", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR"):
        env.pop(name, None)
    env.update(
        {
            "I4H_DEFAULT_SETUP_FILE": str(default_setup),
            "I4H_SKIP_SETUP": "1",
            "I4H_STATE_DIR": str(tmp_path / "state"),
            "I4H_WORKFLOWS": str(ROOT),
        }
    )

    result = subprocess.run(
        [
            str(ENTRYPOINT),
            "bash",
            "-c",
            'printf \'%s\\n\' "$I4H_SETUP_PROJECTS" "$HF_HOME" '
            f'"$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$GIT_TERMINAL_PROMPT" > {output}',
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").splitlines() == [
        "common arena",
        str(tmp_path / "state" / "huggingface"),
        str(tmp_path / "state" / "uv-cache"),
        str(tmp_path / "state" / "python"),
        "0",
    ]


def test_annotator_alias_hides_uv_command(tmp_path: Path) -> None:
    output = tmp_path / "uv.json"
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ENTRYPOINT_TEST_OUTPUT']).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "ENTRYPOINT_TEST_OUTPUT": str(output),
            "I4H_SKIP_SETUP": "1",
            "I4H_STATE_DIR": str(tmp_path / "state"),
            "I4H_WORKFLOWS": str(ROOT),
            "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(ENTRYPOINT), "i4h-annotator", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == [
        "run",
        "--no-sync",
        "--project",
        "tools/annotator",
        "i4h-annotator",
        "--help",
    ]


@pytest.mark.parametrize(
    ("command", "selection"),
    [
        (["./run.sh", "list"], "common"),
        (["./run.sh", "show", "surgical_reach_psm", "--rule-based"], "common"),
        (["./run.sh", "--help"], "common"),
        (["./stop.sh", "all"], "common"),
        (["./run.sh", "surgical_reach_psm", "--rule-based"], "common arena"),
    ],
)
def test_listing_and_stopping_set_up_only_the_common_environment(
    tmp_path: Path, command: list[str], selection: str
) -> None:
    output = tmp_path / "selection"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for script in ("run.sh", "stop.sh"):
        path = workspace / script
        path.write_text(
            "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$I4H_SETUP_PROJECTS" > "$ENTRYPOINT_TEST_OUTPUT"\n',
            encoding="utf-8",
        )
        path.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "ENTRYPOINT_TEST_OUTPUT": str(output),
            "I4H_SETUP_PROJECTS": "common arena",
            "I4H_SKIP_SETUP": "1",
            "I4H_STATE_DIR": str(tmp_path / "state"),
            "I4H_WORKFLOWS": str(workspace),
        }
    )

    result = subprocess.run([str(ENTRYPOINT), *command], cwd=ROOT, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").strip() == selection


def test_full_image_uses_baked_environments_without_setup(tmp_path: Path) -> None:
    output = tmp_path / "environment"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fingerprint = "full-image-fingerprint"
    (workspace / ".i4h-dependency-fingerprint").write_text(fingerprint, encoding="utf-8")
    marker = tmp_path / "setup-complete"
    marker.write_text(fingerprint, encoding="utf-8")
    baked_venvs = tmp_path / "full" / "venvs"
    baked_python = tmp_path / "full" / "python"
    env = os.environ.copy()
    env.update(
        {
            "I4H_BAKED_PYTHON_INSTALL_DIR": str(baked_python),
            "I4H_BAKED_VENV_ROOT": str(baked_venvs),
            "I4H_DEFAULT_SETUP_FILE": str(tmp_path / "missing-defaults"),
            "I4H_FULL_SETUP_MARKER": str(marker),
            "I4H_IMAGE_FLAVOR": "full",
            "I4H_SKIP_SETUP": "0",
            "I4H_STATE_DIR": str(tmp_path / "state"),
            "I4H_WORKFLOWS": str(workspace),
        }
    )

    result = subprocess.run(
        [
            str(ENTRYPOINT),
            "bash",
            "-c",
            f'printf \'%s\\n\' "$I4H_VENV_ROOT" "$UV_PYTHON_INSTALL_DIR" "$UV_NO_SYNC" > {output}',
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "using environments included in the full image" in result.stdout
    assert output.read_text(encoding="utf-8").splitlines() == [str(baked_venvs), str(baked_python), "1"]


def test_policy_alias_runs_the_unified_policy_command(tmp_path: Path) -> None:
    output = tmp_path / "policy.json"
    policy = tmp_path / "policy.sh"
    policy.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$I4H_SETUP_PROJECTS" "$I4H_THIRD_PARTY_TARGET" "$*" > "$ENTRYPOINT_TEST_OUTPUT"\n',
        encoding="utf-8",
    )
    policy.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "ENTRYPOINT_TEST_OUTPUT": str(output),
            "I4H_POLICY_COMMAND": str(policy),
            "I4H_SKIP_SETUP": "1",
            "I4H_STATE_DIR": str(tmp_path / "state"),
            "I4H_WORKFLOWS": str(ROOT),
        }
    )
    result = subprocess.run(
        [str(ENTRYPOINT), "i4h-policy", "gr00t_n17/scissor_pick_and_place"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").splitlines() == [
        "tasks/gr00t_n17",
        "tasks/gr00t_n17",
        "gr00t_n17/scissor_pick_and_place",
    ]


def test_policy_alias_serves_the_task_with_its_stacks_serve_module(tmp_path: Path) -> None:
    output = tmp_path / "argv"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$ENTRYPOINT_TEST_OUTPUT"\n', encoding="utf-8")
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "ENTRYPOINT_TEST_OUTPUT": str(output),
            "I4H_SKIP_SETUP": "1",
            "I4H_STATE_DIR": str(tmp_path / "state"),
            "I4H_WORKFLOWS": str(ROOT),
            "PATH": f"{tmp_path}:{env['PATH']}",
        }
    )
    result = subprocess.run(
        [str(ENTRYPOINT), "i4h-policy", "openpi_pi0/ultrasound_liver_scan", "--port", "8000"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").split() == [
        "run",
        "--no-sync",
        "--project",
        "tasks/openpi_pi0",
        "python",
        "-m",
        "i4h_tasks.openpi_pi0.serve",
        "ultrasound_liver_scan",
        "--port",
        "8000",
    ]
