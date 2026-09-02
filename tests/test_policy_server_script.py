# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy container command tests."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "docker" / "policy-server.sh"


def _run(tmp_path: Path, *args: str, stack: str = "gr00t_n15") -> tuple[subprocess.CompletedProcess[str], dict | None]:
    output = tmp_path / "uv.json"
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "payload = {'args': sys.argv[1:]}\n"
        "payload['connect'] = os.environ.get('I4H_ZENOH_CONNECT')\n"
        "payload['listen'] = os.environ.get('I4H_ZENOH_LISTEN')\n"
        "Path(os.environ['POLICY_TEST_OUTPUT']).write_text(json.dumps(payload))\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "I4H_POLICY_STACK": stack,
            "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}",
            "POLICY_TEST_OUTPUT": str(output),
        }
    )
    result = subprocess.run([str(SCRIPT), *args], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    return result, json.loads(output.read_text()) if output.exists() else None


def test_workflow_id_selects_task_and_namespace(tmp_path: Path) -> None:
    result, payload = _run(tmp_path, "scissor_pick_and_place")

    assert result.returncode == 0, result.stderr
    assert payload == {
        "args": [
            "run",
            "--project",
            "tasks/gr00t_n15",
            "--no-sync",
            "python",
            "-m",
            "i4h_tasks.gr00t_n15.server",
            "--namespace",
            "scissor_pick_and_place",
            "--preload",
            "gr00t_n15/scissor_pick_and_place",
        ],
        "connect": None,
        "listen": "tcp/0.0.0.0:7448",
    }


def test_remote_policy_can_listen_for_the_workflow(tmp_path: Path) -> None:
    result, payload = _run(tmp_path, "gr00t_n15/scissor_pick_and_place", "--listen", "tcp/0.0.0.0:7447")

    assert result.returncode == 0, result.stderr
    assert payload is not None
    assert payload["connect"] is None
    assert payload["listen"] == "tcp/0.0.0.0:7447"


def test_task_id_selects_another_policy_stack(tmp_path: Path) -> None:
    result, payload = _run(tmp_path, "gr00t_n16/locomanip_push_cart")

    assert result.returncode == 0, result.stderr
    assert payload is not None
    assert payload["args"][2] == "tasks/gr00t_n16"
    assert "i4h_tasks.gr00t_n16.server" in payload["args"]


def test_stack_option_selects_policy_for_a_workflow_id(tmp_path: Path) -> None:
    result, payload = _run(tmp_path, "scissor_pick_and_place", "--stack", "gr00t_n17")

    assert result.returncode == 0, result.stderr
    assert payload is not None
    assert payload["args"][2] == "tasks/gr00t_n17"
    assert "gr00t_n17/scissor_pick_and_place" in payload["args"]
