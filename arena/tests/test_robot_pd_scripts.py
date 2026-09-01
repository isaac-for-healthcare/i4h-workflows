# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKFLOWS_ROOT = Path(__file__).parents[2]
TUNE_SCRIPT = WORKFLOWS_ROOT / "arena" / "scripts" / "robot_pd_tune.py"


def test_robot_pd_dry_run_uses_live_workflow_and_python_server() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(TUNE_SCRIPT),
            "surgical_reach_psm",
            "--modes",
            "inspect-usd",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["config"]["pd_diagnostics"]["modes"] == ["inspect-usd"]
    assert payload["live_command"][1:3] == ["surgical_reach_psm", "--live"]
    assert payload["remote_command"][1].endswith("isaacsim_send.py")
    assert "bridge-exec" not in payload["remote_command"]


def test_workflow_launcher_exposes_robot_pd_command() -> None:
    launcher = (WORKFLOWS_ROOT / "run.sh").read_text()

    assert "robot-pd|pd-tune|pd-diagnostics" in launcher
    assert "arena/scripts/robot_pd_tune.py" in launcher
