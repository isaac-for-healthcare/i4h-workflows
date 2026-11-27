# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKFLOWS_ROOT = Path(__file__).parents[2]
TUNE_SCRIPT = WORKFLOWS_ROOT / "arena" / "scripts" / "robot_pd_tune.py"


def test_robot_pd_dry_run_builds_the_scene_as_an_arena_environment() -> None:
    completed = subprocess.run(
        [sys.executable, str(TUNE_SCRIPT), "psm_reach", "--modes", "inspect-usd", "--presets", "newton", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["config"]["pd_diagnostics"]["modes"] == ["inspect-usd"]
    # Launcher options stay ahead of the environment, as Arena's parser expects.
    assert payload["arena_args"] == [
        "--presets",
        "newton",
        "--external_environment_class_path",
        "i4h_arena.environments:PsmReach",
        "psm_reach",
    ]
    assert payload["run_dir"].endswith("_pd_tuning")
