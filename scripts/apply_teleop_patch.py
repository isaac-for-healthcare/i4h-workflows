#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install or migrate the supported Isaac Lab recording patch without undoing edits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FRESH_ACTION = (
    "                    # The command sampled above predates the reset; read a fresh one for the new episode.\n"
    "                    action = teleop_interface.advance()\n"
)


def apply(checkout: Path, patch: Path) -> None:
    current = patch.read_text()
    # The pre-eb9b200f patch differs by exactly these two added lines. --recount
    # recalculates hunk lengths; Git still checks every context and changed line.
    previous = current.replace("".join("+" + line for line in FRESH_ACTION.splitlines(keepends=True)), "")
    migration = (
        "--- a/scripts/tools/record_demos.py\n"
        "+++ b/scripts/tools/record_demos.py\n"
        "@@ -755,6 +755,8 @@\n"
        "                     camera_feed_session.refresh()\n"
        "                     should_reset_recording_instance = False\n"
        + "".join("+" + line for line in FRESH_ACTION.splitlines(keepends=True))
        + " \n"
        "                 if action is None:\n"
    )

    def git_apply(text: str, *options: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(checkout), "apply", "--recount", *options, "-"],
            input=text,
            text=True,
            capture_output=True,
        )

    if git_apply(current, "--reverse", "--check").returncode == 0:
        print("Isaac Lab teleop patch already current")
        return
    if git_apply(current, "--check").returncode == 0:
        selected = current
    elif (
        git_apply(previous, "--reverse", "--check").returncode == 0 and git_apply(migration, "--check").returncode == 0
    ):
        selected = migration
    else:
        raise RuntimeError(
            f"Unrecognized Isaac Lab teleop edits in {checkout}/scripts/tools/record_demos.py. "
            "Inspect git diff there and reconcile with third_party/isaaclab_teleop_no_record.patch, then rerun setup. "
            "No files were changed."
        )
    result = git_apply(selected)
    if result.returncode:
        raise RuntimeError(result.stderr)
    print("Isaac Lab teleop patch installed")


if __name__ == "__main__":
    try:
        apply(Path(sys.argv[1]), Path(__file__).resolve().parents[1] / "third_party/isaaclab_teleop_no_record.patch")
    except RuntimeError as error:
        sys.exit(str(error))
