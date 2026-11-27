#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify normal liver execution through run.sh and Arena's policy runner."""

import argparse
import re
import subprocess
from pathlib import Path

from i4h_rl.liver_experiment import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("rule-based", "policy"), default="rule-based")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        str(root / "run.sh"),
        "liver_reach_retract",
        "--mode",
        args.mode,
        "--episodes",
        "1",
        "--seed",
        "7341",
        "--record",
        "--run-dir",
        str(args.output.resolve()),
    ]
    if args.checkpoint:
        command += ["--checkpoint", str(args.checkpoint.resolve())]
    if args.headless:
        command += ["--headless"]
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output / "runner.log").write_text(result.stdout)
    print(result.stdout, end="")
    match = re.search(r"(\d+)/(\d+) episodes succeeded", result.stdout)
    report = dict(
        mode=args.mode,
        runner="isaaclab_arena.policy_runner",
        exit_code=result.returncode,
        successes=int(match[1]) if match else None,
        episodes=int(match[2]) if match else None,
    )
    write_json(args.output / "result.json", report)
    if result.returncode or not match or report["successes"] != 1 or report["episodes"] != 1:
        raise SystemExit("native liver workflow did not complete one successful full-cycle episode")


if __name__ == "__main__":
    main()
