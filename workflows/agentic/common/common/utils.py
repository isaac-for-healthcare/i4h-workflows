# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

_WORKFLOW_ROOT = Path(__file__).resolve().parents[2]


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed


def resolve_path(path: str | Path | None, env: str | None, default: str | Path | None = None) -> Path | None:
    if path is None or path == "":
        if default is None:
            return None
        path = default
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    base = workflow_runs_dir() / env if env else workflow_runs_dir()
    return base / resolved


def workflow_runs_dir() -> Path:
    return _WORKFLOW_ROOT / "runs"
