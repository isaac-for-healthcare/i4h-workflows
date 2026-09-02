# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable path and metadata helpers shared by RL trainer backends."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_output_path(workflows_root: Path, value: str) -> Path:
    """Resolve an output path from an absolute or repository-relative value."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (workflows_root / path).resolve()


def resolve_input_path(workflows_root: Path, value: str) -> Path:
    """Resolve an existing input from an absolute, caller, or repository path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, workflows_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return resolve_output_path(workflows_root, value)


def resolve_run_dir(workflows_root: Path, workflow: str, requested: str | None) -> Path:
    if requested:
        return resolve_output_path(workflows_root, requested)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = workflows_root / "runs" / workflow / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = workflows_root / "runs" / workflow / f"{stamp}_{suffix:02d}"
    return candidate


def checkpoint_iteration(path: Path) -> int:
    """Return the numeric RSL-RL iteration from ``model_<N>.pt``."""
    try:
        return int(path.stem.removeprefix("model_"))
    except ValueError as exc:
        raise ValueError(f"invalid RSL-RL checkpoint name: {path.name}") from exc
