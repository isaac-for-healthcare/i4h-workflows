# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate the workflow repository from any component environment.

Every project installs its siblings as editable path dependencies, so the tree
is always on disk somewhere. Registries need that path to glob manifests, and
they must find it identically whether they are running from ``arena``'s venv,
``tasks/basic``'s, or a bare checkout.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ENV_VAR = "I4H_WORKFLOWS"

#: Files that together identify the workflow root unambiguously.
_MARKERS = ("DESIGN.md", "engine", "common")


def _looks_like_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in _MARKERS)


@lru_cache(maxsize=1)
def workflow_root() -> Path:
    """Absolute path to the repository root.

    Resolution order: ``$I4H_WORKFLOWS``, then this file's ancestors (works for
    editable installs, which is every install we make).
    """
    override = os.environ.get(ENV_VAR)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not _looks_like_root(candidate):
            raise RuntimeError(f"{ENV_VAR}={candidate} does not look like the workflow repository (want {_MARKERS})")
        return candidate

    for parent in Path(__file__).resolve().parents:
        if _looks_like_root(parent):
            return parent

    raise RuntimeError(f"cannot locate the workflow tree from {__file__}; set {ENV_VAR} to the repository root")


def artifact_dir(*parts: str) -> Path:
    """A writable path under ``runs/`` for datasets, logs and checkpoints."""
    path = workflow_root() / "runs"
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path
