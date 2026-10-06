# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate pinned third-party checkouts from ``third_party/setup.sh``, the one place their revisions are named."""

from __future__ import annotations

import re
from pathlib import Path


def pinned_dir(workflows_root: Path, variable: str) -> Path:
    """``third_party/<dir>`` for a ``<VARIABLE>="<dir>"`` line of ``third_party/setup.sh``."""
    setup = Path(workflows_root) / "third_party" / "setup.sh"
    match = re.search(rf'^{re.escape(variable)}="([^"]+)"', setup.read_text(), re.MULTILINE)
    if match is None:
        raise SystemExit(f"{setup} does not define {variable}")
    return Path(workflows_root) / "third_party" / match.group(1)


def isaaclab_dir(workflows_root: Path) -> Path:
    """The pinned Isaac Lab checkout."""
    return pinned_dir(workflows_root, "ISAACLAB_DIR")
