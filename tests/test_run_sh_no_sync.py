# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run.sh launches arena from the environment ./setup.sh made, without syncing it."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS_ROOT = Path(__file__).resolve().parent.parent
RUN_SH = WORKFLOWS_ROOT / "run.sh"


def test_every_arena_launch_skips_the_sync() -> None:
    # A sync resolves every path source of arena/pyproject.toml, so one missing
    # component checkout (xray-simulator, catheter-vasculature-solver) would stop
    # workflows that never import it (NVBug 7047868).
    launches = [line for line in RUN_SH.read_text().splitlines() if re.search(r"uv run --project arena\b", line)]
    assert launches, "run.sh no longer launches arena through uv run"
    for line in launches:
        assert "--no-sync" in line, line
