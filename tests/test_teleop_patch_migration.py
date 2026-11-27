# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise patch state recognition with real Git and isolated files."""

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("teleop_patch", ROOT / "scripts/apply_teleop_patch.py")
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)
PATCH = ROOT / "third_party/isaaclab_teleop_no_record.patch"


@pytest.mark.parametrize("state", ["fresh", "previous", "current", "conflict"])
def test_patch_migration(tmp_path, monkeypatch, state):
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    text = PATCH.read_text()
    # Reconstruct the pinned preimage's relevant hunks, with inert padding between.
    for section in text.split("diff --git ")[1:]:
        relative = section.splitlines()[0].split()[0][2:]
        lines = []
        for hunk in re.split(r"(?m)^@@ ", section)[1:]:
            header, body = hunk.split("\n", 1)
            start = int(re.match(r"-(\d+)", header)[1])
            lines.extend(["# padding\n"] * (start - 1 - len(lines)))
            lines.extend(line[1:] for line in body.splitlines(keepends=True) if line.startswith((" ", "-")))
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines))
    source = tmp_path / "scripts/tools/record_demos.py"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    if state in ("previous", "current"):
        selected = text
        if state == "previous":
            selected = selected.replace(
                "".join("+" + line for line in patcher.FRESH_ACTION.splitlines(keepends=True)), ""
            )
        subprocess.run(["git", "-C", str(tmp_path), "apply", "--recount", "-"], input=selected, text=True, check=True)
    if state == "conflict":
        source.write_text(source.read_text().replace("action = teleop_interface.advance()", "action = custom()"))
    before = source.read_bytes()
    if state == "conflict":
        with pytest.raises(RuntimeError, match="Unrecognized"):
            patcher.apply(tmp_path, PATCH)
        assert source.read_bytes() == before
        return
    patcher.apply(tmp_path, PATCH)
    assert patcher.FRESH_ACTION in source.read_text()
    after = source.read_bytes()
    patcher.apply(tmp_path, PATCH)
    assert source.read_bytes() == after
    if state == "current":
        assert before == after
