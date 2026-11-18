# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``local-agent/run.sh start`` stops waiting when the model container dies (NVBug 7048351).

``docker`` and ``curl`` are stubs: ``docker inspect`` answers from a list of states, one
per call (the last one repeats), and ``curl`` answers the health URL with a fixed code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_SH = ROOT / "local-agent" / "run.sh"

DOCKER = """#!/bin/sh
case "$1" in
  ps) exit 0 ;;
  inspect)
    n=$(cat "$STUB/inspect_calls" 2>/dev/null || echo 0)
    echo $((n + 1)) > "$STUB/inspect_calls"
    total=$(wc -l < "$STUB/states")
    [ "$n" -ge "$total" ] && n=$((total - 1))
    sed -n "$((n + 1))p" "$STUB/states"
    ;;
  logs) echo "torch.OutOfMemoryError: CUDA out of memory" ;;
  *) : ;;
esac
"""

CURL = """#!/bin/sh
printf '%s' "$STUB_HEALTH"
"""


def _start(tmp_path: Path, states: list[str], *, health: str = "000", timeout: int = 600):
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "states").write_text("".join(f"{state}\n" for state in states))
    for name, body in (("docker", DOCKER), ("curl", CURL)):
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("I4H_AGENT_")}
    env.update(
        PATH=f"{stub}:{env['PATH']}",
        HOME=str(tmp_path),
        STUB=str(stub),
        STUB_HEALTH=health,
        I4H_AGENT_PROFILE="nemotron-3.5-lightning",
        I4H_AGENT_GPU="0",
        I4H_AGENT_HF_CACHE=str(tmp_path / "hf"),
        I4H_AGENT_STARTUP_TIMEOUT=str(timeout),
        I4H_AGENT_WARMUP="0",
    )
    began = time.monotonic()
    result = subprocess.run(
        ["bash", str(RUN_SH), "start"], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    return result, time.monotonic() - began


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")


@pytest.mark.parametrize("state", ["exited 0", "dead 0", "restarting 1"])
def test_start_stops_waiting_when_the_container_stops(tmp_path, state) -> None:
    result, elapsed = _start(tmp_path, ["running 0", state])

    assert result.returncode == 1
    assert elapsed < 30
    assert f"is {state.split()[0]} after" in result.stderr
    assert "CUDA out of memory" in result.stderr
    assert "timed out" not in result.stderr


def test_start_stops_waiting_when_the_restart_policy_relaunches_a_crash(tmp_path) -> None:
    result, _ = _start(tmp_path, ["running 0", "running 1"])

    assert result.returncode == 1
    assert "is running after 1 restart(s) while loading" in result.stderr
    assert "CUDA out of memory" in result.stderr


def test_start_stops_waiting_when_the_container_is_removed(tmp_path) -> None:
    result, _ = _start(tmp_path, ["running 0", ""])

    assert result.returncode == 1
    assert "no longer exists" in result.stderr


def test_start_reports_ready_from_the_health_url(tmp_path) -> None:
    result, _ = _start(tmp_path, ["running 0"], health="200")

    assert result.returncode == 0, result.stderr
    assert "model ready at" in result.stdout


def test_start_still_times_out_when_the_container_stays_up(tmp_path) -> None:
    # A positive timeout runs the poll loop once: a container still loading keeps the wait going.
    result, _ = _start(tmp_path, ["running 0"], timeout=1)

    assert result.returncode == 1
    assert "waiting for model (HTTP 000)" in result.stdout
    assert int((tmp_path / "stub" / "inspect_calls").read_text()) >= 2
    assert "timed out waiting for model" in result.stderr
    assert "error: model server container" not in result.stderr
