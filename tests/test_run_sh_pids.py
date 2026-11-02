# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-tracking regression tests for the shared launcher and stop.sh.

Two launches in one checkout used to share a single pid file, so the second one
truncated the first one's list and the first run's teardown left its policy
backend running.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

WORKFLOWS_ROOT = Path(__file__).resolve().parent.parent
RUN_SH = WORKFLOWS_ROOT / "run.sh"
STOP_SH = WORKFLOWS_ROOT / "stop.sh"
WORKFLOW = "scissor_pick_and_place"

# Stands in for the Arena CLI: plays a backend this launch started by adding a
# process to the one tracking file that is not another run's.
LAUNCH_TRACKED_BACKEND = """
    set -- $(ls "$I4H_PID_DIR" | grep -vx other-run)
    [ $# -eq 1 ] || exit 3
    sleep 300 </dev/null >/dev/null 2>&1 &
    echo $! | tee -a "$I4H_PID_DIR/$1" > "$I4H_RUN_DIR/backend.pid"
    exit 0
"""


def _uv_running_arena_step(tmp_path: Path, step: str) -> dict[str, str]:
    """Return an environment whose ``uv`` runs ``step`` in place of the Arena CLI."""
    uv = shutil.which("uv")
    assert uv is not None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "uv"
    shim.write_text(
        f'#!/usr/bin/env bash\ncase "$*" in\n  *i4h_arena.cli*){step};;\nesac\nexec {shlex.quote(uv)} "$@"\n'
    )
    shim.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def _running(pid: int) -> bool:
    """Whether ``pid`` is alive; a zombie waiting for its reaper is not."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _stopped(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while _running(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_run_tears_down_only_the_backends_it_tracked(tmp_path: Path) -> None:
    pid_dir = tmp_path / "pids.d"
    pid_dir.mkdir()
    run_dir = tmp_path / "run"
    backend_pid = run_dir / "backend.pid"
    env = _uv_running_arena_step(tmp_path, LAUNCH_TRACKED_BACKEND)
    env["I4H_PID_DIR"] = str(pid_dir)
    other_backend = subprocess.Popen(["sleep", "300"])
    try:
        (pid_dir / "other-run").write_text(f"{other_backend.pid}\n")

        result = subprocess.run(
            [str(RUN_SH), WORKFLOW, "--dry-run", "--no-backend", "--run-dir", str(run_dir)],
            cwd=WORKFLOWS_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert _stopped(int(backend_pid.read_text()))
        assert other_backend.poll() is None
        assert [path.name for path in pid_dir.iterdir()] == ["other-run"]
        assert (pid_dir / "other-run").read_text() == f"{other_backend.pid}\n"
    finally:
        other_backend.kill()
        other_backend.wait()
        if backend_pid.exists() and _running(pid := int(backend_pid.read_text())):
            os.kill(pid, signal.SIGKILL)


def test_stop_all_stops_the_backends_of_every_run(tmp_path: Path) -> None:
    # A copy treats tmp_path as its checkout, so its interpreter sweep cannot
    # reach a real run in this one.
    root = tmp_path / "checkout"
    pid_dir = root / ".run" / "pids.d"
    pid_dir.mkdir(parents=True)
    shutil.copy2(STOP_SH, root / "stop.sh")
    backends = [subprocess.Popen(["sleep", "300"]) for _ in range(3)]
    try:
        (pid_dir / "run.a").write_text(f"{backends[0].pid}\n")
        (pid_dir / "run.b").write_text(f"{backends[1].pid}\n")
        # Where a launcher from before per-run tracking records its backends.
        (root / ".run" / "pids").write_text(f"{backends[2].pid}\n")
        env = {name: value for name, value in os.environ.items() if name != "I4H_PID_DIR"}

        result = subprocess.run(
            [str(root / "stop.sh"), "all"], cwd=tmp_path, env=env, capture_output=True, text=True, check=True
        )

        assert [backend.wait(timeout=5) for backend in backends] == [-signal.SIGTERM] * 3
        assert result.stdout.splitlines()[-1] == "stopped 3 process(es)"
        assert not any(pid_dir.iterdir())
        assert not (root / ".run" / "pids").exists()
    finally:
        for backend in backends:
            backend.kill()
            backend.wait()
