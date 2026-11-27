# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct and bridge live launches share a lease, using CPU Kit/client fixtures."""

import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from i4h_common.port_lease import PortLease

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    for directory in (
        "scripts",
        "local-agent",
        "common/i4h_common",
        "third_party/IsaacSim-fixture/skills/isaac-sim-remote/scripts",
    ):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "scripts/run_workflow.py").write_text(
        (ROOT / "scripts/run_workflow.py").read_text().replace("8226", str(port))
    )
    (tmp_path / "local-agent/bridge.sh").write_text(
        (ROOT / "local-agent/bridge.sh").read_text().replace("PORT=8226", f"PORT={port}")
    )
    shutil.copy2(ROOT / "common/i4h_common/port_lease.py", tmp_path / "common/i4h_common/port_lease.py")
    (tmp_path / "kit.py").write_text(
        "import socket,sys,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
        f's.bind(("127.0.0.1",{port})); s.listen(); print("kit ready",flush=True); time.sleep(60)'
    )
    (tmp_path / "scripts/launch.py").write_text(
        "import sys; from pathlib import Path; import run_workflow as r; "
        f"root=Path({str(tmp_path)!r}); "
        'p=r.Plan(r.Workflow(name=sys.argv[1],scene="fixture"),"live",Path(sys.argv[4]),'
        '[sys.executable,str(root/"kit.py")],{}); '
        "r._decline_telemetry=lambda:None; r._summary=lambda p:None; sys.exit(r.run(p))"
    )
    (tmp_path / "run.sh").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path}/scripts/launch.py" "$@"\n')
    (tmp_path / "run.sh").chmod(0o755)
    (tmp_path / "third_party/IsaacSim-fixture/skills/isaac-sim-remote/scripts/isaacsim_send.py").write_text(
        f'from pathlib import Path; p=Path({str(tmp_path / "queries")!r}); '
        'p.write_text(p.read_text()+"query\\n" if p.exists() else "query\\n"); '
        'print(\'{"output": "True"}\')'
    )
    return tmp_path, port, {**os.environ, "PYTHONPATH": str(tmp_path / "common"), "I4H_AGENT_BRIDGE_TIMEOUT": "15"}


def bridge(checkout, *args):
    root, _, env = checkout
    return subprocess.run(
        ["bash", str(root / "local-agent/bridge.sh"), *args], env=env, capture_output=True, text=True, timeout=30
    )


def test_bridge_direct_exclusion_and_restart(checkout):
    root, port, env = checkout
    try:
        start = bridge(checkout, "start", "first")
        assert start.returncode == 0, start.stderr
        assert "READY" in start.stdout
        assert bridge(checkout, "status", "first").returncode == 0
        # A direct launch cannot start a second Kit even though bridge start is done.
        direct = subprocess.run(
            [str(root / "run.sh"), "direct", "--live", "--run-dir", str(root / "runs/direct/now")],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert direct.returncode == 1
        assert "reserved" in direct.stderr
        assert bridge(checkout, "start", "second").returncode == 1
    finally:
        assert bridge(checkout, "stop", "first").returncode == 0
    assert bridge(checkout, "status", "first").returncode == 1
    with PortLease(port):
        pass
    try:
        assert bridge(checkout, "start", "restart").returncode == 0
    finally:
        assert bridge(checkout, "stop", "restart").returncode == 0


def test_bridge_never_queries_foreign_listener(checkout):
    root, port, _ = checkout
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
        sock.listen()
        assert bridge(checkout, "start", "foreign").returncode == 1
        # A stale PID file must also fail ownership before a status readiness query.
        state = root / ".run/local-agent-bridge"
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            (state / "foreign.pid").write_text(str(unrelated.pid))
            assert bridge(checkout, "status", "foreign").returncode == 1
        finally:
            unrelated.terminate()
            unrelated.wait()
        assert not (root / "queries").exists()


def test_simultaneous_bridge_starts_admit_one(checkout):
    root, _, env = checkout
    processes = [
        subprocess.Popen(
            ["bash", str(root / "local-agent/bridge.sh"), "start", name],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for name in ("one", "two")
    ]
    try:
        results = [p.communicate(timeout=25) for p in processes]
        assert sorted(p.returncode for p in processes) == [0, 1], results
    finally:
        for name in ("one", "two"):
            bridge(checkout, "stop", name)
        for p in processes:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
                p.wait()
