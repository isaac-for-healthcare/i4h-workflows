# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launcher integration: reject foreign/failed servers before simulator execution."""

import importlib.util
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from i4h_common.port_lease import PortLease

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("managed_launcher", ROOT / "scripts/run_workflow.py")
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


@pytest.fixture(autouse=True)
def restore_signals():
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


def test_exited_server_cannot_be_satisfied_by_foreign_listener(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        server = subprocess.Popen([sys.executable, "-c", "pass"])
        server.wait()
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="server exited"):
            launcher._wait_for_port(sock.getsockname()[1], server, tmp_path / "log")
        assert time.monotonic() - started < 1


def test_running_foreign_listener_cannot_satisfy_readiness(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with pytest.raises(RuntimeError, match="outside process"):
                launcher._wait_for_port(sock.getsockname()[1], server, tmp_path / "log")
        finally:
            server.terminate()
            server.wait()


def fixture_plan(tmp_path, monkeypatch):
    plan = launcher.plan(["scissor_pick_and_place", "--policy", "--port", "6123", "--run-dir", str(tmp_path / "run")])
    monkeypatch.setattr(launcher, "_decline_telemetry", lambda: None)
    monkeypatch.setattr(launcher, "_summary", lambda _plan: None)
    marker = tmp_path / "simulator-started"
    plan.command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
        "--remote_port",
        "6123",
    ]
    return plan, marker


def test_explicit_collision_never_starts_simulator(tmp_path, monkeypatch):
    plan, marker = fixture_plan(tmp_path, monkeypatch)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        plan.port = sock.getsockname()[1]
        assert launcher.run(plan) == 1
    assert not marker.exists()


def test_failed_start_releases_lease(tmp_path, monkeypatch):
    plan, marker = fixture_plan(tmp_path, monkeypatch)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        plan.port = sock.getsockname()[1]
    plan.server = [sys.executable, "-c", "raise SystemExit(7)", "--port", str(plan.port)]
    assert launcher.run(plan) == 1
    assert not marker.exists()
    with PortLease(plan.port):
        pass


def test_automatic_port_updates_server_and_client_together(tmp_path, monkeypatch):
    plan, marker = fixture_plan(tmp_path, monkeypatch)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        plan.port = sock.getsockname()[1]
    original = plan.port
    observed = tmp_path / "client-port"
    plan.automatic_port = True
    plan.server = [
        sys.executable,
        "-c",
        "import socket,sys,time; s=socket.socket(); "
        's.bind(("127.0.0.1",int(sys.argv[-1]))); s.listen(); time.sleep(30)',
        "--port",
        str(original),
    ]
    plan.command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; import sys; " f"Path({str(observed)!r}).write_text(sys.argv[-1])",
        "--remote_port",
        str(original),
    ]
    with PortLease(original):
        assert launcher.run(plan) == 0
    assert original < plan.port < original + 50
    assert observed.read_text() == str(plan.port)
    with PortLease(plan.port):
        pass
