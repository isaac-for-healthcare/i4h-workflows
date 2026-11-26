# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CPU sockets/processes exercise shared leases and Linux ownership checks."""

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from i4h_common.port_lease import PortLease, PortUnavailable, listener_owned, reserve_port

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "common")}


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("exception", [False, True])
def test_lease_cleanup_and_no_descriptor_inheritance(exception):
    port = unused_port()
    try:
        with PortLease(port):
            with pytest.raises(PortUnavailable, match="reserved"):
                PortLease(port)
            # Explicitly disable Popen's own fd closing to test close-on-exec.
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], close_fds=False)
            if exception:
                raise ValueError("fixture")
    except ValueError:
        assert exception
    try:
        with PortLease(port):
            pass
    finally:
        child.terminate()
        child.wait()


def test_cross_checkout_simultaneous_automatic_ports(tmp_path):
    port = unused_port()
    code = (
        "from i4h_common.port_lease import reserve_port; import sys; "
        "lease=reserve_port(int(sys.argv[1]), automatic=True); "
        "print(lease.port, flush=True); input()"
    )
    processes = []
    try:
        for name in ("checkout-a", "checkout-b"):
            directory = tmp_path / name
            directory.mkdir()
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", code, str(port)],
                    cwd=directory,
                    env=ENV,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
        ports = [int(process.stdout.readline()) for process in processes]
        assert len(set(ports)) == 2
        assert all(port <= chosen < port + 50 for chosen in ports)
    finally:
        for process in processes:
            process.communicate("\n", timeout=5)
    with PortLease(port):
        pass


def test_foreign_listener_and_explicit_collision():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        with pytest.raises(PortUnavailable, match="listener"):
            reserve_port(port)
        with pytest.raises(RuntimeError, match="outside process|Cannot establish"):
            listener_owned(port, 999999999)
        assert listener_owned(port, os.getpid())


def test_descendant_owned_listener():
    port = unused_port()
    code = (
        "import subprocess,sys; "
        'p=subprocess.Popen([sys.executable,"-c",'
        '"import socket,sys; s=socket.socket(); s.bind((\\"127.0.0.1\\",int(sys.argv[1]))); '
        's.listen(); print(\\"ready\\",flush=True); input()",sys.argv[1]]); p.wait()'
    )
    process = subprocess.Popen([sys.executable, "-c", code, str(port)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert process.stdout.readline().strip() == b"ready"
        assert listener_owned(port, process.pid)
    finally:
        process.communicate(b"\n", timeout=5)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_releases_lease(signum):
    port = unused_port()
    code = (
        "import signal,sys; from i4h_common.port_lease import PortLease; "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
        'lease=PortLease(int(sys.argv[1])); print("ready",flush=True); signal.pause()'
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(port)], env=ENV, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        assert process.stdout.readline().strip() == b"ready"
        process.send_signal(signum)
        process.wait(timeout=5)
        with PortLease(port):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_ownership_inspection_failure_is_fatal(monkeypatch):
    import i4h_common.port_lease as helper

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()

        def denied(_pid):
            raise PermissionError("fixture")

        monkeypatch.setattr(helper, "_descendants", denied)
        with pytest.raises(RuntimeError, match="Cannot establish"):
            listener_owned(sock.getsockname()[1], os.getpid())


def test_proven_listener_does_not_require_unrelated_crash_reporter_fds(monkeypatch):
    import i4h_common.port_lease as helper

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        real_iterdir = Path.iterdir

        def descriptors(path):
            if str(path) == "/proc/999999998/fd":
                raise PermissionError("non-dumpable crash reporter")
            return real_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", descriptors)
        monkeypatch.setattr(helper, "_descendants", lambda pid: {pid, os.getpid()})
        assert listener_owned(sock.getsockname()[1], 999999998)
