# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CPU sockets/processes exercise shared leases and Linux ownership checks."""

import errno
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
            listener_owned(sock.getsockname()[1], 999999998)


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


@pytest.mark.parametrize(
    "stage,code", [("socket", errno.EAFNOSUPPORT), ("socket", errno.EPROTONOSUPPORT), ("bind", errno.EADDRNOTAVAIL)]
)
def test_ipv4_only_host_can_lease_and_verify_listeners(monkeypatch, stage, code):
    import i4h_common.port_lease as helper

    original_socket = socket.socket
    original_read = Path.read_text

    class DisabledIPv6:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            raise OSError(code, "IPv6 disabled")

    def sockets(family=socket.AF_INET, *args, **kwargs):
        if family == socket.AF_INET6:
            if stage == "socket":
                raise OSError(code, "IPv6 unavailable")
            return DisabledIPv6()
        return original_socket(family, *args, **kwargs)

    def tables(path, *args, **kwargs):
        if path == Path("/proc/net/tcp6"):
            raise FileNotFoundError("IPv6 table unavailable")
        return original_read(path, *args, **kwargs)

    port = unused_port()
    monkeypatch.setattr(helper.socket, "socket", sockets)
    monkeypatch.setattr(Path, "read_text", tables)
    with PortLease(port), original_socket() as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()
        assert listener_owned(port, os.getpid())
        with pytest.raises(RuntimeError, match="outside process|Cannot establish"):
            listener_owned(port, 999999999)
    with PortLease(port):
        pass


@pytest.mark.parametrize("family", ["tcp", "tcp6"])
@pytest.mark.parametrize("error_type", [FileNotFoundError, PermissionError])
def test_enabled_tcp_table_inspection_errors_are_fatal(monkeypatch, family, error_type):
    import i4h_common.port_lease as helper

    original_read = Path.read_text

    def denied(path, *args, **kwargs):
        if path == Path(f"/proc/net/{family}"):
            raise error_type("fixture table unavailable")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr(helper, "_probe_port", lambda *_args: True)
    with pytest.raises(RuntimeError, match="Cannot inspect"):
        helper.listener_inodes(12345)


def test_parent_owned_listener_is_checked_before_child_tree(monkeypatch):
    original_iterdir = Path.iterdir

    def no_child_walk(path):
        if path == Path(f"/proc/{os.getpid()}/task"):
            raise AssertionError("parent already owns every listener; child walk is unnecessary")
        return original_iterdir(path)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setattr(Path, "iterdir", no_child_walk)
        assert listener_owned(listener.getsockname()[1], os.getpid())


@pytest.mark.parametrize("exit_point", ["task_open", "task_iteration", "children_read"])
def test_disappearing_child_does_not_abort_descendant_listener(monkeypatch, exit_point):
    original_iterdir = Path.iterdir
    original_read = Path.read_text
    parent, vanished = 999999997, 999999998

    def gone_during_iteration():
        raise FileNotFoundError("child exited after opening task directory")
        yield  # make the error occur during iteration

    def directories(path):
        if path == Path(f"/proc/{parent}/task"):
            return iter([Path(f"/proc/{parent}/task/{parent}")])
        if path == Path(f"/proc/{vanished}/task"):
            if exit_point == "task_open":
                raise FileNotFoundError("child exited before opening task directory")
            if exit_point == "task_iteration":
                return gone_during_iteration()
            return iter([Path(f"/proc/{vanished}/task/{vanished}")])
        return original_iterdir(path)

    def children(path, *args, **kwargs):
        if path == Path(f"/proc/{parent}/task/{parent}/children"):
            # The pending stack visits the vanished child before the listener owner.
            return f"{os.getpid()} {vanished}"
        if path == Path(f"/proc/{vanished}/task/{vanished}/children"):
            raise FileNotFoundError("thread exited before reading children")
        return original_read(path, *args, **kwargs)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setattr(Path, "iterdir", directories)
        monkeypatch.setattr(Path, "read_text", children)
        assert listener_owned(listener.getsockname()[1], parent)


def test_ipv6_listener_still_blocks_port_lease():
    import i4h_common.port_lease as helper

    if not helper._probe_port(socket.AF_INET6, 0):
        pytest.skip("IPv6 unavailable on this host")
    with socket.socket(socket.AF_INET6) as listener:
        listener.bind(("::1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(PortUnavailable, match="listener"):
            PortLease(port)
        assert listener_owned(port, os.getpid())


def test_ipv6_permission_errors_are_not_treated_as_unavailable(monkeypatch):
    import i4h_common.port_lease as helper

    original_socket = socket.socket

    def denied(family=socket.AF_INET, *args, **kwargs):
        if family == socket.AF_INET6:
            raise PermissionError(errno.EACCES, "IPv6 inspection denied")
        return original_socket(family, *args, **kwargs)

    monkeypatch.setattr(helper.socket, "socket", denied)
    with pytest.raises(RuntimeError, match="Cannot inspect"):
        helper.listener_inodes(12345)
