# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-user Linux port leases and fail-closed listener ownership checks.

Lock files are shared across checkouts and never unlinked: replacing a locked
inode would allow a second owner. Descriptors are close-on-exec.
"""

from __future__ import annotations

import errno
import fcntl
import os
import socket
import stat
from collections.abc import Iterator
from pathlib import Path


class PortUnavailable(RuntimeError):
    """A managed lease or another socket already occupies the port."""


def _probe_port(family: int, port: int) -> bool:
    """Return False only for an unavailable IPv6 family; propagate other errors."""
    try:
        with socket.socket(family) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            probe.bind(("::" if family == socket.AF_INET6 else "0.0.0.0", port))
    except OSError as error:
        if family == socket.AF_INET6 and error.errno in (
            errno.EAFNOSUPPORT,
            errno.EPROTONOSUPPORT,
            errno.EADDRNOTAVAIL,
        ):
            return False
        raise
    return True


def listener_inodes(port: int) -> set[str]:
    """Read TCP tables for enabled families; inspection errors remain fatal."""
    try:
        tables = ("tcp", "tcp6") if _probe_port(socket.AF_INET6, 0) else ("tcp",)
        return {
            fields[9]
            for name in tables
            for line in Path(f"/proc/net/{name}").read_text().splitlines()[1:]
            if (fields := line.split())[3] == "0A" and int(fields[1].split(":")[1], 16) == port
        }
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError(f"Cannot inspect listeners on port {port}: {error}") from error


def _descendants(pid: int) -> Iterator[int]:
    """Yield the parent before walking children; disappearing processes are normal."""
    found = set()
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        yield current
        # Children can be created by any thread, not just the thread-group leader.
        try:
            for thread in (Path("/proc") / str(current) / "task").iterdir():
                try:
                    pending.extend(map(int, (thread / "children").read_text().split()))
                except FileNotFoundError:
                    continue  # a thread exited during inspection
        except FileNotFoundError:
            continue  # the process exited before or during task enumeration


def listener_owned(port: int, pid: int) -> bool:
    """False until listening; raise on a foreign listener or failed inspection."""
    listeners = listener_inodes(port)
    if not listeners:
        return False
    try:
        owned = set()
        inspection_errors = []
        for child in _descendants(pid):
            directory = Path("/proc") / str(child) / "fd"
            try:
                descriptors = list(directory.iterdir())
            except FileNotFoundError:
                continue
            except OSError as error:
                inspection_errors.append(error)
                continue
            for descriptor in descriptors:
                try:
                    target = descriptor.readlink().as_posix()
                except FileNotFoundError:
                    continue
                except OSError as error:
                    inspection_errors.append(error)
                    continue
                if target.startswith("socket:["):
                    owned.add(target[8:-1])
            # Kit may spawn a non-dumpable crash reporter. Once every listener
            # is proven owned, inspecting unrelated descendants is unnecessary.
            if listeners <= owned:
                return True
        if inspection_errors:
            raise inspection_errors[0]
        if not listeners <= owned:
            raise RuntimeError(f"Port {port} has a listener outside process {pid} and its descendants")
        return True
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot establish listener ownership on port {port}: {error}") from error


class PortLease:
    """Hold until all managed children have completed cleanup."""

    def __init__(self, port: int):
        self.port = port
        self._fd: int | None = None
        if not 1 <= port <= 65535:
            raise ValueError(f"Invalid port: {port}")
        directory = Path("/tmp") / f"i4h-port-leases-{os.getuid()}"
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError(f"Unsafe port lease directory: {directory}")
        fd = os.open(directory / f"{port}.lock", os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise PortUnavailable(f"Port {port} is reserved by another managed run") from error
            if listener_inodes(port):
                raise PortUnavailable(f"Port {port} already has a listener")
            for family in (socket.AF_INET, socket.AF_INET6):
                try:
                    _probe_port(family, port)
                except OSError as error:
                    raise PortUnavailable(f"Port {port} cannot be bound: {error}") from error
            self._fd = fd
        except BaseException:
            os.close(fd)
            raise

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> PortLease:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def reserve_port(preferred: int, *, automatic: bool = False) -> PortLease:
    for port in range(preferred, min(preferred + (50 if automatic else 1), 65536)):
        try:
            return PortLease(port)
        except PortUnavailable:
            if not automatic:
                raise
    raise PortUnavailable(f"No available port in the 50-port range starting at {preferred}")


if __name__ == "__main__":
    import sys

    try:
        ready = listener_owned(int(sys.argv[1]), int(sys.argv[2]))
    except RuntimeError as error:
        sys.exit(str(error))
    sys.exit(0 if ready else 1)
