# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose Slang's CUDA device, which needs the CUDA Toolkit's NVRTC.

Standard library only: the launcher imports this before Kit boots.
"""

from __future__ import annotations

import ctypes
import functools
import glob
import os
import subprocess
import sys
from pathlib import Path

# Slang's CUDA device loads the CUDA Toolkit's unversioned libnvrtc.so; the versioned
# libnvrtc.so.N that pip wheels ship is not enough. slangpy owns the real search, so these
# locations only explain a failure: they never decide whether the CUDA device may be used.
_NVRTC_LIBRARY = "libnvrtc.so"
_CUDA_INSTALL_GLOB = "/usr/local/cuda*"
_SYSTEM_LIBRARY_GLOB = "/usr/lib/*-linux-gnu"
DEVICE_FAILURE = "Failed to create device"

NVRTC_HINT = (
    "install a CUDA Toolkit 12.x or 13.x no newer than the CUDA version nvidia-smi reports, "
    "so Slang can load its libnvrtc.so, or render with --fluoro-device vulkan; see TROUBLESHOOTING.md"
)

_PROBE = f"""
import slangpy
from i4h_arena.medical.slang_cuda import cuda_device_error
try:
    slangpy.create_device(slangpy.DeviceType.cuda).close()
except RuntimeError as error:
    if {DEVICE_FAILURE!r} not in str(error):
        raise
    print(cuda_device_error(error))
    raise SystemExit(1)
"""


@functools.cache
def find_cuda_nvrtc() -> str | None:
    """Return the first libnvrtc.so in the toolkit locations or on the loader path, or None."""
    roots = [os.environ["CUDA_PATH"]] if os.environ.get("CUDA_PATH") else []
    roots += sorted(glob.glob(_CUDA_INSTALL_GLOB))
    directories: list[str] = []
    for root in roots:
        directories += [f"{root}/lib64", f"{root}/lib", *sorted(glob.glob(f"{root}/targets/*/lib"))]
    directories += sorted(glob.glob(_SYSTEM_LIBRARY_GLOB))
    for directory in directories:
        candidate = Path(directory, _NVRTC_LIBRARY)
        if candidate.is_file():
            return str(candidate)
    try:
        ctypes.CDLL(_NVRTC_LIBRARY)
    except OSError:
        return None
    return _NVRTC_LIBRARY


def nvrtc_version(library: str) -> tuple[int, int] | None:
    """Return NVRTC's (major, minor) as reported by nvrtcVersion, or None if it cannot load."""
    try:
        nvrtc = ctypes.CDLL(library)
        major, minor = ctypes.c_int(), ctypes.c_int()
        if nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return major.value, minor.value


def describe_cuda_nvrtc() -> str:
    library = find_cuda_nvrtc()
    if library is None:
        return (
            "no libnvrtc.so was found under CUDA_PATH, /usr/local/cuda*, the system library "
            "directories or the loader path"
        )
    version = nvrtc_version(library)
    found = f"NVRTC {version[0]}.{version[1]}" if version is not None else "an NVRTC of unknown version"
    return f"the first libnvrtc.so found is {found} at {library}"


def cuda_device_error(error: BaseException) -> str:
    reason = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return f"Slang could not create a CUDA device ({reason}); {describe_cuda_nvrtc()}. To fix it, {NVRTC_HINT}"


def check_slang_cuda_device(timeout_s: float = 120.0) -> str | None:
    """Create a Slang CUDA device in a child process; return why it failed, or None if it works.

    The child keeps Slang's NVRTC out of this process: the toolkit's libnvrtc.so carries the same
    soname as the copy Kit ships, so loading it before Kit boots could change which one Kit binds.
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=timeout_s, env=env)
    if result.returncode == 0:
        return None
    output = result.stdout.strip().splitlines()
    return output[-1] if output else result.stderr.strip()[-2000:]
