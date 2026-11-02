# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configure the pinned XLA before importing OpenPI or initializing JAX."""

from __future__ import annotations

import ctypes
import logging
import os

logger = logging.getLogger("i4h_tasks.openpi_pi0.runtime")

#: Blackwell compute capabilities the XLA in the pinned jaxlib 0.5.3 can target
#: (``sm_100``, ``sm_101``, ``sm_120``). It compiles an unknown one, GB300's 10.3
#: or GB10's 12.1, as the nearest known target, and its Triton GEMM kernels then
#: fail in ptxas or LLVM and abort the backend (NVBug 6935977). Every earlier
#: architecture is known.
XLA_KNOWN_BLACKWELL = frozenset({(10, 0), (10, 1), (12, 0)})
TRITON_GEMM_FLAG = "--xla_gpu_enable_triton_gemm"


def cuda_compute_capability() -> tuple[int, int] | None:
    """Compute capability of CUDA device 0, asked of the driver without starting JAX."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), 0):
        return None
    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
    if cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device) or cuda.cuDeviceGetAttribute(
        ctypes.byref(minor), 76, device
    ):
        return None
    return major.value, minor.value


def configure_xla(capability: tuple[int, int] | None) -> None:
    """Route GEMMs through cuBLAS on a GPU the pinned XLA cannot build Triton kernels for.

    Must run before JAX creates its GPU backend, which reads ``XLA_FLAGS`` once.
    Only those GPUs pay for it: on an RTX PRO 6000 a warm PI0 inference takes
    0.05 s with Triton GEMMs and 0.36 s without. An ``XLA_FLAGS`` that already
    sets the option is left alone.
    """
    if capability is None or capability[0] < 10 or capability in XLA_KNOWN_BLACKWELL:
        return
    flags = os.environ.get("XLA_FLAGS", "")
    if TRITON_GEMM_FLAG in flags:
        return
    os.environ["XLA_FLAGS"] = f"{flags} {TRITON_GEMM_FLAG}=false".strip()
    logger.info(
        "compute capability %s.%s is newer than the pinned XLA targets; disabling its Triton GEMM fusions",
        *capability,
    )
