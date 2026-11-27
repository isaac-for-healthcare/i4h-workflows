#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Resolve Slang before Kit, then execute the unchanged native simulation runner."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


def resolve_device(requested: str) -> str:
    from i4h_arena.medical.slang_cuda import check_slang_cuda_device, prepare_slang_for_kit

    resolved = requested
    if requested in ("auto", "cuda"):
        reason = check_slang_cuda_device()
        if reason is not None:
            if requested == "cuda":
                raise RuntimeError(f"CUDA fluoroscopy preflight failed: {reason}. Try --fluoro-device vulkan.")
            print(f"[fluoroscopy] CUDA unavailable: {reason}; falling back to Vulkan", flush=True)
            resolved = "vulkan"
        else:
            resolved = "cuda"
    prepare_slang_for_kit(resolved)
    print(f"[fluoroscopy] resolved device: {resolved}", flush=True)
    return resolved


def main() -> None:
    resolved = resolve_device(os.environ["I4H_FLUORO_DEVICE_REQUEST"])
    metadata = os.environ.get("I4H_RECORDING_METADATA")
    if metadata:
        # Use the launcher's same lock and atomic publication discipline.
        import fcntl
        import tempfile

        path = Path(metadata)
        with (path.parent / ".run-metadata.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            payload = json.loads(path.read_text())
            if payload.get("fluoro_device", resolved) != resolved:
                raise ValueError("recording fluoroscopy device conflicts with resolved device")
            payload["fluoro_device"] = resolved
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            temporary.replace(path)
    target = sys.argv[1:]
    sys.argv = [*target, "--fluoro_device", resolved]
    if target[0] == "-m":
        sys.argv = [target[1], *target[2:], "--fluoro_device", resolved]
        runpy.run_module(target[1], run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(target[0], run_name="__main__")


if __name__ == "__main__":
    main()
