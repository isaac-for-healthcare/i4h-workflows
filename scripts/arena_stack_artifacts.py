#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage and verify the exact artifacts required by the Arena/Newton qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL_NAME = "warp_lang-1.17.0-py3-none-linux_x86_64.whl"
WHEEL_SHA256 = "dff334c1589510f4374807eb3fbdc85cfa5e20ab8a0e05548dc54e0b64990546"
NATIVE_SHA256 = "eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb"
ASSET_SHA256 = "b9eeaa91038c261ad10364e522837be3225c9432ad3740b45941f0676abde91c"
NUMERICAL = ROOT / "third_party/i4h-numerical-sim-d1fc95c/numerical_sim"


def verify_hash(path: Path, expected: str) -> str:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, received {actual}")
    return actual


def stage_wheel() -> Path:
    destination = ROOT / "third_party/wheels" / WHEEL_NAME
    supplied = os.environ.get("I4H_WARP_WHEEL")
    source = Path(supplied).expanduser().resolve() if supplied else destination
    if not source.is_file():
        raise FileNotFoundError(f"Set I4H_WARP_WHEEL to the operator-supplied patched {WHEEL_NAME}; missing {source}")
    verify_hash(source, WHEEL_SHA256)
    if source != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        verify_hash(temporary, WHEEL_SHA256)
        temporary.replace(destination)
    print(json.dumps({"wheel": str(destination), "sha256": verify_hash(destination, WHEEL_SHA256)}))
    return destination


def verify_asset() -> None:
    relative = "numerical_sim/data/vessels/AortaWithVesselsStatic.usdc"
    manifest = json.loads((NUMERICAL / "docs/asset-manifest.json").read_text())
    entries = [entry for entry in manifest if entry["path"] == relative]
    if len(entries) != 1 or entries[0]["sha256"] != ASSET_SHA256:
        raise ValueError("Pinned aorta manifest does not match the approved asset hash")
    asset = NUMERICAL / "data/vessels/AortaWithVesselsStatic.usdc"
    print(
        json.dumps(
            {"asset": str(asset), "sha256": verify_hash(asset, ASSET_SHA256), "I4H_DATA": str(NUMERICAL / "data")}
        )
    )


def verify_runtime() -> None:
    import warp as wp

    wp.init()
    # Check the actual dlopen handle, not a guessed library beside the Python module.
    native = Path(importlib.import_module("warp._src.context").runtime.core._name).resolve()
    verify_hash(native, NATIVE_SHA256)
    versions = {}
    for distribution, module_name, expected in (
        ("isaacsim", "isaacsim", "6.2.0.0"),
        ("torch", "torch", "2.11.0"),
        ("torchvision", "torchvision", "0.26.0"),
        ("warp-lang", "warp", "1.17.0"),
        ("mujoco", "mujoco", "3.17.0"),
        ("mujoco-warp", "mujoco_warp", "3.17.0"),
        ("newton", "newton", None),
    ):
        version = importlib.metadata.version(distribution)
        module = importlib.import_module(module_name)
        if expected is not None and version.split("+")[0] != expected:
            raise ValueError(f"{distribution}: expected {expected}, received {version}")
        versions[distribution] = {"version": version, "path": module.__file__}
    newton_source = json.loads(importlib.metadata.distribution("newton").read_text("direct_url.json"))
    if newton_source.get("vcs_info", {}).get("commit_id") != "d31603ff5e8389d1d6b2a5daef6ea2cb074c26ff":
        raise ValueError(f"Unexpected Newton source: {newton_source}")
    versions["newton"]["source"] = newton_source
    from i4h_isaaclab.newton_manager import host_api

    host = host_api().__name__
    if host != "i4h_isaaclab.lab30":
        raise ValueError(f"Expected i4h_isaaclab.lab30, received {host}")
    print(
        json.dumps(
            {"packages": versions, "host_api": host, "native_library": str(native), "native_sha256": NATIVE_SHA256},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage-wheel", "verify-asset", "verify-runtime", "verify-native"))
    operation = parser.parse_args().operation
    if operation == "stage-wheel":
        stage_wheel()
    elif operation == "verify-asset":
        verify_asset()
    elif operation == "verify-native":
        import warp as wp

        wp.init()
        native = Path(importlib.import_module("warp._src.context").runtime.core._name).resolve()
        print(json.dumps({"native_library": str(native), "sha256": verify_hash(native, NATIVE_SHA256)}))
    else:
        verify_runtime()


if __name__ == "__main__":
    main()
