#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect the Arena runtime and verify source-bound assets.

Warp comes from the package index. Native hashes are provenance, not install
requirements; the upstream captured-HashGrid defect is tracked in
https://github.com/NVIDIA/warp/issues/2426.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_SHA256 = "b9eeaa91038c261ad10364e522837be3225c9432ad3740b45941f0676abde91c"
sys.path.insert(0, str(ROOT / "common"))


def verify_hash(path: Path, expected: str) -> str:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, received {actual}")
    return actual


def native_identity() -> dict[str, str]:
    """Report the actual loaded library without requiring a locally built binary."""
    import warp as wp

    wp.init()
    native = Path(importlib.import_module("warp._src.context").runtime.core._name).resolve()
    return dict(
        warp_version=importlib.metadata.version("warp-lang"),
        native_library=str(native),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
    )


def verify_asset() -> None:
    from i4h_common.paths import numerical_source

    numerical = numerical_source(ROOT)
    relative = "numerical_sim/data/vessels/AortaWithVesselsStatic.usdc"
    manifest_path = numerical / "data/asset-manifest.json"
    if not manifest_path.is_file():
        manifest_path = numerical / "docs/asset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest if entry["path"] == relative]
    if len(entries) != 1 or entries[0]["sha256"] != ASSET_SHA256:
        raise ValueError("Pinned aorta manifest does not match the approved asset hash")
    asset = numerical / "data/vessels/AortaWithVesselsStatic.usdc"
    print(
        json.dumps(
            {"asset": str(asset), "sha256": verify_hash(asset, ASSET_SHA256), "I4H_DATA": str(numerical / "data")}
        )
    )


def verify_runtime(numerical_root: Path | None = None) -> None:
    from i4h_common.paths import package_source

    native = native_identity()
    versions = {}
    for package in ("i4h-physics", "i4h-isaaclab", "i4h-host", "i4h-surgical", "i4h-assessment", "i4h-endoluminal"):
        source = json.loads(importlib.metadata.distribution(package).read_text("direct_url.json"))
        from urllib.parse import unquote, urlparse

        installed = Path(unquote(urlparse(source["url"]).path)).resolve()
        declared = (
            (numerical_root / "numerical_sim" / package.removeprefix("i4h-"))
            if numerical_root
            else package_source(package, ROOT)
        )
        if installed != declared:
            raise ValueError(f"{package}: installed source {installed} differs from declared {declared}")
        versions[package] = {"source": str(installed)}
    for distribution, module_name, expected in (
        ("isaacsim", "isaacsim", "6.2.0.0"),
        ("torch", "torch", "2.11.0"),
        ("torchvision", "torchvision", "0.26.0"),
        ("warp-lang", "warp", "1.17.0"),
        ("mujoco-core", "mujoco", "4.0.1"),
        ("mujoco-warp", "mujoco_warp", "4.0.1"),
        ("newton", "newton", None),
    ):
        version = importlib.metadata.version(distribution)
        module = importlib.import_module(module_name)
        if expected is not None and version.split("+")[0] != expected:
            raise ValueError(f"{distribution}: expected {expected}, received {version}")
        versions[distribution] = {"version": version, "path": module.__file__}
    newton_source = json.loads(importlib.metadata.distribution("newton").read_text("direct_url.json"))
    if newton_source.get("vcs_info", {}).get("commit_id") != "b70ad54cfcaf7570907d3a3aad6c60acc7443c33":
        raise ValueError(f"Unexpected Newton source: {newton_source}")
    versions["newton"]["source"] = newton_source
    from i4h_isaaclab.newton_manager import host_api

    host = host_api().__name__
    if host != "i4h_isaaclab.lab30":
        raise ValueError(f"Expected i4h_isaaclab.lab30, received {host}")
    print(
        json.dumps(
            {"packages": versions, "host_api": host, **native},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("verify-asset", "verify-runtime", "verify-native"))
    parser.add_argument("--numerical-root", type=Path, help="Explicit candidate repository for isolated qualification")
    args = parser.parse_args()
    operation = args.operation
    if operation == "verify-asset":
        verify_asset()
    elif operation == "verify-native":
        print(json.dumps(native_identity()))
    else:
        verify_runtime(args.numerical_root.resolve() if args.numerical_root else None)


if __name__ == "__main__":
    main()
