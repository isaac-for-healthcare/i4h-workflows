# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accept index Warp builds while retaining source and asset integrity checks."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "arena_stack_artifacts", Path(__file__).resolve().parents[1] / "scripts/arena_stack_artifacts.py"
)
artifacts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(artifacts)


@pytest.mark.parametrize("payload", [b"stock Warp native library", b"custom Warp native library"])
def test_native_identity_reports_any_installed_build(tmp_path, monkeypatch, payload):
    import sys
    from types import SimpleNamespace

    native = tmp_path / "warp.so"
    native.write_bytes(payload)
    context = SimpleNamespace(runtime=SimpleNamespace(core=SimpleNamespace(_name=str(native))))
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(init=lambda: None))
    monkeypatch.setitem(sys.modules, "warp._src.context", context)
    monkeypatch.setattr(artifacts.importlib.metadata, "version", lambda name: "1.17.0")
    identity = artifacts.native_identity()
    assert identity["warp_version"] == "1.17.0"
    assert identity["native_library"] == str(native)
    assert identity["native_sha256"] == hashlib.sha256(payload).hexdigest()


def test_arena_warp_is_pinned_without_a_local_wheel_requirement():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "arena/pyproject.toml").read_text())
    assert "warp-lang==1.17.0" in project["project"]["dependencies"]
    assert "warp-lang" not in project["tool"]["uv"]["sources"]
    lock = tomllib.loads((root / "arena/uv.lock").read_text())
    warp = next(p for p in lock["package"] if p["name"] == "warp-lang")
    assert warp["version"] == "1.17.0" and "registry" in warp["source"]


@pytest.fixture
def active_asset(tmp_path, monkeypatch):
    import json

    from i4h_common.paths import numerical_source

    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    arena = tmp_path / "arena"
    arena.mkdir()
    packages = ("physics", "isaaclab", "host", "surgical", "assessment", "endoluminal")
    (arena / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        + "".join(f'i4h-{name} = {{ path = "../third_party/active/numerical_sim/{name}" }}\n' for name in packages)
    )
    numerical = numerical_source(tmp_path)
    (numerical / "docs").mkdir(parents=True)
    (numerical / "docs/asset-manifest.json").write_text(
        json.dumps(
            [{"path": "numerical_sim/data/vessels/AortaWithVesselsStatic.usdc", "sha256": artifacts.ASSET_SHA256}]
        )
    )
    return numerical


def test_missing_active_asset_never_falls_back(active_asset):
    with pytest.raises(FileNotFoundError):
        artifacts.verify_asset()


def test_changed_manifest_hash_fails(active_asset):
    manifest = active_asset / "docs/asset-manifest.json"
    manifest.write_text(manifest.read_text().replace(artifacts.ASSET_SHA256, "unapproved"))
    with pytest.raises(ValueError, match="approved asset hash"):
        artifacts.verify_asset()


def test_changed_asset_hash_fails(active_asset):
    asset = active_asset / "data/vessels/AortaWithVesselsStatic.usdc"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        artifacts.verify_asset()


@pytest.mark.parametrize("package", ["isaaclab", "host"])
def test_inconsistent_numerical_declarations_are_rejected(active_asset, package):
    from i4h_common.paths import numerical_source

    project = artifacts.ROOT / "arena/pyproject.toml"
    project.write_text(
        project.read_text().replace(f"active/numerical_sim/{package}", f"retired/numerical_sim/{package}")
    )
    with pytest.raises(ValueError, match="sources disagree"):
        numerical_source(artifacts.ROOT)


def test_missing_active_manifest_is_rejected(active_asset):
    (active_asset / "docs/asset-manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        artifacts.verify_asset()
