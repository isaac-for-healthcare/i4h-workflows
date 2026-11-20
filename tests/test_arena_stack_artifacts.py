# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject missing or corrupt qualification artifacts before syncing an environment."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "arena_stack_artifacts", Path(__file__).resolve().parents[1] / "scripts/arena_stack_artifacts.py"
)
artifacts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(artifacts)


def test_missing_operator_wheel_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    monkeypatch.delenv("I4H_WARP_WHEEL", raising=False)
    with pytest.raises(FileNotFoundError, match="Set I4H_WARP_WHEEL"):
        artifacts.stage_wheel()


def test_bad_operator_wheel_does_not_replace_valid_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    destination = tmp_path / "third_party/wheels" / artifacts.WHEEL_NAME
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"valid staged wheel")
    supplied = tmp_path / "supplied.whl"
    supplied.write_bytes(b"corrupt")
    monkeypatch.setenv("I4H_WARP_WHEEL", str(supplied))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        artifacts.stage_wheel()
    assert destination.read_bytes() == b"valid staged wheel"


def test_wheel_staging_is_verified_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    supplied = tmp_path / "supplied.whl"
    supplied.write_bytes(b"patched native library")
    monkeypatch.setattr(artifacts, "WHEEL_SHA256", hashlib.sha256(supplied.read_bytes()).hexdigest())
    monkeypatch.setenv("I4H_WARP_WHEEL", str(supplied))
    destination = artifacts.stage_wheel()
    assert destination.read_bytes() == supplied.read_bytes()
    monkeypatch.delenv("I4H_WARP_WHEEL")
    assert artifacts.stage_wheel() == destination
    destination.write_bytes(b"corrupt staged wheel")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        artifacts.stage_wheel()


@pytest.fixture
def active_asset(tmp_path, monkeypatch):
    import json

    from i4h_common.paths import numerical_source

    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    arena = tmp_path / "arena"
    arena.mkdir()
    packages = ("physics", "isaaclab", "surgical", "assessment", "endoluminal")
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


def test_inconsistent_numerical_declarations_are_rejected(active_asset):
    from i4h_common.paths import numerical_source

    project = artifacts.ROOT / "arena/pyproject.toml"
    project.write_text(project.read_text().replace("active/numerical_sim/isaaclab", "retired/numerical_sim/isaaclab"))
    with pytest.raises(ValueError, match="sources disagree"):
        numerical_source(artifacts.ROOT)


def test_missing_active_manifest_is_rejected(active_asset):
    (active_asset / "docs/asset-manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        artifacts.verify_asset()
