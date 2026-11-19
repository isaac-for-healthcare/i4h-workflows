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
