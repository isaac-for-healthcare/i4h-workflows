# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "workflow": "fast_room",
        "environment_root": "/World/CustomEnv",
        "items": [
            {
                "kind": "known_asset",
                "name": "table",
                "preset": "surgical_table",
                "relative_prim_path": "Table",
                "position_m": [0.0, 0.0, 0.237744],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "scale": [0.7, 0.7, 0.52],
            },
            {
                "kind": "known_asset",
                "name": "robot",
                "preset": "g1",
                "relative_prim_path": "Robot",
                "position_m": [-2.64008, 0.0, 0.792273],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
        ],
    }


def test_snapshot_info_resolves_catalog_and_manifest_facts(tmp_path: Path, capsys) -> None:
    module = _load_script("authoring_info.py")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(_snapshot()), encoding="utf-8")

    assert module.main(["snapshot", "fast_room", str(snapshot)]) == 0

    payload = json.loads(capsys.readouterr().out)
    items = {item["name"]: item for item in payload["snapshot"]["items"]}
    assert payload["snapshot"]["environment_root"] == "/World/CustomEnv"
    assert items["robot"]["catalog"]["embodiment"]["registry_name"] == "g1_locomanip_wbc_joint"
    assert payload["manifest_capabilities"] == {
        "action_space": "joint_position",
        "cameras": ["head"],
        "control_hz": 50.0,
        "dof": 50,
        "embodiment": "g1",
        "gripper": False,
        "objects": ["table"],
        "robots": ["robot"],
    }


def test_snapshot_validation_rejects_noncanonical_robot_runtime_path(tmp_path: Path) -> None:
    module = _load_script("authoring_info.py")
    raw = _snapshot()
    raw["items"][1]["relative_prim_path"] = "PreviewRobot"
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(raw), encoding="utf-8")

    try:
        module.main(["snapshot", "fast_room", str(snapshot)])
    except ValueError as exc:
        assert "must use runtime path 'Robot'" in str(exc)
    else:
        raise AssertionError("expected mismatched robot path to be rejected")


def test_asset_info_returns_catalog_facts_without_simulator(capsys) -> None:
    module = _load_script("authoring_info.py")

    assert module.main(["asset", "scissors"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "scissors"
    assert payload["physics"] == "rigid"
    assert payload["mass_kg"] == 0.15
    assert payload["canonical_size_m"] == [0.123452, 0.034816, 0.023982]


def test_blank_creator_contract_remains_valid_after_incremental_authoring() -> None:
    module = _load_script("create_blank_environment.py")

    source = module._contract_test("future_room")

    compile(source, "test_future_room_contract.py", "exec")
    assert 'if spec.embodiment == "none":' in source
    assert "assert spec.cameras == ()" not in source
    assert "assert spec.objects == ()" not in source


def test_blank_creator_rejects_an_existing_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("create_blank_environment.py")
    existing = tmp_path / "arena" / "i4h_arena" / "scenes" / "manifest" / "future_room.yaml"
    existing.parent.mkdir(parents=True)
    existing.write_text("impl: a:B\n", encoding="utf-8")

    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["create_blank_environment.py", "future_room", "--dry-run"])

    with pytest.raises(SystemExit, match="scene 'future_room' already exists"):
        module.main()

    assert not (tmp_path / "arena/i4h_arena/scenes/future_room.py").exists()


def test_blank_creator_lists_the_scene_files_without_a_workflow_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    module = _load_script("create_blank_environment.py")
    (tmp_path / "arena" / "i4h_arena" / "scenes" / "manifest").mkdir(parents=True)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["create_blank_environment.py", "future_room", "--dry-run"])

    assert module.main() == 0

    listed = capsys.readouterr().out
    assert "arena/i4h_arena/scenes/manifest/future_room.yaml" in listed
    assert "tests/test_future_room_contract.py" in listed
    assert "workflows/" not in listed
