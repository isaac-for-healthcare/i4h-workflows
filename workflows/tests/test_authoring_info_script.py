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
    path = Path(__file__).parents[2] / "scripts" / name
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
    assert items["robot"]["catalog"]["embodiment"]["registry_name"] == "g1_wbc_joint"
    assert payload["manifest_capabilities"] == {
        "action_space": "joint_position",
        "cameras": ["head"],
        "control_hz": 30.0,
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
    assert 'assert "idle" in source.modes' in source
    assert 'if spec.embodiment == "none":' in source
    assert 'assert set(source.modes) == {"idle"}' not in source
    assert "assert spec.cameras == ()" not in source
    assert "assert spec.objects == ()" not in source


def test_blank_creator_requires_an_approved_product_specialty() -> None:
    module = _load_script("create_blank_environment.py")
    parser = module._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["future_room"])

    args = parser.parse_args(["future_room", "--specialty", "laparoscopic-robotics"])
    assert args.specialty == "laparoscopic-robotics"


def test_blank_creator_rejects_an_existing_id_in_another_specialty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("create_blank_environment.py")
    existing = tmp_path / "workflows" / "i4h_workflows" / "ultrasound-robotics" / "future_room.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("WORKFLOW = object()\n", encoding="utf-8")

    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_blank_environment.py",
            "future_room",
            "--specialty",
            "hospital-automation-robotics",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit, match="workflow id 'future_room' already exists") as caught:
        module.main()

    assert "workflows/i4h_workflows/ultrasound-robotics/future_room.py" in str(caught.value)
    assert not (tmp_path / "workflows/i4h_workflows/hospital-automation-robotics/future_room.py").exists()
    assert not (tmp_path / "arena/i4h_arena/scenes/future_room.py").exists()
