# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`inspect` reads the dataset's own declaration and warns about the one field
the GR00T loader silently depends on."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tools.tactile.cli import convert_command, inspect, main  # noqa: E402

FINGERTIPS = ("left_thumb", "left_index", "right_thumb")


def _dataset(tmp_path: Path, *, tactile_path: bool = True) -> Path:
    meta = tmp_path / "meta"
    meta.mkdir()
    features = {
        "observation.state": {"dtype": "float32", "shape": [58]},
        "observation.tactile.force": {"dtype": "float32", "shape": [60]},
        **{f"observation.tactile.{name}": {"dtype": "video", "shape": [240, 240, 1]} for name in FINGERTIPS},
    }
    info = {"codebase_version": "v2.1", "fps": 30, "total_episodes": 3, "total_frames": 900, "features": features}
    if tactile_path:
        info["tactile_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    modality = {
        "tactile": {f"{name}_deform": {"original_key": f"observation.tactile.{name}"} for name in FINGERTIPS},
        "tactile_force": {
            "force": {"original_key": "observation.tactile.force", "start": 0, "end": 60},
            "force_180": {"original_key": "observation.tactile_180.force"},
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality), encoding="utf-8")
    return tmp_path


def test_inspect_reads_the_declared_tactile_columns(tmp_path: Path) -> None:
    summary = inspect(_dataset(tmp_path))
    assert summary["fps"] == 30 and summary["episodes"] == 3
    assert summary["deform_keys"]["left_thumb_deform"] == "observation.tactile.left_thumb"
    assert summary["deform_shape"] == [240, 240, 1]
    assert summary["force_column"] == "observation.tactile.force"
    assert summary["force_shape"] == [60]
    assert summary["force_180_column"] == "observation.tactile_180.force"


def test_inspect_warns_when_the_loader_would_skip_the_maps(tmp_path: Path, capsys) -> None:
    assert main(["inspect", str(_dataset(tmp_path, tactile_path=False))]) == 0
    assert "no tactile_path" in capsys.readouterr().out


def test_a_dataset_without_tactile_is_reported_not_rejected(tmp_path: Path, capsys) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 30, "features": {}}), encoding="utf-8")
    assert main(["inspect", str(tmp_path)]) == 0
    assert "no `tactile` section" in capsys.readouterr().out


def test_convert_points_at_the_public_converter(tmp_path: Path) -> None:
    command = convert_command(tmp_path / "raw", tmp_path / "out", "lab/h2-pads", "Peel the pad")
    assert "convert_h2_sharpa_180_to_lerobot_v21" in command
    assert "--repo-id lab/h2-pads" in command
    assert "--tactile-model-profile bimanual" in command
