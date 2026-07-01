#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a LeRobot v2/v3 dataset to Agentic/Cosmos-compatible HDF5.

Output layout (per demo):
    data/demo_<N>/
        obs/actions      (T, action_dim) float32
        obs/joint_pos    (T, state_dim) float32
        obs/<camera>     (T, H, W, 3) uint8 RGB

Usage:
    python lerobot_to_hdf5.py --dataset-path ../real_ur_10ep
    python lerobot_to_hdf5.py --dataset-path ../real_ur_10ep --output ../real_ur_10ep/real_ur_10ep.hdf5
    python lerobot_to_hdf5.py --dataset-path ../real_ur_10ep --episode-indices 0,1 --no-videos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq

from validator import (
    _load_json,
    _load_jsonl,
    _parquet_path,
    _stack_column,
    _video_path,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def _camera_mapping(info: dict[str, Any], modality: dict[str, Any] | None) -> dict[str, str]:
    """Map HDF5 camera keys (short) to LeRobot video feature keys."""
    if modality and isinstance(modality.get("video"), dict):
        mapping: dict[str, str] = {}
        for alias, spec in modality["video"].items():
            if isinstance(spec, dict) and "original_key" in spec:
                mapping[str(alias)] = str(spec["original_key"])
        if mapping:
            return mapping

    video_keys = [
        key
        for key, spec in info.get("features", {}).items()
        if isinstance(spec, dict) and spec.get("dtype") == "video" and key.startswith("observation.images.")
    ]
    return {key.removeprefix("observation.images."): key for key in video_keys}


def _read_video_rgb(path: Path, expected_frames: int | None = None) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required to decode videos (pip install -r requirements.txt)")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"video has no frames: {path}")
    if expected_frames is not None and len(frames) != expected_frames:
        raise ValueError(f"{path}: decoded {len(frames)} frames, expected {expected_frames}")

    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def _episode_filter(episodes: list[dict[str, Any]], indices: set[int] | None) -> list[dict[str, Any]]:
    if not indices:
        return episodes
    selected = [row for row in episodes if int(row["episode_index"]) in indices]
    missing = sorted(indices - {int(row["episode_index"]) for row in selected})
    if missing:
        raise ValueError(f"episode indices not found in episodes.jsonl: {missing}")
    return sorted(selected, key=lambda row: int(row["episode_index"]))


def _task_text(tasks: list[dict[str, Any]], task_index: int) -> str | None:
    for row in tasks:
        if int(row.get("task_index", -1)) == task_index:
            return str(row.get("task", ""))
    return None


def convert_lerobot_to_hdf5(
    dataset_path: Path,
    output_path: Path,
    *,
    episode_indices: set[int] | None = None,
    include_videos: bool = True,
    mark_success: bool = True,
    compression: str | None = "gzip",
) -> dict[str, Any]:
    root = dataset_path.resolve()
    meta = root / "meta"
    info = _load_json(meta / "info.json")
    episodes = _episode_filter(_load_jsonl(meta / "episodes.jsonl"), episode_indices)
    tasks = _load_jsonl(meta / "tasks.jsonl")

    modality_path = meta / "modality.json"
    modality = _load_json(modality_path) if modality_path.is_file() else None
    camera_map = _camera_mapping(info, modality)

    action_dim = int(info["features"]["action"]["shape"][0])
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    fps = float(info.get("fps", 30))

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset_path": str(root),
        "output_path": str(output_path.resolve()),
        "demos": [],
        "camera_keys": list(camera_map.keys()),
        "fps": fps,
        "robot_type": info.get("robot_type"),
    }

    compression_kwargs = {"compression": compression} if compression else {}

    with h5py.File(output_path, "w") as h5:
        h5.attrs["source_format"] = "lerobot"
        h5.attrs["source_dataset"] = str(root)
        h5.attrs["codebase_version"] = info.get("codebase_version", "")
        h5.attrs["robot_type"] = info.get("robot_type", "")
        h5.attrs["fps"] = fps

        data = h5.create_group("data")
        written = 0

        for row in episodes:
            episode_index = int(row["episode_index"])
            chunk_index = int(row.get("chunk_index", 0))
            length = int(row["length"])
            task_index = int(row.get("task_index", 0))

            parquet_path = root / _parquet_path(info, episode_index, chunk_index)
            table = pq.read_table(parquet_path)
            if table.num_rows != length:
                raise ValueError(
                    f"{parquet_path.name}: {table.num_rows} parquet rows, expected {length} from episodes.jsonl"
                )

            actions = _stack_column(table, "action").astype(np.float32, copy=False)
            states = _stack_column(table, "observation.state").astype(np.float32, copy=False)
            if actions.shape != (length, action_dim):
                raise ValueError(f"episode {episode_index}: action shape {actions.shape}, expected ({length}, {action_dim})")
            if states.shape != (length, state_dim):
                raise ValueError(f"episode {episode_index}: state shape {states.shape}, expected ({length}, {state_dim})")

            demo_name = f"demo_{written}"
            demo = data.create_group(demo_name)
            obs = demo.create_group("obs")

            obs.create_dataset("actions", data=actions, dtype=np.float32, **compression_kwargs)
            obs.create_dataset("joint_pos", data=states, dtype=np.float32, **compression_kwargs)

            camera_shapes: dict[str, list[int]] = {}
            if include_videos:
                for hdf5_cam, lerobot_cam in camera_map.items():
                    video_path = root / _video_path(info, lerobot_cam, episode_index, chunk_index)
                    frames = _read_video_rgb(video_path, expected_frames=length)
                    obs.create_dataset(hdf5_cam, data=frames, dtype=np.uint8, **compression_kwargs)
                    camera_shapes[hdf5_cam] = list(frames.shape)

            demo.attrs["episode_index"] = episode_index
            demo.attrs["chunk_index"] = chunk_index
            demo.attrs["task_index"] = task_index
            demo.attrs["length"] = length
            demo.attrs["source_lerobot_episode"] = f"episode_{episode_index:06d}"
            if mark_success:
                demo.attrs["success"] = True

            task_text = _task_text(tasks, task_index) or (row.get("tasks") or [None])[0]
            if task_text:
                demo.attrs["task_description"] = str(task_text)
            if row.get("tasks"):
                demo.attrs["tasks"] = json.dumps(row["tasks"])

            summary["demos"].append(
                {
                    "name": demo_name,
                    "episode_index": episode_index,
                    "length": length,
                    "cameras": camera_shapes,
                }
            )
            written += 1
            print(
                f"  wrote {demo_name} (episode {episode_index}, {length} steps, "
                f"{len(camera_shapes)} camera(s))",
                flush=True,
            )

        data.attrs["total"] = written

    summary["total_demos"] = written
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a LeRobot dataset to Agentic/Cosmos HDF5.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="LeRobot dataset root (meta/, data/, videos/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .hdf5 path (default: <dataset-path>/<dataset-name>.hdf5).",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Comma-separated episode indices to convert (default: all in episodes.jsonl).",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip video decode; write actions/state only (Cosmos will not work).",
    )
    parser.add_argument(
        "--no-success",
        action="store_true",
        help="Do not set demo attrs success=True.",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable gzip compression on HDF5 datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output file if it already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    dataset_path: Path = args.dataset_path.resolve()
    output_path = args.output or (dataset_path / f"{dataset_path.name}.hdf5")

    if output_path.exists() and not args.overwrite:
        print(f"Output exists: {output_path} (pass --overwrite to replace)", file=sys.stderr)
        return 1

    if not args.no_videos and cv2 is None:
        print("Install opencv-python-headless to decode videos, or pass --no-videos.", file=sys.stderr)
        return 1

    episode_indices: set[int] | None = None
    if args.episode_indices:
        episode_indices = {int(part.strip()) for part in args.episode_indices.split(",") if part.strip()}

    print(f"Converting {dataset_path} -> {output_path}")
    summary = convert_lerobot_to_hdf5(
        dataset_path,
        output_path,
        episode_indices=episode_indices,
        include_videos=not args.no_videos,
        mark_success=not args.no_success,
        compression=None if args.no_compression else "gzip",
    )
    print(
        f"Done: {summary['total_demos']} demo(s), "
        f"robot_type={summary.get('robot_type')}, cameras={summary['camera_keys']}"
    )
    print(f"HDF5: {summary['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
