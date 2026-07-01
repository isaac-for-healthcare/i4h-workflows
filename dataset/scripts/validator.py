#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate LeRobot v2/v3 datasets (structure, parquet, videos, GR00T metadata).

Usage:
    python validator.py --dataset-path ../real_ur_10ep
    python validator.py --dataset-path ../real_ur_10ep --check-videos --strict
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    import cv2
except ImportError:  # pragma: no cover - optional at import time, required for --check-videos
    cv2 = None  # type: ignore[assignment]


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Issue:
    severity: Severity
    code: str
    message: str
    episode: int | None = None


@dataclass
class ValidationReport:
    dataset_path: Path
    issues: list[Issue] = field(default_factory=list)

    def add(self, severity: Severity, code: str, message: str, episode: int | None = None) -> None:
        self.issues.append(Issue(severity, code, message, episode))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def ok(self, strict: bool = False) -> bool:
        if self.errors:
            return False
        if strict and self.warnings:
            return False
        return True


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _resolve_template(template: str, episode_index: int, chunk_index: int) -> str:
    return template.format(
        episode_index=episode_index,
        episode_chunk=chunk_index,
        video_key="{video_key}",
    )


def _video_path(info: dict[str, Any], video_key: str, episode_index: int, chunk_index: int) -> str:
    template = info["video_path"]
    return template.format(
        video_key=video_key,
        episode_index=episode_index,
        episode_chunk=chunk_index,
    )


def _parquet_path(info: dict[str, Any], episode_index: int, chunk_index: int) -> str:
    return info["data_path"].format(episode_index=episode_index, episode_chunk=chunk_index)


def _stack_column(table: pq.Table, name: str) -> np.ndarray:
    column = table[name]
    values = column.to_pylist()
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    first = values[0]
    if isinstance(first, (list, np.ndarray)):
        return np.stack([np.asarray(row, dtype=np.float32) for row in values])
    return np.asarray(values)


def _video_frame_count(path: Path) -> int | None:
    if cv2 is not None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count > 0:
                return count
        finally:
            capture.release()

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def _validate_modality(modality: dict[str, Any], action_dim: int, state_dim: int, report: ValidationReport) -> None:
    for group_name, dim in (("state", state_dim), ("action", action_dim)):
        entries = modality.get(group_name, {})
        if not isinstance(entries, dict):
            report.add(Severity.ERROR, "modality_shape", f"modality.{group_name} must be an object")
            continue
        for key, spec in entries.items():
            if not isinstance(spec, dict):
                continue
            start = spec.get("start")
            end = spec.get("end")
            if start is None or end is None:
                report.add(Severity.WARNING, "modality_slice", f"modality.{group_name}.{key} missing start/end")
                continue
            if not (0 <= start < end <= dim):
                report.add(
                    Severity.ERROR,
                    "modality_slice",
                    f"modality.{group_name}.{key} slice [{start}:{end}) out of range for dim {dim}",
                )

    video = modality.get("video", {})
    if isinstance(video, dict):
        for alias, spec in video.items():
            if not isinstance(spec, dict) or "original_key" not in spec:
                report.add(Severity.WARNING, "modality_video", f"modality.video.{alias} missing original_key")


def _validate_relative_stats(relative_stats: dict[str, Any], report: ValidationReport) -> None:
    for key, block in relative_stats.items():
        if not isinstance(block, dict):
            continue
        lengths: set[int] = set()
        for stat_name, values in block.items():
            if isinstance(values, list) and values and isinstance(values[0], list):
                lengths.add(len(values))
        if len(lengths) > 1:
            report.add(
                Severity.ERROR,
                "relative_stats_shape",
                f"relative_stats.{key} has inconsistent horizon lengths: {sorted(lengths)}",
            )


def validate_dataset(
    dataset_path: Path,
    *,
    check_videos: bool = False,
    max_action_state_delta: float | None = None,
) -> ValidationReport:
    report = ValidationReport(dataset_path=dataset_path.resolve())
    root = report.dataset_path

    if not root.is_dir():
        report.add(Severity.ERROR, "dataset_missing", f"Dataset directory not found: {root}")
        return report

    meta = root / "meta"
    info_path = meta / "info.json"
    episodes_path = meta / "episodes.jsonl"
    tasks_path = meta / "tasks.jsonl"

    for required in (info_path, episodes_path, tasks_path):
        if not required.is_file():
            report.add(Severity.ERROR, "meta_missing", f"Missing required file: {required.relative_to(root)}")

    if report.errors:
        return report

    info = _load_json(info_path)
    episodes = _load_jsonl(episodes_path)
    tasks = _load_jsonl(tasks_path)

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    fps = float(info.get("fps", 0))
    features = info.get("features", {})

    if total_episodes <= 0:
        report.add(Severity.ERROR, "info_episodes", "info.json: total_episodes must be > 0")
    if total_frames <= 0:
        report.add(Severity.ERROR, "info_frames", "info.json: total_frames must be > 0")
    if fps <= 0:
        report.add(Severity.WARNING, "info_fps", "info.json: fps is missing or non-positive")

    if len(episodes) != total_episodes:
        report.add(
            Severity.ERROR,
            "episode_count",
            f"episodes.jsonl has {len(episodes)} rows but info.json total_episodes={total_episodes}",
        )

    task_indices = {int(row["task_index"]) for row in tasks if "task_index" in row}
    if not task_indices:
        report.add(Severity.WARNING, "tasks_empty", "tasks.jsonl has no task_index entries")

    modality_path = meta / "modality.json"
    modality: dict[str, Any] | None = None
    if modality_path.is_file():
        modality = _load_json(modality_path)
    else:
        report.add(Severity.WARNING, "modality_missing", "meta/modality.json not found (required for GR00T)")

    stats_path = meta / "stats.json"
    if not stats_path.is_file():
        report.add(Severity.WARNING, "stats_missing", "meta/stats.json not found (recommended for GR00T)")

    relative_stats_path = meta / "relative_stats.json"
    if not relative_stats_path.is_file():
        report.add(Severity.WARNING, "relative_stats_missing", "meta/relative_stats.json not found (needed for relative actions)")
    else:
        _validate_relative_stats(_load_json(relative_stats_path), report)

    action_spec = features.get("action", {})
    state_spec = features.get("observation.state", {})
    action_dim = int(action_spec.get("shape", [0])[0]) if action_spec else 0
    state_dim = int(state_spec.get("shape", [0])[0]) if state_spec else 0

    if action_dim <= 0 or state_dim <= 0:
        report.add(Severity.ERROR, "feature_dims", "info.json: action and observation.state dims must be defined")
        return report

    if modality is not None:
        _validate_modality(modality, action_dim, state_dim, report)

    video_keys = [
        key
        for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") == "video" and key.startswith("observation.images.")
    ]

    frame_sum = 0
    seen_episode_indices: set[int] = set()

    for row in episodes:
        episode_index = int(row.get("episode_index", -1))
        chunk_index = int(row.get("chunk_index", 0))
        length = int(row.get("length", 0))
        task_index = int(row.get("task_index", -1))

        if episode_index in seen_episode_indices:
            report.add(Severity.ERROR, "duplicate_episode", f"Duplicate episode_index {episode_index}", episode_index)
        seen_episode_indices.add(episode_index)

        if length <= 0:
            report.add(Severity.ERROR, "episode_length", "Episode length must be > 0", episode_index)

        if task_index not in task_indices:
            report.add(
                Severity.WARNING,
                "task_index",
                f"task_index {task_index} not found in tasks.jsonl",
                episode_index,
            )

        parquet_rel = _parquet_path(info, episode_index, chunk_index)
        parquet_path = root / parquet_rel
        if not parquet_path.is_file():
            report.add(
                Severity.ERROR,
                "parquet_missing",
                f"Missing parquet: {parquet_rel}",
                episode_index,
            )
            continue

        table = pq.read_table(parquet_path)
        if table.num_rows != length:
            report.add(
                Severity.ERROR,
                "parquet_rows",
                f"{parquet_rel}: {table.num_rows} rows, expected {length} from episodes.jsonl",
                episode_index,
            )

        required_columns = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index"]
        for col in required_columns:
            if col not in table.column_names:
                report.add(Severity.ERROR, "parquet_column", f"{parquet_rel}: missing column '{col}'", episode_index)

        if "action" in table.column_names and "observation.state" in table.column_names:
            actions = _stack_column(table, "action")
            states = _stack_column(table, "observation.state")
            if actions.shape != (length, action_dim):
                report.add(
                    Severity.ERROR,
                    "action_shape",
                    f"{parquet_rel}: action shape {actions.shape}, expected ({length}, {action_dim})",
                    episode_index,
                )
            if states.shape != (length, state_dim):
                report.add(
                    Severity.ERROR,
                    "state_shape",
                    f"{parquet_rel}: state shape {states.shape}, expected ({length}, {state_dim})",
                    episode_index,
                )
            if max_action_state_delta is not None and actions.shape == states.shape and length > 0:
                delta = float(np.max(np.abs(actions - states)))
                if delta > max_action_state_delta:
                    report.add(
                        Severity.WARNING,
                        "action_state_delta",
                        f"{parquet_rel}: max |action-state|={delta:.6f} exceeds {max_action_state_delta}",
                        episode_index,
                    )

            ep_col = table["episode_index"].to_numpy()
            if not np.all(ep_col == episode_index):
                report.add(
                    Severity.ERROR,
                    "episode_index_mismatch",
                    f"{parquet_rel}: episode_index column does not match {episode_index}",
                    episode_index,
                )

            timestamps = table["timestamp"].to_numpy().astype(np.float64)
            if length > 1 and np.any(np.diff(timestamps) < 0):
                report.add(Severity.WARNING, "timestamp_order", f"{parquet_rel}: timestamps not monotonic", episode_index)

        for video_key in video_keys:
            video_rel = _video_path(info, video_key, episode_index, chunk_index)
            video_path = root / video_rel
            if not video_path.is_file():
                report.add(
                    Severity.ERROR,
                    "video_missing",
                    f"Missing video: {video_rel}",
                    episode_index,
                )
                continue

            if check_videos:
                if cv2 is None and not _ffprobe_available():
                    report.add(
                        Severity.ERROR,
                        "video_check_deps",
                        "Install opencv-python-headless or ffprobe to use --check-videos",
                    )
                    return report
                frame_count = _video_frame_count(video_path)
                if frame_count is None:
                    report.add(
                        Severity.ERROR,
                        "video_unreadable",
                        f"Could not read frame count: {video_rel}",
                        episode_index,
                    )
                elif frame_count != length:
                    report.add(
                        Severity.ERROR,
                        "video_frames",
                        f"{video_rel}: {frame_count} frames, expected {length} (parquet rows)",
                        episode_index,
                    )

                spec = features[video_key]
                expected_shape = spec.get("shape", [])
                if len(expected_shape) == 3 and cv2 is not None:
                    capture = cv2.VideoCapture(str(video_path))
                    ok, frame = capture.read()
                    capture.release()
                    if ok and frame is not None:
                        h, w = frame.shape[:2]
                        exp_h, exp_w = int(expected_shape[0]), int(expected_shape[1])
                        if (h, w) != (exp_h, exp_w):
                            report.add(
                                Severity.WARNING,
                                "video_resolution",
                                f"{video_rel}: resolution {w}x{h}, info.json expects {exp_w}x{exp_h}",
                                episode_index,
                            )

        frame_sum += length

    if frame_sum != total_frames:
        report.add(
            Severity.ERROR,
            "total_frames",
            f"Sum of episode lengths ({frame_sum}) != info.json total_frames ({total_frames})",
        )

    report.add(
        Severity.INFO,
        "summary",
        f"Validated {len(episodes)} episodes, {frame_sum} frames, {len(video_keys)} video streams, robot_type={info.get('robot_type', '?')}",
    )
    return report


def _ffprobe_available() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _print_report(report: ValidationReport, verbose: bool) -> None:
    for issue in report.issues:
        if not verbose and issue.severity == Severity.INFO and issue.code == "summary":
            print(f"[{issue.severity}] {issue.message}")
            continue
        if not verbose and issue.severity == Severity.INFO:
            continue
        prefix = f"[{issue.severity}]"
        ep = f" (episode {issue.episode})" if issue.episode is not None else ""
        print(f"{prefix} {issue.code}{ep}: {issue.message}")

    print()
    print(
        f"Result: {len(report.errors)} error(s), {len(report.warnings)} warning(s) "
        f"— {'PASS' if not report.errors else 'FAIL'}"
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a LeRobot dataset directory.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset root (contains meta/, data/, videos/).",
    )
    parser.add_argument(
        "--check-videos",
        action="store_true",
        help="Decode videos and verify frame counts match parquet row counts.",
    )
    parser.add_argument(
        "--max-action-state-delta",
        type=float,
        default=None,
        help="Warn if max |action - state| per episode exceeds this threshold.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (exit code 1).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print INFO-level messages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    report = validate_dataset(
        args.dataset_path,
        check_videos=args.check_videos,
        max_action_state_delta=args.max_action_state_delta,
    )
    _print_report(report, verbose=args.verbose)
    return 0 if report.ok(strict=args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
