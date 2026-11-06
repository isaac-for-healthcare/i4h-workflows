# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect and convert fingertip tactile recordings.

A tactile LeRobot dataset is an ordinary LeRobot v2.x dataset with, per frame,
one single-channel video feature per fingertip (``observation.tactile.<name>``)
and one numeric force column (``observation.tactile.force``), declared under
the ``tactile`` and ``tactile_force`` sections of ``meta/modality.json``. This
tool reads that declaration; nothing here depends on a policy stack.

``convert`` does not reimplement the capture-to-LeRobot conversion. The
converter that aligns the 180 Hz tactile stream with the 30 Hz policy rows is
maintained with the recording pipeline; this command prints the pointer and the
invocation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("i4h_tools.tactile")

#: Where the capture-to-LeRobot converter lives, branch included.
CONVERTER = "https://github.com/binliunls/unitree_lerobot/tree/haochens/feat-tactile-180hz"
CONVERTER_MODULE = "unitree_lerobot.utils.convert_h2_sharpa_180_to_lerobot_v21"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i4h-tactile", description="Inspect tactile recordings.")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect", help="summarise the tactile columns of a LeRobot dataset")
    inspect_cmd.add_argument("dataset", type=Path, help="LeRobot v2.x dataset root (the directory holding meta/)")

    convert_cmd = sub.add_parser("convert", help="print the capture-to-LeRobot conversion command")
    convert_cmd.add_argument("raw_dir", type=Path, help="recording directory produced by the capture pipeline")
    convert_cmd.add_argument("output_dir", type=Path)
    convert_cmd.add_argument("--repo-id", required=True, help="LeRobot dataset identifier")
    convert_cmd.add_argument("--task-description", default="", help="instruction stored with every frame")
    return parser


def inspect(dataset: Path) -> dict[str, Any]:
    """What the dataset declares about tactile data, as one dictionary."""
    meta = dataset / "meta"
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    modality_path = meta / "modality.json"
    modality = json.loads(modality_path.read_text(encoding="utf-8")) if modality_path.is_file() else {}
    features = info.get("features", {})
    deform = {name: entry.get("original_key", "") for name, entry in (modality.get("tactile") or {}).items()}
    force = modality.get("tactile_force") or {}
    force_column = (force.get("force") or {}).get("original_key", "")
    return {
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "codebase_version": info.get("codebase_version"),
        "deform_keys": deform,
        "deform_shape": (features.get(next(iter(deform.values()), ""), {}) or {}).get("shape"),
        "force_column": force_column,
        "force_shape": (features.get(force_column, {}) or {}).get("shape"),
        "force_180_column": (force.get("force_180") or {}).get("original_key", ""),
        # The GR00T loader only decodes the fingertip videos when info.json
        # names a tactile_path; without it they are silently skipped.
        "tactile_path": info.get("tactile_path"),
    }


def print_summary(dataset: Path, summary: dict[str, Any]) -> None:
    print(f"{dataset}: LeRobot {summary['codebase_version']}, {summary['episodes']} episodes, {summary['fps']} fps")
    if not summary["deform_keys"]:
        print("  no `tactile` section in meta/modality.json")
    else:
        print(f"  {len(summary['deform_keys'])} fingertip maps {summary['deform_shape']}:")
        for name, column in summary["deform_keys"].items():
            print(f"    {name:24s} <- {column}")
    if summary["force_column"]:
        print(f"  force {summary['force_shape']} <- {summary['force_column']}")
    if summary["force_180_column"]:
        print(f"  180 Hz force history <- {summary['force_180_column']}")
    if summary["deform_keys"] and not summary["tactile_path"]:
        print("  warning: info.json has no tactile_path; the GR00T loader will skip the fingertip maps")


def convert_command(raw_dir: Path, output_dir: Path, repo_id: str, task_description: str) -> str:
    argv = [
        "python",
        "-m",
        CONVERTER_MODULE,
        "--raw-dir",
        str(raw_dir),
        "--output-dir",
        str(output_dir),
        "--repo-id",
        repo_id,
        "--tactile-model-profile",
        "bimanual",
    ]
    if task_description:
        argv += ["--task-description", repr(task_description)]
    return " ".join(argv)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[i4h-tactile] %(message)s")
    if args.command == "inspect":
        print_summary(args.dataset, inspect(args.dataset))
        return 0
    print(f"# converter: {CONVERTER}")
    print(convert_command(args.raw_dir, args.output_dir, args.repo_id, args.task_description))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
