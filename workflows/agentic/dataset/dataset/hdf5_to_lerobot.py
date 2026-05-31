# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for converting an agentic Arena HDF5 recording into a LeRobot dataset."""

from __future__ import annotations

import argparse
import logging
import os

from common.utils import resolve_path
from dataset.converter import apply_env_defaults, convert_hdf5_to_lerobot, nonnegative_int, parse_split


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert an arena HDF5 recording to a LeRobot dataset.")
    parser.add_argument("--env", default=None, help="Environment id from config/environments/<env>.yaml for defaults.")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--repo-id", default=None, help="Local dataset name or HF repo id. Defaults to local/<env>.")
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated camera keys to pull from obs/ (default: auto-detect uint8 (T,H,W,3) arrays).",
    )
    parser.add_argument("--action-dim", type=nonnegative_int, default=None)
    parser.add_argument("--state-dim", type=nonnegative_int, default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Resize all camera frames to H W (defaults to the env YAML policy.image_size). "
        "Unset keeps each camera's native resolution.",
    )
    parser.add_argument(
        "--state-obs-key",
        default="joint_pos",
        help="Name of the per-step proprioceptive obs to write as observation.state.",
    )
    parser.add_argument("--action-names", default=None, help="Comma-separated names for the action vector.")
    parser.add_argument("--state-names", default=None, help="Comma-separated names for the state vector.")
    parser.add_argument(
        "--state-split",
        default=None,
        type=parse_split,
        help="modality.json state split, e.g. 'single_arm:0-5,gripper:5-6'.",
    )
    parser.add_argument(
        "--action-split",
        default=None,
        type=parse_split,
        help="modality.json action split (same syntax as --state-split).",
    )
    parser.add_argument(
        "--joint-space",
        choices=("raw", "remap"),
        default=None,
        help="Conversion applied before writing actions/state. Defaults to dataset.joint_space for --env.",
    )
    parser.add_argument(
        "--action-source-frame",
        choices=("absolute", "relative_to_home"),
        default=None,
        help="Frame of obs/actions in HDF5. Defaults to dataset.action_source_frame for --env.",
    )
    parser.add_argument(
        "--action-output-frame",
        choices=("absolute", "relative_to_home"),
        default=None,
        help="Frame to write to LeRobot action. Defaults to dataset.action_output_frame for --env.",
    )
    parser.add_argument(
        "--home-joint-pos-rad",
        default=None,
        help="Override the env's comma-separated home joint pose in radians for relative_to_home action conversion.",
    )
    parser.add_argument(
        "--skip-frames",
        type=nonnegative_int,
        default=None,
        help="Drop leading frames from each demo. Defaults to dataset.skip_frames for --env, else 0.",
    )
    parser.add_argument(
        "--video-codec",
        default=os.environ.get("AGENTIC_LEROBOT_VIDEO_CODEC", "h264"),
        help=(
            "LeRobot video codec. Default 'h264' is required for GR00T finetuning "
            "(decord cannot decode AV1). Pass 'libsvtav1' only for offline storage "
            "or visualization datasets that will not be trained on."
        ),
    )
    parser.add_argument(
        "--image-writer-processes",
        type=nonnegative_int,
        default=int(os.environ.get("AGENTIC_LEROBOT_IMAGE_WRITER_PROCESSES", "0")),
    )
    parser.add_argument(
        "--image-writer-threads",
        type=nonnegative_int,
        default=int(os.environ.get("AGENTIC_LEROBOT_IMAGE_WRITER_THREADS", "8")),
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_env_defaults(args, parser)
    args.hdf5_path = str(resolve_path(args.hdf5_path, args.env))
    convert_hdf5_to_lerobot(args)


if __name__ == "__main__":
    main()
