# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI argument helpers for MP4 video recording during simulation runs."""

import argparse


def add_video_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for saving camera streams to MP4 files."""
    parser.add_argument("--save_video", action="store_true", help="Save simulation camera streams to MP4 files")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./eval_videos",
        help="Directory for MP4 output when --save_video is set",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="Frame rate for saved MP4 files when --save_video is set",
    )
    parser.add_argument(
        "--video_env_id",
        type=int,
        default=0,
        help="Environment index to record when num_envs > 1 and --save_video is set",
    )
    return parser
