# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI argument helpers for WebRTC streaming and HTTP trigger functionality."""

import argparse


def add_webrtc_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for WebRTC camera streaming.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The parser with WebRTC arguments added.
    """
    parser.add_argument(
        "--webrtc_cam",
        action="store_true",
        help="Enable WebRTC livestream of a camera using a lightweight internal WebRTC server.",
    )
    parser.add_argument(
        "--webrtc_host",
        type=str,
        default="0.0.0.0",
        help="Host interface for the WebRTC server (only used with --webrtc_cam).",
    )
    parser.add_argument(
        "--webrtc_port",
        type=int,
        default=8080,
        help="Port for the WebRTC server (only used with --webrtc_cam).",
    )
    parser.add_argument(
        "--webrtc_fps",
        type=int,
        default=30,
        help="Target frame rate for the WebRTC stream (only used with --webrtc_cam).",
    )
    return parser


def add_trigger_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for the HTTP trigger server.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The parser with trigger arguments added.
    """
    parser.add_argument(
        "--trigger_port",
        type=int,
        default=8081,
        help="Port for the HTTP trigger server. POST to /trigger to start policy.",
    )
    parser.add_argument(
        "--trigger_host",
        type=str,
        default="0.0.0.0",
        help="Host interface for the HTTP trigger server.",
    )
    return parser
