# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_arena.cli import build_parser
from i4h_arena.scenes.endoluminal_navigation import resolve_fluoroscopy_backend


def test_python_server_is_explicit_opt_in() -> None:
    parser = build_parser()

    assert parser.parse_args(["--workflow", "example"]).python_server is False
    assert parser.parse_args(["--workflow", "example", "--python-server"]).python_server is True


def test_idle_duration_can_be_extended_for_live_authoring() -> None:
    args = build_parser().parse_args(["--workflow", "example", "--mode", "idle", "--idle-seconds", "3600"])

    assert args.idle_seconds == 3600


def test_patient_fluoroscopy_options_are_parsed() -> None:
    args = build_parser().parse_args(
        [
            "--workflow",
            "endoluminal_navigation",
            "--fluoro-backend",
            "slang",
            "--fluoro-device",
            "vulkan",
            "--patient-twin",
            "/tmp/patient_twin.yaml",
            "--view-sensor",
            "fluoroscopy",
        ]
    )

    assert args.fluoro_backend == "slang"
    assert args.fluoro_device == "vulkan"
    assert args.patient_twin == "/tmp/patient_twin.yaml"
    assert args.view_sensor == ["fluoroscopy"]


def test_fluoroscopy_backend_is_automatic_by_default() -> None:
    args = build_parser().parse_args(["--workflow", "endoluminal_navigation"])

    assert args.fluoro_backend is None


def test_fluoroscopy_backend_follows_patient_twin() -> None:
    assert resolve_fluoroscopy_backend(None, None) == "synthetic"
    assert resolve_fluoroscopy_backend(None, "/tmp/patient_twin.yaml") == "slang"
    assert resolve_fluoroscopy_backend("synthetic", "/tmp/patient_twin.yaml") == "synthetic"
