# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

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
    assert args.patient_twin == ["/tmp/patient_twin.yaml"]
    assert args.view_sensor == ["fluoroscopy"]


def test_patient_twin_repeats_once_per_environment() -> None:
    args = build_parser().parse_args(
        ["--workflow", "endoluminal_navigation", "--envs", "2", "--patient-twin", "a.yaml", "--patient-twin", "b.yaml"]
    )

    assert args.patient_twin == ["a.yaml", "b.yaml"]


def test_fluoroscopy_backend_is_automatic_by_default() -> None:
    args = build_parser().parse_args(["--workflow", "endoluminal_navigation"])

    assert args.fluoro_backend is None


def test_fluoroscopy_backend_follows_patient_twin() -> None:
    assert resolve_fluoroscopy_backend(None, None) == "synthetic"
    assert resolve_fluoroscopy_backend(None, "/tmp/patient_twin.yaml") == "slang"
    assert resolve_fluoroscopy_backend("synthetic", "/tmp/patient_twin.yaml") == "synthetic"


def _fluoroscopy_args(*extra: str):
    return build_parser().parse_args(["--workflow", "endoluminal_navigation", *extra])


def _device_check(monkeypatch, result: str | None) -> list[int]:
    from i4h_arena.medical import slang_cuda

    calls: list[int] = []
    monkeypatch.setattr(slang_cuda, "check_slang_cuda_device", lambda: calls.append(1) or result)
    return calls


def test_fluoroscopy_device_is_automatic_by_default() -> None:
    assert _fluoroscopy_args().fluoro_device == "auto"


def test_a_cuda_fluoroscopy_device_slang_cannot_create_stops_the_run_before_kit(monkeypatch) -> None:
    from i4h_arena.cli import _resolve_fluoroscopy_device

    _device_check(monkeypatch, "Slang could not create a CUDA device (Failed to create device!)")

    with pytest.raises(SystemExit, match=r"--fluoro-device cuda: Slang could not create a CUDA device"):
        _resolve_fluoroscopy_device(_fluoroscopy_args("--patient-twin", "/tmp/twin.yaml", "--fluoro-device", "cuda"))


def test_a_cuda_fluoroscopy_device_slang_can_create_starts_normally(monkeypatch) -> None:
    from i4h_arena.cli import _resolve_fluoroscopy_device

    calls = _device_check(monkeypatch, None)
    args = _fluoroscopy_args("--fluoro-backend", "slang", "--fluoro-device", "cuda")

    _resolve_fluoroscopy_device(args)
    assert calls == [1]
    assert args.fluoro_device == "cuda"


def test_automatic_device_renders_patient_fluoroscopy_on_cuda_when_slang_can_create_it(monkeypatch) -> None:
    from i4h_arena.cli import _resolve_fluoroscopy_device

    calls = _device_check(monkeypatch, None)
    args = _fluoroscopy_args("--patient-twin", "/tmp/twin.yaml")

    _resolve_fluoroscopy_device(args)
    assert calls == [1]
    assert args.fluoro_device == "cuda"


def test_automatic_device_falls_back_to_vulkan_and_says_why(monkeypatch, capsys) -> None:
    from i4h_arena.cli import _resolve_fluoroscopy_device

    _device_check(monkeypatch, "Slang could not create a CUDA device (Failed to create device!)")
    args = _fluoroscopy_args("--patient-twin", "/tmp/twin.yaml")

    _resolve_fluoroscopy_device(args)
    assert args.fluoro_device == "vulkan"
    assert "Slang could not create a CUDA device" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra, expected",
    [
        ((), "vulkan"),
        (("--fluoro-device", "cuda"), "cuda"),
        (("--patient-twin", "/tmp/twin.yaml", "--fluoro-device", "vulkan"), "vulkan"),
        (("--patient-twin", "/tmp/twin.yaml", "--fluoro-backend", "synthetic"), "vulkan"),
        (("--patient-twin", "/tmp/twin.yaml", "--fluoro-backend", "synthetic", "--fluoro-device", "cuda"), "cuda"),
    ],
)
def test_the_cuda_device_is_only_checked_when_slang_may_render_on_it(monkeypatch, extra, expected) -> None:
    from i4h_arena.cli import _resolve_fluoroscopy_device

    calls = _device_check(monkeypatch, "unused")
    args = _fluoroscopy_args(*extra)

    _resolve_fluoroscopy_device(args)
    assert calls == []
    assert args.fluoro_device == expected
