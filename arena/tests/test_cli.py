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


def test_vessel_contrast_is_opt_in() -> None:
    assert _fluoroscopy_args().vessel_contrast is False
    assert _fluoroscopy_args("--vessel-contrast").vessel_contrast is True


def test_vessel_contrast_without_a_patient_stops_the_run_before_kit() -> None:
    from i4h_arena.cli import _resolve_fluoroscopy_device

    with pytest.raises(SystemExit, match="--vessel-contrast needs the Slang fluoroscopy backend"):
        _resolve_fluoroscopy_device(_fluoroscopy_args("--vessel-contrast"))


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


@pytest.mark.parametrize("value", ["-1", "-500"])
def test_a_negative_episode_step_cap_is_refused(value, capsys) -> None:
    """NVBug 7042164: only the upper bound was checked, so -1 reached the runner."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--workflow", "example", "--episode-steps", value])

    assert exit_info.value.code == 2
    assert "must be 0 (the scene default) or a positive cap" in capsys.readouterr().err


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("25", 25)])
def test_a_zero_or_positive_episode_step_cap_parses(value, expected) -> None:
    assert build_parser().parse_args(["--workflow", "example", "--episode-steps", value]).episode_steps == expected


def _view_sensor_args(*extra: str):
    return build_parser().parse_args(["--workflow", "example", *extra])


@pytest.mark.parametrize("flag", [None, "--headless", "--no-sensor-view", "--no-cameras"])
def test_an_undeclared_view_sensor_is_refused_whatever_the_display_flags(flag, capsys) -> None:
    """NVBug 7035806: with --headless (or no windows) an unknown name was dropped silently."""
    from i4h_arena.cli import _check_view_sensors

    parser = build_parser()
    args = _view_sensor_args("--view-sensor", "nosuch", *([flag] if flag else []))
    with pytest.raises(SystemExit) as exit_info:
        _check_view_sensors(parser, args, ("room", "wrist"))

    assert exit_info.value.code == 2
    assert (
        "cannot view undeclared scene sensor(s) ['nosuch']; available camera-compatible sensors: ['room', 'wrist']"
        in (capsys.readouterr().err)
    )


def test_a_declared_view_sensor_without_windows_is_reported_as_ignored(capsys) -> None:
    from i4h_arena.cli import _check_view_sensors

    args = _view_sensor_args("--view-sensor", "room", "--headless")
    _check_view_sensors(build_parser(), args, ("room", "wrist"))

    assert "--view-sensor is ignored: --headless opens no sensor windows" in capsys.readouterr().out


def test_a_declared_view_sensor_with_windows_is_accepted_quietly(capsys) -> None:
    from i4h_arena.cli import _check_view_sensors

    _check_view_sensors(build_parser(), _view_sensor_args("--view-sensor", "room"), ("room", "wrist"))

    assert capsys.readouterr().out == ""


def test_a_missing_replay_recording_is_named_before_kit_boots(tmp_path) -> None:
    """NVBug 7042243: the documented message, before the simulator starts."""
    from types import SimpleNamespace

    from i4h_arena.cli import _check_replay_dataset

    missing = tmp_path / "missing.hdf5"
    args = build_parser().parse_args(["--workflow", "example", "--dataset", str(missing)])
    with pytest.raises(FileNotFoundError, match=f"no recording at {missing}"):
        _check_replay_dataset(args, SimpleNamespace(mode="replay"))

    _check_replay_dataset(args, SimpleNamespace(mode="rule_based"))
    missing.write_bytes(b"")
    _check_replay_dataset(args, SimpleNamespace(mode="replay"))


def test_a_second_python_server_is_refused_while_the_port_is_taken(monkeypatch) -> None:
    """NVBug 7067702: a second --live opened another bridge on the same port."""
    import socket

    import i4h_arena.cli as arena_cli

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    monkeypatch.setattr(arena_cli, "PYTHON_SERVER_PORT", port)
    args = build_parser().parse_args(["--workflow", "example", "--python-server"])
    try:
        with pytest.raises(SystemExit, match=f"127.0.0.1:{port} is already in use"):
            arena_cli._check_python_server_port(args)
    finally:
        listener.close()

    arena_cli._check_python_server_port(args)
    arena_cli._check_python_server_port(build_parser().parse_args(["--workflow", "example"]))


def test_overlapping_python_server_launches_are_refused_before_either_listens(monkeypatch, tmp_path) -> None:
    """Greptile on #302: the bridge listens only after Kit boots, so a port check alone lets two starts through."""
    import i4h_arena.cli as arena_cli

    monkeypatch.setattr(arena_cli, "_python_server_lock_path", lambda: tmp_path / "bridge.lock")
    # The lock is under test, not the socket: a live session on the real port must not matter here.
    monkeypatch.setattr(arena_cli, "_check_python_server_port", lambda _args: None)
    args = build_parser().parse_args(["--workflow", "example", "--python-server"])

    # The first session is "booting": nothing listens on the port yet, so a port check alone would pass.
    with (
        arena_cli._python_server_guard(args),
        pytest.raises(SystemExit, match="another live session is starting or running"),
        arena_cli._python_server_guard(args),
    ):
        pytest.fail("the second launch must not get past the guard")


def test_the_python_server_lock_is_released_when_the_session_ends_or_fails(monkeypatch, tmp_path) -> None:
    import i4h_arena.cli as arena_cli

    monkeypatch.setattr(arena_cli, "_python_server_lock_path", lambda: tmp_path / "bridge.lock")
    monkeypatch.setattr(arena_cli, "_check_python_server_port", lambda _args: None)
    args = build_parser().parse_args(["--workflow", "example", "--python-server"])

    with arena_cli._python_server_guard(args):
        pass
    with pytest.raises(RuntimeError, match="boom"), arena_cli._python_server_guard(args):
        raise RuntimeError("boom")
    with arena_cli._python_server_guard(args):
        pass


def test_a_refused_port_check_does_not_keep_the_python_server_lock(monkeypatch, tmp_path) -> None:
    import socket

    import i4h_arena.cli as arena_cli

    monkeypatch.setattr(arena_cli, "_python_server_lock_path", lambda: tmp_path / "bridge.lock")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    monkeypatch.setattr(arena_cli, "PYTHON_SERVER_PORT", listener.getsockname()[1])
    args = build_parser().parse_args(["--workflow", "example", "--python-server"])
    try:
        with pytest.raises(SystemExit, match="is already in use"), arena_cli._python_server_guard(args):
            pytest.fail("an occupied port must stop the launch")
    finally:
        listener.close()
    with arena_cli._python_server_guard(args):
        pass


def test_the_python_server_guard_is_inert_without_the_flag(monkeypatch, tmp_path) -> None:
    import i4h_arena.cli as arena_cli

    lock = tmp_path / "bridge.lock"
    monkeypatch.setattr(arena_cli, "_python_server_lock_path", lambda: lock)
    with arena_cli._python_server_guard(build_parser().parse_args(["--workflow", "example"])):
        assert not lock.exists()


def test_the_python_server_lock_lives_in_a_private_directory(monkeypatch, tmp_path) -> None:
    import os
    import stat

    import i4h_arena.cli as arena_cli

    monkeypatch.setattr(arena_cli.tempfile, "gettempdir", lambda: str(tmp_path))
    path = arena_cli._python_server_lock_path()
    assert path.parent == tmp_path / f"i4h-arena-{os.getuid()}"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert arena_cli._python_server_lock_path() == path

    path.parent.chmod(0o755)
    with pytest.raises(SystemExit, match="mode 0700"):
        arena_cli._python_server_lock_path()


def test_a_planted_lock_directory_or_file_is_refused(monkeypatch, tmp_path) -> None:
    import os

    import i4h_arena.cli as arena_cli

    monkeypatch.setattr(arena_cli.tempfile, "gettempdir", lambda: str(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (tmp_path / f"i4h-arena-{os.getuid()}").symlink_to(elsewhere)
    with pytest.raises(SystemExit, match="mode 0700"):
        arena_cli._python_server_lock_path()

    fifo = tmp_path / "fifo.lock"
    os.mkfifo(fifo)
    with pytest.raises(SystemExit, match="not a regular lock file"):
        arena_cli._open_python_server_lock(fifo)  # returns at once instead of hanging on the FIFO

    link = tmp_path / "link.lock"
    link.symlink_to(tmp_path / "target.lock")
    with pytest.raises(OSError):  # O_NOFOLLOW: never creates or locks through a link
        arena_cli._open_python_server_lock(link)
