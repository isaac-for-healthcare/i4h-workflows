# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys

import pytest

from i4h_arena.medical import slang_cuda


@pytest.fixture(autouse=True)
def _no_system_nvrtc(monkeypatch, tmp_path):
    """Point every search location at an empty directory and make the loader fail."""
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(slang_cuda, "_CUDA_INSTALL_GLOB", str(tmp_path / "no-cuda*"))
    monkeypatch.setattr(slang_cuda, "_SYSTEM_LIBRARY_GLOB", str(tmp_path / "no-lib*"))

    def missing(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(slang_cuda.ctypes, "CDLL", missing)
    slang_cuda.find_cuda_nvrtc.cache_clear()
    yield
    slang_cuda.find_cuda_nvrtc.cache_clear()


def _library(root, relative: str = "lib64"):
    library = root / relative / "libnvrtc.so"
    library.parent.mkdir(parents=True)
    library.touch()
    return library


@pytest.mark.parametrize(
    "layout",
    ["cuda-13.0/lib64", "cuda-13.0/targets/sbsa-linux/lib", "cuda-13.0/targets/x86_64-linux/lib"],
)
def test_nvrtc_is_found_in_an_installed_toolkit(tmp_path, monkeypatch, layout) -> None:
    monkeypatch.setattr(slang_cuda, "_CUDA_INSTALL_GLOB", str(tmp_path / "cuda*"))
    library = _library(tmp_path, layout)

    assert slang_cuda.find_cuda_nvrtc() == str(library)


def test_nvrtc_is_found_in_the_distro_package_layout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(slang_cuda, "_SYSTEM_LIBRARY_GLOB", str(tmp_path / "usr-lib" / "*-linux-gnu"))
    library = _library(tmp_path / "usr-lib", "aarch64-linux-gnu")

    assert slang_cuda.find_cuda_nvrtc() == str(library)


def test_cuda_path_takes_precedence_over_installed_toolkits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(slang_cuda, "_CUDA_INSTALL_GLOB", str(tmp_path / "cuda*"))
    _library(tmp_path / "cuda-12.1")
    chosen = _library(tmp_path / "toolkit")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "toolkit"))

    assert slang_cuda.find_cuda_nvrtc() == str(chosen)


def test_an_empty_cuda_path_falls_through_to_installed_toolkits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(slang_cuda, "_CUDA_INSTALL_GLOB", str(tmp_path / "cuda*"))
    library = _library(tmp_path / "cuda-13.0")
    monkeypatch.setenv("CUDA_PATH", "")

    assert slang_cuda.find_cuda_nvrtc() == str(library)


def test_only_a_versioned_wheel_nvrtc_is_not_found(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(slang_cuda, "_CUDA_INSTALL_GLOB", str(tmp_path / "cuda*"))
    wheel = tmp_path / "cuda-wheel" / "lib" / "libnvrtc.so.13"
    wheel.parent.mkdir(parents=True)
    wheel.touch()

    assert slang_cuda.find_cuda_nvrtc() is None


def test_nvrtc_on_the_loader_path_is_found(monkeypatch) -> None:
    loaded = []
    monkeypatch.setattr(slang_cuda.ctypes, "CDLL", loaded.append)

    assert slang_cuda.find_cuda_nvrtc() == "libnvrtc.so"
    assert loaded == ["libnvrtc.so"]


def test_the_search_runs_once_per_process(tmp_path, monkeypatch) -> None:
    searches = []
    real_glob = slang_cuda.glob.glob
    monkeypatch.setattr(slang_cuda.glob, "glob", lambda pattern: searches.append(pattern) or real_glob(pattern))

    slang_cuda.find_cuda_nvrtc()
    first = len(searches)
    slang_cuda.find_cuda_nvrtc()

    assert first > 0
    assert len(searches) == first


class _FakeNvrtc:
    def __init__(self, major: int, minor: int, status: int = 0) -> None:
        self._version = (major, minor)
        self._status = status

    def nvrtcVersion(self, major, minor) -> int:
        major._obj.value, minor._obj.value = self._version
        return self._status


def test_nvrtc_version_reads_nvrtc_itself(monkeypatch) -> None:
    monkeypatch.setattr(slang_cuda.ctypes, "CDLL", lambda name: _FakeNvrtc(12, 1))

    assert slang_cuda.nvrtc_version("/usr/local/cuda-12.1/lib64/libnvrtc.so") == (12, 1)


def test_nvrtc_version_is_none_when_nvrtc_reports_an_error(monkeypatch) -> None:
    monkeypatch.setattr(slang_cuda.ctypes, "CDLL", lambda name: _FakeNvrtc(12, 1, status=1))

    assert slang_cuda.nvrtc_version("libnvrtc.so") is None


def test_nvrtc_version_is_none_when_the_library_cannot_load() -> None:
    assert slang_cuda.nvrtc_version("libnvrtc.so") is None


def test_the_device_error_names_the_nvrtc_that_was_found(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(slang_cuda, "_CUDA_INSTALL_GLOB", str(tmp_path / "cuda*"))
    library = _library(tmp_path / "cuda-11.8")
    monkeypatch.setattr(slang_cuda.ctypes, "CDLL", lambda name: _FakeNvrtc(11, 8))

    message = slang_cuda.cuda_device_error(RuntimeError("Failed to create device!\n/project/src/device.cpp:280"))

    assert "(Failed to create device!)" in message
    assert f"NVRTC 11.8 at {library}" in message
    assert "--fluoro-device vulkan" in message
    assert "device.cpp" not in message


def test_the_device_error_says_when_no_nvrtc_was_found() -> None:
    message = slang_cuda.cuda_device_error(RuntimeError("Failed to create device!"))

    assert "no libnvrtc.so was found" in message
    assert "TROUBLESHOOTING.md" in message


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_the_device_check_asks_slang_in_a_child_process(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(slang_cuda.subprocess, "run", lambda command, **kwargs: calls.append(command) or _completed(0))

    assert slang_cuda.check_slang_cuda_device() is None
    assert calls[0][0] == sys.executable
    assert "slangpy.create_device(slangpy.DeviceType.cuda)" in calls[0][-1]


def test_the_device_check_returns_the_childs_diagnosis(monkeypatch) -> None:
    stdout = "[WARN] slang log noise\nSlang could not create a CUDA device (Failed to create device!); ..."
    monkeypatch.setattr(slang_cuda.subprocess, "run", lambda command, **kwargs: _completed(1, stdout))

    assert slang_cuda.check_slang_cuda_device() == stdout.splitlines()[-1]


def test_the_device_check_falls_back_to_stderr(monkeypatch) -> None:
    monkeypatch.setattr(
        slang_cuda.subprocess, "run", lambda command, **kwargs: _completed(1, stderr="ModuleNotFoundError: slangpy")
    )

    assert slang_cuda.check_slang_cuda_device() == "ModuleNotFoundError: slangpy"


def test_the_device_check_reports_only_the_last_line_of_a_traceback(monkeypatch) -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 2, in <module>\n'
        "ModuleNotFoundError: No module named 'slangpy'\n"
    )
    monkeypatch.setattr(slang_cuda.subprocess, "run", lambda command, **kwargs: _completed(1, stderr=stderr))

    assert slang_cuda.check_slang_cuda_device() == "ModuleNotFoundError: No module named 'slangpy'"


def test_the_device_check_reports_a_silent_failure(monkeypatch) -> None:
    monkeypatch.setattr(slang_cuda.subprocess, "run", lambda command, **kwargs: _completed(-11))

    assert slang_cuda.check_slang_cuda_device() == "the Slang CUDA device check exited with status -11"


def test_a_device_check_that_hangs_reports_the_timeout_instead_of_raising(monkeypatch) -> None:
    def hang(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(slang_cuda.subprocess, "run", hang)

    assert slang_cuda.check_slang_cuda_device(timeout_s=5) == "creating a Slang CUDA device did not finish within 5 s"


def test_the_probe_script_compiles() -> None:
    compile(slang_cuda._PROBE, "<probe>", "exec")
