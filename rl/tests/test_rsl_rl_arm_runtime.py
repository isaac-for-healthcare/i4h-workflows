# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARM Kit launches need the same OpenMP/TLS environment as run.sh."""

import platform
from pathlib import Path

import pytest

from i4h_rl.backends import rsl_rl
from i4h_rl.profile import RLProfile

ROOT = Path(__file__).resolve().parents[2]
LIBGOMP = Path("/lib/aarch64-linux-gnu/libgomp.so.1")


@pytest.mark.parametrize("existing", ["", "/custom/lib.so"])
def test_arm_runtime_preloads_system_openmp(monkeypatch, existing):
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(Path, "glob", lambda _self, _pattern: iter([LIBGOMP]))
    monkeypatch.setenv("LD_PRELOAD", existing)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    env = rsl_rl._runtime_env(ROOT, RLProfile.load(ROOT / "rl/profiles/ultrasound_probe_reach.yaml"))
    assert env["LD_PRELOAD"] == str(LIBGOMP) + (f":{existing}" if existing else "")
    assert env["GLIBC_TUNABLES"] == "glibc.rtld.optional_static_tls=2000000"
    assert env["OMNI_KIT_ACCEPT_EULA"]


def test_arm_runtime_preserves_user_tls_setting(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(Path, "glob", lambda _self, _pattern: iter([LIBGOMP]))
    monkeypatch.setenv("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=4000000")
    env = rsl_rl._runtime_env(ROOT, RLProfile.load(ROOT / "rl/profiles/ultrasound_probe_reach.yaml"))
    assert env["GLIBC_TUNABLES"] == "glibc.rtld.optional_static_tls=4000000"


def test_x86_runtime_preserves_preload_and_tls(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("LD_PRELOAD", "/custom/lib.so")
    monkeypatch.setenv("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=4000000")
    env = rsl_rl._runtime_env(ROOT, RLProfile.load(ROOT / "rl/profiles/ultrasound_probe_reach.yaml"))
    assert env["LD_PRELOAD"] == "/custom/lib.so"
    assert env["GLIBC_TUNABLES"] == "glibc.rtld.optional_static_tls=4000000"
