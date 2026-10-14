# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARM Kit launches share idempotent OpenMP/TLS setup across both backends."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))

from i4h_rl import runtime  # noqa: E402
from i4h_rl.backends import rlinf, rsl_rl  # noqa: E402
from i4h_rl.profile import RLProfile  # noqa: E402

LIBGOMP = Path("/lib/aarch64-linux-gnu/libgomp.so.1")


@pytest.fixture
def arm(monkeypatch):
    monkeypatch.setattr(runtime.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(runtime, "_system_libgomp", lambda: LIBGOMP)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)


@pytest.fixture(params=["rsl_rl", "rlinf"])
def launch_env(request, monkeypatch):
    def build(values):
        with monkeypatch.context() as context:
            for key, value in values.items():
                context.setenv(key, value)
            if request.param == "rsl_rl":
                profile = RLProfile.load(ROOT / "rl/profiles/ultrasound_probe_reach.yaml")
                return rsl_rl._runtime_env(ROOT, profile)
            original = os.environ.copy()
            result = rlinf._sim_env(ROOT, original, gpu="1")
            assert original == os.environ.copy(), "simulator setup must not modify the model environment"
            assert result["CUDA_VISIBLE_DEVICES"] == "1"
            return result

    return build


@pytest.mark.parametrize(
    "existing", ["", "/custom/lib.so", str(LIBGOMP), f"/custom/lib.so:{LIBGOMP}:{LIBGOMP}", f"/custom/lib.so {LIBGOMP}"]
)
def test_arm_preloads_system_openmp_once_and_first(arm, launch_env, existing):
    env = launch_env({"LD_PRELOAD": existing})
    expected = str(LIBGOMP) + (":/custom/lib.so" if "/custom/lib.so" in existing else "")
    assert env["LD_PRELOAD"] == expected
    assert env["GLIBC_TUNABLES"] == runtime.TLS_DEFAULT
    runtime.apply_arm_openmp_env(env)
    assert env["LD_PRELOAD"] == expected
    assert env["GLIBC_TUNABLES"] == runtime.TLS_DEFAULT


@pytest.mark.parametrize(
    "existing, expected",
    [
        ("", runtime.TLS_DEFAULT),
        ("glibc.malloc.arena_max=4", "glibc.malloc.arena_max=4:" + runtime.TLS_DEFAULT),
        ("glibc.rtld.optional_static_tls=4000000", "glibc.rtld.optional_static_tls=4000000"),
        (
            "glibc.malloc.arena_max=4:glibc.rtld.optional_static_tls=4000000",
            "glibc.malloc.arena_max=4:glibc.rtld.optional_static_tls=4000000",
        ),
        ("glibc.rtld.optional_static_tls_extra=42", "glibc.rtld.optional_static_tls_extra=42:" + runtime.TLS_DEFAULT),
    ],
)
def test_arm_preserves_individual_tunables(arm, launch_env, existing, expected):
    env = launch_env({"GLIBC_TUNABLES": existing})
    assert env["GLIBC_TUNABLES"] == expected
    runtime.apply_arm_openmp_env(env)
    assert env["GLIBC_TUNABLES"] == expected


def test_missing_system_openmp_still_adds_tls(arm, launch_env, monkeypatch):
    monkeypatch.setattr(runtime, "_system_libgomp", lambda: None)
    env = launch_env({"LD_PRELOAD": "/custom/lib.so"})
    assert env["LD_PRELOAD"] == "/custom/lib.so"
    assert env["GLIBC_TUNABLES"] == runtime.TLS_DEFAULT


def test_missing_system_openmp_does_not_create_preload(arm, monkeypatch):
    monkeypatch.setattr(runtime, "_system_libgomp", lambda: None)
    env = {}
    runtime.apply_arm_openmp_env(env)
    assert "LD_PRELOAD" not in env
    assert env["GLIBC_TUNABLES"] == runtime.TLS_DEFAULT


def test_x86_preserves_preload_and_tls(launch_env, monkeypatch):
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime, "_system_libgomp", lambda: pytest.fail("x86 must not look up system OpenMP"))
    values = {"LD_PRELOAD": "/custom/lib.so", "GLIBC_TUNABLES": "glibc.malloc.arena_max=4"}
    env = launch_env(values)
    assert all(env[key] == value for key, value in values.items())


def test_run_sh_uses_shared_helper_and_safe_exports(arm, monkeypatch, capsys, tmp_path):
    # Exercise the real shell branch and helper output, preserving shell metacharacters.
    marker = tmp_path / "unexpected-expansion"
    tunable = f"glibc.malloc.arena_max=4;$(touch {marker})"
    monkeypatch.setenv("GLIBC_TUNABLES", tunable)
    monkeypatch.setattr(sys, "argv", ["runtime.py", "--shell"])
    runtime.main()
    exports = capsys.readouterr().out
    script = (ROOT / "run.sh").read_text()
    branch = script.split('if [ "$(uname -m)" = "aarch64" ]; then', 1)[1].split("\nelse", 1)[0]
    assert '"$ROOT/rl/i4h_rl/runtime.py" --shell' in branch
    mock = "light() { printf '%s' " + shlex.quote(exports) + "; }\n"
    result = subprocess.run(
        ["bash", "-eu", "-c", mock + branch + '\nprintf \'%s\n%s\n\' "$LD_PRELOAD" "$GLIBC_TUNABLES"'],
        env={"PATH": os.environ["PATH"], "ROOT": str(ROOT)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == [str(LIBGOMP), tunable + ":" + runtime.TLS_DEFAULT]
    assert not marker.exists()
