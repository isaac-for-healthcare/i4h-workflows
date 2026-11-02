# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared ARM OpenMP setup for Kit launchers; imports only the standard library."""

import argparse
import os
import platform
import re
import shlex
from pathlib import Path

TLS_KEY = "glibc.rtld.optional_static_tls"
TLS_DEFAULT = f"{TLS_KEY}=2000000"


def _system_libgomp() -> Path | None:
    candidates = sorted(Path("/lib").glob("*/libgomp.so.1"))
    return candidates[0] if candidates else None


def apply_arm_openmp_env(env: dict[str, str]) -> None:
    """Put system OpenMP first and supply only the missing TLS tunable on ARM."""
    if platform.machine() != "aarch64":
        return
    library = _system_libgomp()
    if library is not None:
        path = str(library)
        preloads = re.split(r"[:\s]+", env.get("LD_PRELOAD", ""))
        env["LD_PRELOAD"] = ":".join([path, *(entry for entry in preloads if entry and entry != path)])
    tunables = env.get("GLIBC_TUNABLES", "")
    if not any(entry.partition("=")[0] == TLS_KEY for entry in tunables.split(":")):
        env["GLIBC_TUNABLES"] = ":".join(filter(None, (tunables, TLS_DEFAULT)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", action="store_true", required=True)
    parser.parse_args()
    env = os.environ.copy()
    apply_arm_openmp_env(env)
    for key in ("LD_PRELOAD", "GLIBC_TUNABLES"):
        if key in env:
            print(f"export {key}={shlex.quote(env[key])}")


if __name__ == "__main__":
    main()
