#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"
[[ $# -gt 0 ]] || set -- --help

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

arch="$(uname -m)"
if [[ "$arch" == "aarch64" ]]; then
  sys_libgomp="$(ls /lib/*/libgomp.so.1 2>/dev/null | head -1 || true)"
  if [[ -n "$sys_libgomp" ]]; then
    export LD_PRELOAD="${sys_libgomp}${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2000000}"
else
  libgomp_path="$(env -u VIRTUAL_ENV uv run --no-sync python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib" / "libgomp.so.1")' 2>/dev/null || true)"
  if [[ -n "$libgomp_path" && -e "$libgomp_path" ]]; then
    export LD_PRELOAD="${libgomp_path}${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

exec env -u VIRTUAL_ENV uv run i4h-agentic-arena "$@"
