#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

inputs=(setup.sh third_party/setup.sh)
while IFS= read -r -d '' path; do
  inputs+=("${path#./}")
done < <(
  find common engine workflows arena rl tasks tools \
    -type f \( -name pyproject.toml -o -name uv.lock \) -print0 \
    | sort -z
)
while IFS= read -r -d '' path; do
  inputs+=("${path#./}")
done < <(find third_party -maxdepth 1 -type f -name '*.patch' -print0 | sort -z)

sha256sum "${inputs[@]}" | sha256sum | cut -d ' ' -f 1
