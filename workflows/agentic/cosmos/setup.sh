#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
uv sync

if ! command -v docker >/dev/null 2>&1; then
  echo "[setup] docker not found; local export/import still works, Cosmos execution will be manual."
  exit 0
fi
if ! command -v git >/dev/null 2>&1; then
  echo "[setup] git is required to build the Cosmos Docker image" >&2
  exit 1
fi

image="${COSMOS_DOCKER_IMAGE:-cosmos-transfer-2.5}"
if docker image inspect "$image" >/dev/null 2>&1 && \
   docker run --rm --entrypoint /bin/sh "$image" -lc 'test -f /workspace/bin/entrypoint.sh && test -f /workspace/examples/inference.py' >/dev/null 2>&1; then
  echo "[setup] Cosmos Docker image ready: $image"
else
  echo "[setup] building standalone Cosmos Docker image: $image"
  scripts/build-image.sh "$image"
fi
