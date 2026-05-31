#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

image="${1:-cosmos-transfer-2.5}"
repo="${COSMOS_TRANSFER_REPO:-https://github.com/nvidia-cosmos/cosmos-transfer2.5.git}"
python_version="${COSMOS_TRANSFER_PYTHON_VERSION:-3.10}"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

git clone --depth 1 "$repo" "$workdir/cosmos-transfer2.5"
printf '%s\n' "$python_version" > "$workdir/cosmos-transfer2.5/.python-version"
docker build --ulimit nofile=131071:131071 --build-arg STANDALONE=true -t "$image" "$workdir/cosmos-transfer2.5"
