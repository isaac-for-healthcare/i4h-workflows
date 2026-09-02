#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Serve Cosmos 3 for transfer-control video-to-video.
#
# The prebuilt cosmos3-generator NIM does not expose transfer controls — only
# text-to-video and image-to-video — so restyling a recorded camera stream goes
# through the vLLM-Omni build instead.
#
#   ./tools/cosmos/scripts/serve.sh              # foreground, port 8000
#   COSMOS3_PORT=8010 ./tools/cosmos/scripts/serve.sh
set -euo pipefail

IMAGE="${COSMOS3_IMAGE:-vllm/vllm-omni:cosmos3}"
MODEL="${COSMOS3_MODEL:-nvidia/Cosmos3-Super}"
PORT="${COSMOS3_PORT:-8000}"
TP="${COSMOS3_TENSOR_PARALLEL:-1}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
WORKDIR="${COSMOS3_WORKDIR:-$PWD}"

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }
mkdir -p "$HF_HOME"

exec docker run --runtime nvidia --gpus all --rm --ipc=host \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${WORKDIR}:/workspace" \
  -p "${PORT}:8000" \
  -w /workspace \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --tensor-parallel-size "$TP" \
    --enable-layerwise-offload \
    --port 8000 \
    --init-timeout 1800
